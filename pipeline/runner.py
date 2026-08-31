from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.files import File
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from matches.models import (
    AnalysisArtifact,
    AnalysisRun,
    Event,
    Match,
    MatchPeriod,
    PlayerMatchStat,
    PossessionSegment,
    TeamMatchStat,
    Track,
)

from .ball_in_play import BallInPlayEngine
from .camera import CameraStabilizer, PitchProjector
from .clips import ClipPlanner, render_clip
from .events import EventEngine
from .periods import PeriodDetector
from .providers.base import build_provider
from .quality import VideoQualityAnalyzer
from .stats import StatsAggregator, blank_metrics
from .types import EventCandidate, FrameAnalysis, ObjectRole, PossessionSample, PossessionSpan
from .video import iter_frames, probe_video


logger = logging.getLogger(__name__)


class AnalysisCancelled(RuntimeError):
    pass


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _save_json_artifact(
    run: AnalysisRun,
    kind: str,
    filename: str,
    payload: Any,
    *,
    metadata: dict | None = None,
) -> AnalysisArtifact:
    artifact = AnalysisArtifact(analysis_run=run, kind=kind, metadata=metadata or {})
    artifact.file.save(filename, ContentFile(_json_bytes(payload)), save=True)
    return artifact


def _save_local_artifact(
    run: AnalysisRun,
    kind: str,
    filename: str,
    local_path: Path,
    *,
    metadata: dict | None = None,
) -> AnalysisArtifact:
    artifact = AnalysisArtifact(analysis_run=run, kind=kind, metadata=metadata or {})
    with local_path.open("rb") as handle:
        artifact.file.save(filename, File(handle), save=True)
    return artifact


class MatchAnalysisRunner:
    """Orchestrate one complete, reviewable match-analysis run.

    Video time is kept as the immutable source clock. Match time is derived from a
    confirmed or suggested ``MatchPeriod``. This prevents a pre-match sequence,
    half-time or replay from corrupting the football clock.
    """

    def __init__(self, run: AnalysisRun):
        self.run = AnalysisRun.objects.select_related(
            "match",
            "match__home_team",
            "match__away_team",
            "match__video",
        ).get(pk=run.pk)
        self.match = self.run.match
        self.video = self.match.video
        self.config = dict(self.run.config or {})
        self.team_by_key = {
            "home": self.match.home_team,
            "away": self.match.away_team,
        }
        self.last_progress = -1

    def execute(self) -> AnalysisRun:
        try:
            self._begin()
            metadata = self._probe()
            quality = self._quality(metadata)
            periods = self._periods(quality.signals, metadata.duration_ms)
            if self.analysis_mode == "prepare":
                result = self._empty_result()
                report = self._report(metadata, quality, periods, result, self._empty_clips())
                self._finish(periods, quality, result, report)
                return self.run
            result = self._track(periods, metadata)
            persisted = self._persist(periods, result)
            clips = self._clips(metadata, persisted["events"], result["events"])
            report = self._report(metadata, quality, periods, result, clips)
            self._finish(periods, quality, result, report)
        except AnalysisCancelled:
            self._cancel()
        except Exception as exc:
            logger.exception("Analysis %s failed", self.run.pk)
            self._fail(exc)
            raise
        return self.run

    @property
    def analysis_mode(self) -> str:
        mode = str(self.config.get("analysis_mode", "full"))
        return mode if mode in {"prepare", "sample", "full"} else "full"

    def _empty_result(self) -> dict:
        return {
            "analysis_mode": self.analysis_mode,
            "backend": str(self.config.get("backend", "heuristic")),
            "tracking_fps": 0.0,
            "samples": [],
            "spans": [],
            "events": [],
            "tracks": {},
            "camera": {
                "frames": 0,
                "reliable_frames": 0,
                "resets": 0,
                "mean_inlier_ratio": 0.0,
                "reliable_ratio": 0.0,
            },
            "windows": [],
            "diagnostics": {},
            "previews": [],
        }

    @staticmethod
    def _empty_clips() -> dict:
        return {"enabled": False, "planned": 0, "rendered": 0, "errors": []}

    def _begin(self) -> None:
        self.run.status = AnalysisRun.Status.PROCESSING
        self.run.current_stage = AnalysisRun.Stage.PROBE
        self.run.progress = 1
        self.run.started_at = self.run.started_at or timezone.now()
        self.run.error_message = ""
        self.run.save(
            update_fields=[
                "status",
                "current_stage",
                "progress",
                "started_at",
                "error_message",
            ]
        )
        self.match.status = Match.Status.PROCESSING
        self.match.save(update_fields=["status", "updated_at"])

    def _probe(self):
        self._stage(AnalysisRun.Stage.PROBE, 3)
        metadata = probe_video(self.video.file.path)
        self.video.duration_ms = metadata.duration_ms
        self.video.fps = metadata.fps
        self.video.width = metadata.width
        self.video.height = metadata.height
        self.video.codec = metadata.codec
        self.video.save(
            update_fields=["duration_ms", "fps", "width", "height", "codec"]
        )
        return metadata

    def _quality(self, metadata):
        self._stage(AnalysisRun.Stage.QUALITY, 7)
        analyzer = VideoQualityAnalyzer(
            sample_seconds=float(self.config.get("sample_seconds", 1.0)),
            max_samples=int(self.config.get("quality_max_samples", 360)),
        )
        report = analyzer.analyze(metadata, progress_callback=self._quality_progress)
        self._stage(AnalysisRun.Stage.QUALITY, 15)
        self.video.quality_grade = report.grade
        self.video.quality_score = report.score
        self.video.quality_metrics = report.metrics
        self.video.save(update_fields=["quality_grade", "quality_score", "quality_metrics"])
        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.QUALITY,
            f"quality-{self.run.pk}.json",
            report.to_dict(),
            metadata={"score": report.score, "grade": report.grade},
        )
        return report

    def _quality_progress(self, completed: int, total: int) -> None:
        self._check_cancelled()
        progress = 7 + int(8 * completed / max(total, 1))
        self._stage(AnalysisRun.Stage.QUALITY, min(progress, 15))

    def _periods(self, signals, duration_ms: int) -> list[MatchPeriod]:
        self._stage(AnalysisRun.Stage.PERIODS, 16)
        confirmed = list(self.match.periods.filter(confirmed=True).order_by("number"))
        if len(confirmed) == 2:
            periods = confirmed
            payload = {
                "source": "confirmed",
                "requires_review": False,
                "periods": [self._period_payload(period) for period in periods],
            }
        else:
            detection = PeriodDetector().detect(signals, duration_ms)
            periods = []
            for index, span in enumerate(detection.periods, start=1):
                clock_start = 0 if index == 1 else 2_700_000
                period, _ = MatchPeriod.objects.update_or_create(
                    match=self.match,
                    number=index,
                    defaults={
                        "label": "1re mi-temps" if index == 1 else "2e mi-temps",
                        "video_start_ms": max(0, span.start_ms),
                        "video_end_ms": min(duration_ms, span.end_ms),
                        "match_clock_start_ms": clock_start,
                        "match_clock_end_ms": clock_start + max(0, span.duration_ms),
                        "source": MatchPeriod.Source.AUTO,
                        "confidence": span.confidence,
                        "confirmed": False,
                    },
                )
                periods.append(period)
            payload = detection.to_dict()
        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.PERIODS,
            f"periods-{self.run.pk}.json",
            payload,
            metadata={"confirmed": all(period.confirmed for period in periods)},
        )
        if len(periods) != 2:
            raise RuntimeError("Deux mi-temps valides sont nécessaires pour analyser le match.")
        return periods

    def _track(self, periods: list[MatchPeriod], metadata) -> dict:
        self._stage(AnalysisRun.Stage.TRACKING, 22)
        backend = str(self.config.get("backend", "heuristic"))
        device = str(self.config.get("device", "cpu"))
        tracking_fps = max(0.5, float(self.config.get("tracking_fps", 10.0)))
        windows = self._tracking_windows(periods)
        tracking_duration = sum(window["end_ms"] - window["start_ms"] for window in windows)
        estimated_frames = math.ceil(tracking_duration / 1000.0 * tracking_fps)
        operation = "Test rapide" if self.analysis_mode == "sample" else "Tracking"
        self._save_live_progress(
            22,
            {
                "stage": "tracking",
                "stage_progress": 0.0,
                "processed_video_ms": 0,
                "total_video_ms": tracking_duration,
                "frames_processed": 0,
                "frames_total_estimate": estimated_frames,
                "elapsed_seconds": 0,
                "eta_seconds": None,
                "speed_x": 0.0,
                "backend": backend,
                "device": device,
                "tracking_fps": tracking_fps,
                "label": self._tracking_label(
                    backend=backend,
                    device=device,
                    stage_progress=0.0,
                    processed_ms=0,
                    total_ms=tracking_duration,
                    frames_processed=0,
                    frames_total=estimated_frames,
                    speed_x=0.0,
                    eta_seconds=None,
                    initializing=True,
                    operation=operation,
                ),
            },
        )
        provider = build_provider(
            backend,
            model_path=self.config.get("yolo_model_path", ""),
            device=device,
            confidence=float(self.config.get("yolo_confidence", 0.30)),
            image_size=int(self.config.get("yolo_image_size", 1280)),
            team_colors={
                "home": self.match.home_team.primary_color,
                "away": self.match.away_team.primary_color,
            },
        )
        tracking_started_at = time.monotonic()
        last_live_update = 0.0
        frames_processed = 0
        processed_ms = 0
        all_samples: list[tuple[MatchPeriod, PossessionSample]] = []
        all_spans: list[tuple[MatchPeriod, PossessionSpan]] = []
        all_events: list[tuple[MatchPeriod, EventCandidate]] = []
        track_summaries: dict[str, dict] = {}
        camera_summary = {
            "frames": 0,
            "reliable_frames": 0,
            "resets": 0,
            "mean_inlier_ratio": 0.0,
        }
        camera_inliers: list[float] = []
        diagnostic_counts: Counter = Counter()
        team_observations: Counter = Counter()
        preview_artifacts: list[dict] = []

        with tempfile.TemporaryDirectory(prefix="football-tracking-") as temp_dir:
            temp_path = Path(temp_dir)
            tracking_path = Path(temp_dir) / "tracking.ndjson"
            with tracking_path.open("w", encoding="utf-8") as tracking_file:
                for window in windows:
                    self._check_cancelled()
                    period = window["period"]
                    window_start_ms = window["start_ms"]
                    window_end_ms = window["end_ms"]
                    provider.reset()
                    camera = CameraStabilizer()
                    projector = self._projector(period)
                    ball_engine = BallInPlayEngine()
                    period_samples: list[PossessionSample] = []
                    period_prefix = f"p{period.number}-w{window['index']}-"
                    preview_saved = False
                    preview_target_ms = window_start_ms + (window_end_ms - window_start_ms) // 2
                    for timestamp_ms, frame in iter_frames(
                        metadata.path,
                        start_ms=window_start_ms,
                        end_ms=window_end_ms,
                        target_fps=tracking_fps,
                    ):
                        analysis = provider.analyze_frame(frame, timestamp_ms)
                        frames_processed += 1
                        motion = camera.update(frame, scene_cut=analysis.scene_cut)
                        analysis.camera = motion.to_dict()
                        camera_summary["frames"] += 1
                        camera_summary["reliable_frames"] += int(motion.reliable)
                        camera_summary["resets"] += int(motion.reset)
                        camera_inliers.append(motion.inlier_ratio)
                        self._normalize_objects(
                            analysis,
                            camera,
                            projector,
                            period_prefix,
                            track_summaries,
                        )
                        sample = ball_engine.observe(analysis)
                        period_samples.append(sample)
                        all_samples.append((period, sample))
                        diagnostic_counts["frames"] += 1
                        diagnostic_counts["athlete_observations"] += len(analysis.athletes)
                        diagnostic_counts["ball_visible_frames"] += int(analysis.ball is not None)
                        diagnostic_counts["field_frames"] += int(
                            analysis.field_score >= 0.14
                            and not analysis.scene_cut
                            and analysis.replay_probability < 0.65
                        )
                        diagnostic_counts[f"state_{str(sample.state)}"] += 1
                        for athlete in analysis.athletes:
                            team_observations[athlete.team_key or "unknown"] += 1
                        if (
                            self.analysis_mode == "sample"
                            and not preview_saved
                            and timestamp_ms >= preview_target_ms
                        ):
                            preview_path = temp_path / (
                                f"sample-p{period.number}-w{window['index']}.jpg"
                            )
                            self._write_sample_preview(frame, analysis, preview_path)
                            artifact = _save_local_artifact(
                                self.run,
                                AnalysisArtifact.Kind.ANNOTATED_VIDEO,
                                f"sample-{self.run.pk}-p{period.number}-w{window['index']}.jpg",
                                preview_path,
                                metadata={
                                    "artifact_type": "sample_preview",
                                    "period": period.number,
                                    "window": window["index"],
                                    "video_time_ms": timestamp_ms,
                                },
                            )
                            preview_artifacts.append(
                                {
                                    "url": artifact.file.url,
                                    "period": period.number,
                                    "window": window["index"],
                                    "video_time_ms": timestamp_ms,
                                }
                            )
                            preview_saved = True
                        tracking_file.write(
                            json.dumps(
                                {
                                    "period": period.number,
                                    "window": window["index"],
                                    "match_time_ms": self._match_time(period, timestamp_ms),
                                    "frame": analysis.to_dict(),
                                    "possession": sample.to_dict(),
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )
                        current_processed = processed_ms + timestamp_ms - window_start_ms
                        current_processed = max(0, min(tracking_duration, current_processed))
                        now = time.monotonic()
                        progress_ratio = current_processed / max(tracking_duration, 1)
                        percent = min(68, 22 + int(46 * progress_ratio))
                        if now - last_live_update >= 2.0 or percent != self.last_progress:
                            elapsed_seconds = max(0.001, now - tracking_started_at)
                            processed_seconds = current_processed / 1000.0
                            speed_x = processed_seconds / elapsed_seconds
                            remaining_seconds = max(
                                0.0,
                                (tracking_duration - current_processed) / 1000.0,
                            )
                            eta_seconds = (
                                remaining_seconds / speed_x if speed_x > 0.0001 else None
                            )
                            detail = {
                                "stage": "tracking",
                                "stage_progress": round(progress_ratio * 100.0, 2),
                                "processed_video_ms": current_processed,
                                "total_video_ms": tracking_duration,
                                "frames_processed": frames_processed,
                                "frames_total_estimate": estimated_frames,
                                "elapsed_seconds": round(elapsed_seconds, 1),
                                "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
                                "speed_x": round(speed_x, 3),
                                "backend": backend,
                                "device": device,
                                "tracking_fps": tracking_fps,
                            }
                            detail["label"] = self._tracking_label(
                                backend=backend,
                                device=device,
                                stage_progress=detail["stage_progress"],
                                processed_ms=current_processed,
                                total_ms=tracking_duration,
                                frames_processed=frames_processed,
                                frames_total=estimated_frames,
                                speed_x=speed_x,
                                eta_seconds=eta_seconds,
                                operation=operation,
                            )
                            self._save_live_progress(percent, detail)
                            last_live_update = now
                    processed_ms += max(0, window_end_ms - window_start_ms)
                    spans = ball_engine.compress(
                        period_samples,
                        max_gap_ms=max(1_000, int(2_500 / tracking_fps)),
                    )
                    for span in spans:
                        span.end_ms = min(window_end_ms, span.end_ms)
                    all_spans.extend((period, span) for span in spans)
                    if self.analysis_mode == "full":
                        candidates = EventEngine().detect(period_samples)
                        all_events.extend((period, candidate) for candidate in candidates)

            _save_local_artifact(
                self.run,
                AnalysisArtifact.Kind.TRACKING,
                f"tracking-{self.run.pk}.ndjson",
                tracking_path,
                metadata={
                    "backend": backend,
                    "tracking_fps": tracking_fps,
                    "coordinate_systems": ["image_normalized", "pitch_meters"],
                    "analysis_mode": self.analysis_mode,
                    "windows": [self._window_payload(window) for window in windows],
                },
            )

        camera_summary["reliable_ratio"] = round(
            camera_summary["reliable_frames"] / max(camera_summary["frames"], 1), 4
        )
        camera_summary["mean_inlier_ratio"] = round(
            sum(camera_inliers) / max(len(camera_inliers), 1), 4
        )
        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.CAMERA,
            f"camera-{self.run.pk}.json",
            camera_summary,
        )
        diagnostics = self._tracking_diagnostics(
            diagnostic_counts,
            team_observations,
            track_count=len(track_summaries),
            tracking_duration_ms=tracking_duration,
        )
        return {
            "analysis_mode": self.analysis_mode,
            "backend": backend,
            "tracking_fps": tracking_fps,
            "samples": all_samples,
            "spans": all_spans,
            "events": all_events,
            "tracks": track_summaries,
            "camera": camera_summary,
            "windows": [self._window_payload(window) for window in windows],
            "diagnostics": diagnostics,
            "previews": preview_artifacts,
        }

    @staticmethod
    def _write_sample_preview(frame, analysis: FrameAnalysis, output_path: Path) -> None:
        import cv2

        preview = frame.copy()
        colors = {
            "home": (70, 220, 120),
            "away": (70, 130, 255),
            "unknown": (190, 190, 190),
            "ball": (255, 255, 255),
        }
        for obj in analysis.objects:
            x1, y1, x2, y2 = (int(value) for value in obj.bbox_xyxy)
            is_ball = obj.role == ObjectRole.BALL
            color = (
                colors["ball"]
                if is_ball
                else colors.get(obj.team_key or "unknown", colors["unknown"])
            )
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3 if is_ball else 2)
            label = "BALL" if is_ball else f"{obj.team_key or 'unknown'} {obj.track_id}"
            cv2.putText(
                preview,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        max_width = 1280
        if preview.shape[1] > max_width:
            scale = max_width / preview.shape[1]
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError("Impossible d’écrire l’aperçu annoté du test rapide.")

    def _tracking_windows(self, periods: list[MatchPeriod]) -> list[dict]:
        if self.analysis_mode != "sample":
            return [
                {
                    "period": period,
                    "index": 1,
                    "start_ms": period.video_start_ms,
                    "end_ms": period.video_end_ms,
                }
                for period in periods
            ]

        window_ms = max(
            10_000,
            int(float(self.config.get("sample_window_seconds", 30)) * 1_000),
        )
        windows_per_half = max(
            1,
            min(3, int(self.config.get("sample_windows_per_half", 2))),
        )
        positions = (
            [0.50]
            if windows_per_half == 1
            else ([0.25, 0.70] if windows_per_half == 2 else [0.18, 0.50, 0.78])
        )
        windows: list[dict] = []
        for period in periods:
            duration_ms = max(0, period.video_end_ms - period.video_start_ms)
            bounded_window_ms = min(window_ms, duration_ms)
            available_ms = max(0, duration_ms - bounded_window_ms)
            for index, position in enumerate(positions, start=1):
                start_ms = period.video_start_ms + int(available_ms * position)
                windows.append(
                    {
                        "period": period,
                        "index": index,
                        "start_ms": start_ms,
                        "end_ms": start_ms + bounded_window_ms,
                    }
                )
        return windows

    @staticmethod
    def _window_payload(window: dict) -> dict:
        return {
            "period": window["period"].number,
            "index": window["index"],
            "start_ms": window["start_ms"],
            "end_ms": window["end_ms"],
            "duration_ms": max(0, window["end_ms"] - window["start_ms"]),
        }

    @staticmethod
    def _tracking_diagnostics(
        counts: Counter,
        team_observations: Counter,
        *,
        track_count: int,
        tracking_duration_ms: int,
    ) -> dict:
        frames = max(int(counts["frames"]), 0)
        duration_minutes = max(tracking_duration_ms / 60_000, 1 / 60)
        known_team_observations = int(team_observations["home"] + team_observations["away"])
        home_share_pct = (
            100.0 * team_observations["home"] / known_team_observations
            if known_team_observations
            else 0.0
        )
        away_share_pct = (
            100.0 * team_observations["away"] / known_team_observations
            if known_team_observations
            else 0.0
        )
        playable_frames = sum(
            int(counts[f"state_{state}"])
            for state in ("controlled", "contested", "loose")
        )
        diagnostics = {
            "frames_analyzed": frames,
            "duration_seconds": round(tracking_duration_ms / 1_000, 1),
            "average_athletes_per_frame": round(
                counts["athlete_observations"] / max(frames, 1), 2
            ),
            "ball_visibility_pct": round(
                100.0 * counts["ball_visible_frames"] / max(frames, 1), 2
            ),
            "field_live_pct": round(100.0 * counts["field_frames"] / max(frames, 1), 2),
            "playable_candidate_pct": round(100.0 * playable_frames / max(frames, 1), 2),
            "tracks": track_count,
            "tracks_per_minute": round(track_count / duration_minutes, 2),
            "home_team_share_pct": round(home_share_pct, 2),
            "away_team_share_pct": round(away_share_pct, 2),
            "unknown_team_observations": int(team_observations["unknown"]),
            "issues": [],
        }
        failures: list[str] = []
        warnings: list[str] = []
        if frames < 50:
            failures.append("Trop peu d’images ont été analysées.")
        if diagnostics["average_athletes_per_frame"] < 6:
            failures.append("Moins de 6 joueurs sont détectés en moyenne par image.")
        if diagnostics["ball_visibility_pct"] < 10:
            failures.append("Le ballon est visible sur moins de 10 % des images.")
        if diagnostics["playable_candidate_pct"] < 10:
            failures.append("Le jeu effectif est reconnu sur moins de 10 % des images.")
        if known_team_observations < max(50, frames):
            warnings.append("Pas assez de joueurs ont une équipe reconnue.")
        elif min(home_share_pct, away_share_pct) < 15:
            failures.append("La séparation des deux équipes est fortement déséquilibrée.")
        if diagnostics["tracks_per_minute"] > 80:
            failures.append("Les identités de piste se fragmentent beaucoup trop vite.")
        elif diagnostics["tracks_per_minute"] > 50:
            warnings.append("La continuité des pistes est encore fragile.")
        diagnostics["issues"] = failures + warnings
        diagnostics["verdict"] = "fail" if failures else ("warning" if warnings else "pass")
        return diagnostics

    def _save_live_progress(self, progress: int, detail: dict) -> None:
        self._check_cancelled()
        metrics = dict(self.run.metrics or {})
        metrics["live_progress"] = detail
        self.run.current_stage = AnalysisRun.Stage.TRACKING
        self.run.progress = max(0, min(100, progress))
        self.run.metrics = metrics
        self.run.save(update_fields=["current_stage", "progress", "metrics"])
        self.last_progress = self.run.progress

    @classmethod
    def _tracking_label(
        cls,
        *,
        backend: str,
        device: str,
        stage_progress: float,
        processed_ms: int,
        total_ms: int,
        frames_processed: int,
        frames_total: int,
        speed_x: float,
        eta_seconds: float | None,
        initializing: bool = False,
        operation: str = "Tracking",
    ) -> str:
        engine = backend.upper()
        if backend == "yolo":
            engine = f"YOLO {device.upper()}"
        if initializing:
            return (
                f"{operation} · initialisation {engine} · {frames_total:,} images prévues"
            ).replace(",", " ")
        eta = "ETA en calcul"
        if eta_seconds is not None:
            eta = f"reste {cls._duration_label(eta_seconds)}"
        frames = f"{frames_processed:,}/{frames_total:,}".replace(",", " ")
        return (
            f"{operation} {stage_progress:.1f}% · vidéo {cls._duration_label(processed_ms / 1000)}"
            f"/{cls._duration_label(total_ms / 1000)} · {frames} images · "
            f"{speed_x:.2f}× · {eta} · {engine}"
        )

    @staticmethod
    def _duration_label(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _normalize_objects(
        self,
        analysis: FrameAnalysis,
        camera: CameraStabilizer,
        projector: PitchProjector | None,
        period_prefix: str,
        summaries: dict[str, dict],
    ) -> None:
        for obj in analysis.objects:
            if obj.track_id != "ball" and not obj.track_id.startswith(period_prefix):
                obj.track_id = period_prefix + obj.track_id
                if obj.player_key:
                    obj.player_key = obj.track_id
            if obj.image_x is None or obj.image_y is None:
                continue
            raw_x = obj.image_x * analysis.width
            raw_y = obj.image_y * analysis.height
            stable_x, stable_y = camera.stabilize_point(raw_x, raw_y)
            obj.metadata["raw_image_normalized"] = [obj.image_x, obj.image_y]
            obj.image_x = stable_x / max(analysis.width, 1)
            obj.image_y = stable_y / max(analysis.height, 1)
            if projector is not None:
                projected = projector.project(stable_x, stable_y)
                if projected is not None:
                    obj.pitch_x, obj.pitch_y = projected
                    analysis.coordinate_space = "pitch_meters"
            if obj.role == ObjectRole.BALL:
                continue
            summary = summaries.setdefault(
                obj.track_id,
                {
                    "track_uid": obj.track_id,
                    "role": obj.role if obj.role in Track.Role.values else Track.Role.OTHER,
                    "start_ms": analysis.timestamp_ms,
                    "end_ms": analysis.timestamp_ms,
                    "confidence_sum": 0.0,
                    "samples": 0,
                    "team_votes": Counter(),
                    "shirt_votes": Counter(),
                    "points": [],
                },
            )
            summary["end_ms"] = analysis.timestamp_ms
            summary["confidence_sum"] += obj.confidence
            summary["samples"] += 1
            if obj.team_key:
                summary["team_votes"][obj.team_key] += 1
            if obj.shirt_number is not None:
                summary["shirt_votes"][obj.shirt_number] += 1
            if len(summary["points"]) < 2_000:
                summary["points"].append(
                    {
                        "t": analysis.timestamp_ms,
                        "x": obj.pitch_x if obj.pitch_x is not None else obj.image_x,
                        "y": obj.pitch_y if obj.pitch_y is not None else obj.image_y,
                        "space": "pitch_meters" if obj.pitch_x is not None else "image_normalized",
                    }
                )

    def _projector(self, period: MatchPeriod) -> PitchProjector | None:
        calibration = (self.config.get("pitch_calibration") or {}).get(str(period.number), {})
        image_points = calibration.get("image_points") or []
        pitch_points = calibration.get("pitch_points") or []
        projector = PitchProjector()
        return projector if projector.calibrate(image_points, pitch_points) else None

    @transaction.atomic
    def _persist(self, periods: list[MatchPeriod], result: dict) -> dict:
        self._stage(AnalysisRun.Stage.POSSESSION, 70)
        Event.objects.filter(analysis_run=self.run).delete()
        PossessionSegment.objects.filter(analysis_run=self.run).delete()
        Track.objects.filter(analysis_run=self.run).delete()

        tracks: list[Track] = []
        for uid, summary in result["tracks"].items():
            team_key = self._winner(summary["team_votes"])
            shirt_number = self._winner(summary["shirt_votes"])
            samples = max(int(summary["samples"]), 1)
            tracks.append(
                Track(
                    match=self.match,
                    analysis_run=self.run,
                    track_uid=uid,
                    role=summary["role"],
                    team=self.team_by_key.get(team_key),
                    predicted_shirt_number=shirt_number,
                    identity_confidence=round(summary["confidence_sum"] / samples, 4),
                    video_start_ms=summary["start_ms"],
                    video_end_ms=summary["end_ms"],
                    metadata={"samples": samples, "points": summary["points"]},
                )
            )
        Track.objects.bulk_create(tracks, batch_size=500)
        track_by_uid = {
            track.track_uid: track
            for track in Track.objects.filter(analysis_run=self.run).select_related("player")
        }

        possessions: list[PossessionSegment] = []
        for period, span in result["spans"]:
            track = track_by_uid.get(span.player_key or "")
            possessions.append(
                PossessionSegment(
                    match=self.match,
                    analysis_run=self.run,
                    period=period,
                    team=self.team_by_key.get(span.team_key),
                    player=track.player if track else None,
                    owner_track=track,
                    state=str(span.state),
                    video_start_ms=span.start_ms,
                    video_end_ms=min(period.video_end_ms, span.end_ms),
                    match_start_ms=self._match_time(period, span.start_ms),
                    match_end_ms=self._match_time(
                        period, min(period.video_end_ms, span.end_ms)
                    ),
                    confidence=span.confidence,
                )
            )
        PossessionSegment.objects.bulk_create(possessions, batch_size=1_000)

        self._stage(AnalysisRun.Stage.EVENTS, 76)
        events: list[Event] = []
        for period, candidate in result["events"]:
            actor_track = track_by_uid.get(candidate.player_key or "")
            recipient_track = track_by_uid.get(candidate.recipient_key or "")
            confidence = max(0.0, min(1.0, candidate.confidence))
            review_status = (
                Event.ReviewStatus.AUTO_ACCEPTED
                if confidence >= 0.92 and result["backend"] != "heuristic"
                else Event.ReviewStatus.PENDING
            )
            events.append(
                Event(
                    match=self.match,
                    analysis_run=self.run,
                    period=period,
                    event_type=candidate.event_type,
                    team=self.team_by_key.get(candidate.team_key),
                    player=actor_track.player if actor_track else None,
                    recipient=recipient_track.player if recipient_track else None,
                    actor_track=actor_track,
                    recipient_track=recipient_track,
                    video_time_ms=candidate.timestamp_ms,
                    match_time_ms=self._match_time(period, candidate.timestamp_ms),
                    start_x=candidate.start_x,
                    start_y=candidate.start_y,
                    end_x=candidate.end_x,
                    end_y=candidate.end_y,
                    outcome=candidate.outcome,
                    confidence=confidence,
                    visibility="full" if confidence >= 0.75 else "partial",
                    qualifiers=candidate.qualifiers,
                    review_status=review_status,
                    source="ai",
                    model_version=f"{result['backend']}-baseline-v1",
                )
            )
        Event.objects.bulk_create(events, batch_size=1_000)
        saved_events = list(Event.objects.filter(analysis_run=self.run).order_by("video_time_ms"))

        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.EVENTS,
            f"events-{self.run.pk}.json",
            [candidate.to_dict() | {"period": period.number} for period, candidate in result["events"]],
            metadata={"event_count": len(events)},
        )
        if self.analysis_mode == "full":
            self._stats(result)
        else:
            self._stage(AnalysisRun.Stage.STATS, 82)
        return {"tracks": track_by_uid, "events": saved_events}

    def _stats(self, result: dict) -> None:
        self._stage(AnalysisRun.Stage.STATS, 82)
        candidates = [candidate for _, candidate in result["events"]]
        spans = [span for _, span in result["spans"]]
        team_metrics, track_metrics = StatsAggregator().aggregate(candidates, spans)
        effective_minutes = sum(
            span.duration_ms
            for span in spans
            if span.state in {"controlled", "contested", "loose"}
        ) / 60_000
        for team_key, team in self.team_by_key.items():
            metrics = blank_metrics()
            metrics.update(team_metrics.get(team_key, {}))
            TeamMatchStat.objects.update_or_create(
                match=self.match,
                team=team,
                defaults={
                    "analysis_run": self.run,
                    "minutes_played": round(effective_minutes, 2),
                    "metrics": metrics,
                },
            )

        tracks = Track.objects.filter(analysis_run=self.run).select_related("player")
        metrics_by_player: dict[int, dict] = defaultdict(blank_metrics)
        heatmap_by_player: dict[int, list] = defaultdict(list)
        minutes_by_player: dict[int, float] = defaultdict(float)
        for track in tracks:
            metrics = track_metrics.get(track.track_uid, blank_metrics())
            track.metadata = {**track.metadata, "metrics": metrics}
            track.save(update_fields=["metadata"])
            if not track.player_id:
                continue
            for key, value in metrics.items():
                if (
                    isinstance(value, (int, float))
                    and key not in {"pass_accuracy_pct", "possession_pct"}
                ):
                    metrics_by_player[track.player_id][key] += value
            heatmap_by_player[track.player_id].extend(track.metadata.get("points", []))
            minutes_by_player[track.player_id] += max(
                0.0, (track.video_end_ms - track.video_start_ms) / 60_000
            )
        for player_id, metrics in metrics_by_player.items():
            passes = metrics.get("passes", 0)
            metrics["pass_accuracy_pct"] = round(
                100.0 * metrics.get("passes_completed", 0) / max(passes, 1), 2
            ) if passes else 0.0
            PlayerMatchStat.objects.update_or_create(
                match=self.match,
                player_id=player_id,
                defaults={
                    "analysis_run": self.run,
                    "minutes_played": round(minutes_by_player[player_id], 2),
                    "metrics": metrics,
                    "heatmap": heatmap_by_player[player_id][:5_000],
                    "touchmap": [],
                },
            )

    def _clips(self, metadata, saved_events: list[Event], candidates) -> dict:
        self._stage(AnalysisRun.Stage.CLIPS, 87)
        enabled = bool(self.config.get("render_clips", True))
        candidate_list = [candidate for _, candidate in candidates]
        clip_event_types = set(
            self.config.get(
                "clip_event_types",
                ["goal", "shot", "duel", "aerial_duel", "tackle", "dribble"],
            )
        )
        selected = [
            (index, candidate)
            for index, candidate in enumerate(candidate_list)
            if candidate.event_type in clip_event_types
        ]
        windows = ClipPlanner().plan(
            [candidate for _, candidate in selected],
            video_duration_ms=metadata.duration_ms,
        )
        result = {"enabled": enabled, "planned": len(windows), "rendered": 0, "errors": []}
        if not enabled or not windows:
            return result
        clip_root = Path(settings.MEDIA_ROOT) / "matches" / str(self.match.pk) / "clips"
        clip_root.mkdir(parents=True, exist_ok=True)
        for index, window in enumerate(windows):
            self._check_cancelled()
            filename = f"{str(self.run.pk)[:8]}-{index + 1:04d}-{window.label}.mp4"
            output_path = clip_root / filename
            try:
                render_clip(metadata.path, str(output_path), window)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                result["errors"].append(f"{filename}: {exc}")
                continue
            relative_name = str(output_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
            for selected_index in window.event_indexes:
                event_index = selected[selected_index][0]
                if event_index < len(saved_events):
                    saved_events[event_index].clip.name = relative_name
                    saved_events[event_index].save(update_fields=["clip"])
            result["rendered"] += 1
        return result

    def _report(self, metadata, quality, periods, result: dict, clips: dict) -> dict:
        self._stage(AnalysisRun.Stage.REPORT, 95)
        report = {
            "schema": "football-tracking/0.1",
            "run_id": str(self.run.pk),
            "match_id": str(self.match.pk),
            "analysis_mode": self.analysis_mode,
            "video": metadata.to_dict(),
            "quality": {"score": quality.score, "grade": quality.grade, **quality.metrics},
            "periods": [self._period_payload(period) for period in periods],
            "backend": result["backend"],
            "tracking_fps": result["tracking_fps"],
            "camera": result["camera"],
            "windows": result.get("windows", []),
            "diagnostics": result.get("diagnostics", {}),
            "previews": result.get("previews", []),
            "counts": {
                "tracks": len(result["tracks"]),
                "possession_spans": len(result["spans"]),
                "events": len(result["events"]),
            },
            "clips": clips,
            "limitations": self._limitations(result["backend"], periods, quality.grade),
        }
        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.REPORT,
            f"report-{self.run.pk}.json",
            report,
        )
        return report

    def _finish(self, periods, quality, result, report) -> None:
        requires_review = (
            self.analysis_mode != "full"
            or result["backend"] == "heuristic"
            or quality.grade in {"C", "reject"}
            or not all(period.confirmed for period in periods)
            or Event.objects.filter(
                analysis_run=self.run,
                review_status=Event.ReviewStatus.PENDING,
            ).exists()
        )
        self.run.status = AnalysisRun.Status.REVIEW if requires_review else AnalysisRun.Status.COMPLETED
        self.run.current_stage = AnalysisRun.Stage.DONE
        self.run.progress = 100
        self.run.metrics = report
        self.run.finished_at = timezone.now()
        self.run.save(
            update_fields=["status", "current_stage", "progress", "metrics", "finished_at"]
        )
        self.match.status = Match.Status.REVIEW if requires_review else Match.Status.COMPLETED
        self.match.save(update_fields=["status", "updated_at"])

    def _fail(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"[:4_000]
        self.run.status = AnalysisRun.Status.FAILED
        self.run.error_message = message
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "error_message", "finished_at"])
        self.match.status = Match.Status.FAILED
        self.match.save(update_fields=["status", "updated_at"])

    def _cancel(self) -> None:
        self.run.refresh_from_db(fields=["status"])
        self.run.status = AnalysisRun.Status.CANCELLED
        self.run.finished_at = timezone.now()
        self.run.save(update_fields=["status", "finished_at"])
        self.match.status = Match.Status.UPLOADED
        self.match.save(update_fields=["status", "updated_at"])

    def _stage(self, stage: str, progress: int) -> None:
        progress = max(0, min(100, progress))
        if stage == self.run.current_stage and progress == self.last_progress:
            return
        self.run.current_stage = stage
        self.run.progress = progress
        self.run.save(update_fields=["current_stage", "progress"])
        self.last_progress = progress

    def _check_cancelled(self) -> None:
        status = AnalysisRun.objects.values_list("status", flat=True).get(pk=self.run.pk)
        if status == AnalysisRun.Status.CANCELLED:
            raise AnalysisCancelled

    @staticmethod
    def _winner(counter: Counter):
        return counter.most_common(1)[0][0] if counter else None

    @staticmethod
    def _match_time(period: MatchPeriod, video_time_ms: int) -> int:
        return period.match_clock_start_ms + max(0, video_time_ms - period.video_start_ms)

    @staticmethod
    def _period_payload(period: MatchPeriod) -> dict:
        return {
            "number": period.number,
            "label": period.label,
            "video_start_ms": period.video_start_ms,
            "video_end_ms": period.video_end_ms,
            "match_clock_start_ms": period.match_clock_start_ms,
            "match_clock_end_ms": period.match_clock_end_ms,
            "confidence": period.confidence,
            "confirmed": period.confirmed,
            "source": period.source,
        }

    @staticmethod
    def _limitations(backend: str, periods: list[MatchPeriod], quality_grade: str) -> list[str]:
        limitations = []
        if backend == "heuristic":
            limitations.append(
                "Mode diagnostic : installe les dépendances ML et des poids football pour les joueurs et le ballon."
            )
        if not all(period.confirmed for period in periods):
            limitations.append("Les limites des mi-temps doivent être confirmées par un analyste.")
        if quality_grade in {"C", "reject"}:
            limitations.append("La qualité ou l’angle vidéo limite la fiabilité de certaines actions.")
        limitations.append(
            "Les joueurs hors champ ne peuvent pas être localisés avec une seule caméra de diffusion."
        )
        return limitations


def run_analysis(run_id) -> AnalysisRun:
    run = AnalysisRun.objects.get(pk=run_id)
    return MatchAnalysisRunner(run).execute()


def claim_next_analysis() -> AnalysisRun | None:
    """Atomically claim the oldest queued run (safe for one or more workers)."""
    with transaction.atomic():
        run = (
            AnalysisRun.objects.select_for_update()
            .filter(status=AnalysisRun.Status.QUEUED)
            .order_by("created_at")
            .first()
        )
        if run is None:
            return None
        run.status = AnalysisRun.Status.PROCESSING
        run.started_at = run.started_at or timezone.now()
        run.save(update_fields=["status", "started_at"])
        return run
