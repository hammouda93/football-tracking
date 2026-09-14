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

import numpy as np

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
    TrackingGroundTruth,
)

from .ball_in_play import BallInPlayEngine
from .camera import CameraStabilizer, PitchProjector
from .clips import ClipPlanner, render_clip
from .events import EventEngine
from .evaluation import evaluate_tracking
from .gsr import ExternalGSRExecutor, GSR_ENGINE_PROFILES
from .periods import PeriodDetector
from .providers.base import build_provider
from .providers.gsr import ExternalGSRVisionProvider
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
        return mode if mode in {"prepare", "reference", "sample", "full"} else "full"

    def _empty_result(self) -> dict:
        return {
            "analysis_mode": self.analysis_mode,
            "backend": str(self.config.get("backend", "heuristic")),
            "athlete_engine": str(self.config.get("athlete_engine", "legacy")),
            "gsr": {},
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
        if len(confirmed) == 2 and self.analysis_mode != "prepare":
            periods = confirmed
            payload = {
                "source": "confirmed",
                "requires_review": False,
                "periods": [self._period_payload(period) for period in periods],
            }
        else:
            detection = PeriodDetector().detect(signals, duration_ms)
            existing_bounds = {
                period.number: (period.video_start_ms, period.video_end_ms)
                for period in self.match.periods.filter(number__in=(1, 2))
            }
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
            new_bounds = {
                period.number: (period.video_start_ms, period.video_end_ms)
                for period in periods
            }
            if existing_bounds and existing_bounds != new_bounds:
                self._invalidate_previous_samples()
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

    def _invalidate_previous_samples(self) -> None:
        sample_runs = self.match.analysis_runs.exclude(pk=self.run.pk).order_by(
            "-created_at"
        )[:25]
        for sample_run in sample_runs:
            if str((sample_run.config or {}).get("analysis_mode")) != "sample":
                continue
            metrics = dict(sample_run.metrics or {})
            diagnostics = dict(metrics.get("diagnostics") or {})
            diagnostics["periods_changed_since_run"] = True
            diagnostics["manual_approved"] = False
            metrics["diagnostics"] = diagnostics
            sample_run.metrics = metrics
            sample_run.save(update_fields=["metrics"])

    def _track(self, periods: list[MatchPeriod], metadata) -> dict:
        self._stage(AnalysisRun.Stage.TRACKING, 22)
        backend = str(self.config.get("backend", "heuristic"))
        athlete_engine = str(self.config.get("athlete_engine", "legacy")).strip().lower()
        if athlete_engine not in {"legacy", *GSR_ENGINE_PROFILES}:
            raise ValueError(f"Moteur de suivi des athlètes inconnu : {athlete_engine}")
        device = str(self.config.get("device", "cpu"))
        requested_tracking_fps = max(
            0.5, float(self.config.get("tracking_fps", 10.0))
        )
        fps_backend = (
            "yolo"
            if backend == "yolo"
            or (
                athlete_engine != "legacy"
                and str(self.config.get("gsr_ball_backend", "yolo")).lower() == "yolo"
            )
            else backend
        )
        tracking_fps = self._effective_tracking_fps(
            backend=fps_backend,
            requested_fps=requested_tracking_fps,
            native_fps=float(metadata.fps or 0.0),
            minimum_yolo_fps=float(
                self.config.get("min_yolo_tracking_fps", 8.0)
            ),
        )
        athlete_tracking_fps = (
            tracking_fps
            if athlete_engine == "legacy"
            else max(1.0, float(self.config.get("gsr_tracking_fps", 5.0)))
        )
        windows = self._tracking_windows(periods)
        tracking_duration = sum(window["end_ms"] - window["start_ms"] for window in windows)
        estimated_frames = math.ceil(tracking_duration / 1000.0 * tracking_fps)
        operation = (
            "Test court"
            if self.analysis_mode == "reference"
            else ("Validation 2 min" if self.analysis_mode == "sample" else "Tracking")
        )
        progress_backend = backend if athlete_engine == "legacy" else athlete_engine
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
                "athlete_engine": athlete_engine,
                "device": device,
                "tracking_fps": tracking_fps,
                "requested_tracking_fps": requested_tracking_fps,
                "label": self._tracking_label(
                    backend=progress_backend,
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
        provider = (
            self._build_legacy_provider(backend, device, tracking_fps)
            if athlete_engine == "legacy"
            else None
        )
        gsr_audit: dict[str, Any] = {}
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
        diagnostic_frame_series: defaultdict[str, list[int]] = defaultdict(list)
        period_diagnostic_counts: defaultdict[int, Counter] = defaultdict(Counter)
        period_diagnostic_series: defaultdict[
            int, defaultdict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        team_observations: Counter = Counter()
        period_team_observations: defaultdict[int, Counter] = defaultdict(Counter)
        model_classes: dict[str, str] = {}
        tracker_name = str(self.config.get("yolo_tracker", "bytetrack"))
        preview_artifacts: list[dict] = []
        ground_truth_evaluation: dict[str, Any] = {"status": "missing"}
        live_preview_path = (
            Path(settings.MEDIA_ROOT)
            / "matches"
            / str(self.match.pk)
            / "live"
            / f"{self.run.pk}.jpg"
        )
        live_preview_path.parent.mkdir(parents=True, exist_ok=True)
        last_live_preview = 0.0
        native_live_enabled = (
            bool(self.config.get("live_window", False))
            and self.analysis_mode in {"reference", "sample"}
            and (backend == "yolo" or athlete_engine != "legacy")
        )
        native_live_requested = native_live_enabled
        self._live_track_history: dict[str, list[tuple[int, int]]] = {}
        self._live_debug_overlay = False

        with tempfile.TemporaryDirectory(prefix="football-tracking-") as temp_dir:
            temp_path = Path(temp_dir)
            if provider is None:
                provider, gsr_audit = self._build_external_gsr_provider(
                    athlete_engine=athlete_engine,
                    backend=backend,
                    device=device,
                    tracking_fps=tracking_fps,
                    athlete_tracking_fps=athlete_tracking_fps,
                    windows=windows,
                    work_dir=temp_path,
                    live_preview_path=live_preview_path,
                    tracking_duration_ms=tracking_duration,
                    estimated_frames=estimated_frames,
                    operation=operation,
                )
            tracking_started_at = time.monotonic()
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
                    self._live_track_history = {}
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
                        current_live_processed = processed_ms + timestamp_ms - window_start_ms
                        if native_live_enabled:
                            native_live_enabled = self._show_live_tracking(
                                frame,
                                analysis,
                                elapsed_ms=current_live_processed,
                                total_ms=tracking_duration,
                                period_number=period.number,
                                window_index=window["index"],
                                window_count=len(windows),
                                team_labels={
                                    "home": f"{self._team_code(self.match.home_team)} (T1)",
                                    "away": f"{self._team_code(self.match.away_team)} (T2)",
                                },
                            )
                        now = time.monotonic()
                        if now - last_live_preview >= 2.0:
                            self._write_live_preview(
                                frame,
                                analysis,
                                live_preview_path,
                                team_labels={
                                    "home": f"{self._team_code(self.match.home_team)} (T1)",
                                    "away": f"{self._team_code(self.match.away_team)} (T2)",
                                },
                            )
                            last_live_preview = now
                        sample = ball_engine.observe(analysis)
                        period_samples.append(sample)
                        all_samples.append((period, sample))
                        diagnostic_counts["frames"] += 1
                        diagnostic_counts["gsr_missing_frames"] += int(
                            bool(analysis.diagnostics.get("gsr_frame_missing", False))
                        )
                        diagnostic_counts["athlete_observations"] += len(analysis.athletes)
                        raw_athletes = int(
                            analysis.diagnostics.get(
                                "raw_athlete_detections", len(analysis.athletes)
                            )
                        )
                        diagnostic_counts["raw_athlete_detections"] += raw_athletes
                        diagnostic_counts["tracker_dropped_athletes"] += max(
                            0,
                            raw_athletes - len(analysis.athletes),
                        )
                        diagnostic_counts["raw_referee_detections"] += int(
                            analysis.diagnostics.get("raw_referee_detections", 0)
                        )
                        diagnostic_counts["raw_ball_detections"] += int(
                            analysis.diagnostics.get("raw_ball_detections", 0)
                        )
                        diagnostic_counts["raw_other_detections"] += int(
                            analysis.diagnostics.get("raw_other_detections", 0)
                        )
                        diagnostic_counts["rejected_person_detections"] += int(
                            analysis.diagnostics.get("rejected_person_detections", 0)
                        )
                        diagnostic_counts["duplicate_person_detections"] += int(
                            analysis.diagnostics.get("duplicate_person_detections", 0)
                        )
                        period_counts = period_diagnostic_counts[period.number]
                        period_counts["frames"] += 1
                        period_counts["raw_athlete_detections"] += raw_athletes
                        period_counts["athlete_observations"] += len(analysis.athletes)
                        stage_values = {
                            "detector_candidate_athletes": int(
                                analysis.diagnostics.get(
                                    "detector_candidate_athletes", raw_athletes
                                )
                            ),
                            "tracker_input_athletes": int(
                                analysis.diagnostics.get(
                                    "tracker_input_athletes", raw_athletes
                                )
                            ),
                            "tracker_output_athletes": int(
                                analysis.diagnostics.get(
                                    "tracker_output_athletes", len(analysis.athletes)
                                )
                            ),
                            "detector_rescue_athletes": int(
                                analysis.diagnostics.get("detector_rescue_athletes", 0)
                            ),
                            "native_identity_input_athletes": int(
                                analysis.diagnostics.get(
                                    "native_identity_input_athletes",
                                    analysis.diagnostics.get(
                                        "tracker_output_athletes",
                                        len(analysis.athletes),
                                    ),
                                )
                            ),
                            "yolo_deduplicated_athletes": int(
                                analysis.diagnostics.get(
                                    "yolo_deduplicated_athletes", 0
                                )
                            ),
                            "native_duplicates_removed": int(
                                analysis.diagnostics.get(
                                    "native_duplicates_removed", 0
                                )
                            ),
                            "native_identity_births": int(
                                analysis.diagnostics.get(
                                    "native_identity_births", 0
                                )
                            ),
                            "native_direct_assignments": int(
                                analysis.diagnostics.get(
                                    "native_direct_assignments", 0
                                )
                            ),
                            "native_short_stitches": int(
                                analysis.diagnostics.get(
                                    "native_short_stitches", 0
                                )
                            ),
                            "native_long_stitches": int(
                                analysis.diagnostics.get(
                                    "native_long_stitches", 0
                                )
                            ),
                            "native_direct_rejections": int(
                                analysis.diagnostics.get(
                                    "native_direct_rejections", 0
                                )
                            ),
                            "scene_cut_tracker_resets": int(
                                bool(
                                    analysis.diagnostics.get(
                                        "scene_cut_tracker_reset", False
                                    )
                                )
                            ),
                        }
                        for key, value in stage_values.items():
                            diagnostic_counts[key] += value
                            period_counts[key] += value
                        frame_values = {
                            "raw_athlete_detections": raw_athletes,
                            "detector_candidate_athletes": stage_values[
                                "detector_candidate_athletes"
                            ],
                            "tracker_input_athletes": stage_values[
                                "tracker_input_athletes"
                            ],
                            "tracker_output_athletes": stage_values[
                                "tracker_output_athletes"
                            ],
                            "detector_rescue_athletes": stage_values[
                                "detector_rescue_athletes"
                            ],
                            "native_identity_input_athletes": stage_values[
                                "native_identity_input_athletes"
                            ],
                            "athlete_observations": len(analysis.athletes),
                        }
                        for key, value in frame_values.items():
                            diagnostic_frame_series[key].append(value)
                            period_diagnostic_series[period.number][key].append(value)
                        for reason, value in (
                            analysis.diagnostics.get(
                                "native_direct_rejection_reasons"
                            )
                            or {}
                        ).items():
                            reason_key = f"direct_rejection_{reason}"
                            diagnostic_counts[reason_key] += int(value)
                            period_counts[reason_key] += int(value)
                        for confidence_band, value in (
                            analysis.diagnostics.get("athlete_confidence_bins") or {}
                        ).items():
                            band_key = f"athlete_confidence_{confidence_band}"
                            diagnostic_counts[band_key] += int(value)
                            period_counts[band_key] += int(value)
                        ball_reason = str(
                            analysis.diagnostics.get("ball_selection_reason") or ""
                        )
                        if ball_reason:
                            diagnostic_counts[f"ball_reason_{ball_reason}"] += 1
                            period_counts[f"ball_reason_{ball_reason}"] += 1
                        model_classes.update(analysis.diagnostics.get("model_classes") or {})
                        tracker_name = str(
                            analysis.diagnostics.get("tracker") or tracker_name
                        )
                        ball_visible = int(analysis.ball is not None)
                        diagnostic_counts["ball_visible_frames"] += ball_visible
                        period_counts["ball_visible_frames"] += ball_visible
                        period_counts["raw_ball_detections"] += int(
                            analysis.diagnostics.get("raw_ball_detections", 0)
                        )
                        diagnostic_counts["pitch_metric_frames"] += int(
                            any(
                                athlete.pitch_x is not None and athlete.pitch_y is not None
                                for athlete in analysis.athletes
                            )
                        )
                        field_live = int(
                            analysis.field_score >= 0.14
                            and not analysis.scene_cut
                            and analysis.replay_probability < 0.65
                        )
                        diagnostic_counts["field_frames"] += field_live
                        period_counts["field_frames"] += field_live
                        state_key = f"state_{str(sample.state)}"
                        diagnostic_counts[state_key] += 1
                        period_counts[state_key] += 1
                        for athlete in analysis.athletes:
                            team_key = athlete.team_key or "unknown"
                            team_observations[team_key] += 1
                            period_team_observations[period.number][team_key] += 1
                        if (
                            self.analysis_mode in {"reference", "sample"}
                            and not preview_saved
                            and timestamp_ms >= preview_target_ms
                        ):
                            preview_path = temp_path / (
                                f"sample-p{period.number}-w{window['index']}.jpg"
                            )
                            self._write_sample_preview(
                                frame,
                                analysis,
                                preview_path,
                                team_labels={
                                    "home": f"{self._team_code(self.match.home_team)} (T1)",
                                    "away": f"{self._team_code(self.match.away_team)} (T2)",
                                },
                            )
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
                        # Keep the detailed detector rows only long enough to
                        # render the side-by-side diagnostic. They would make
                        # the full NDJSON artifact unnecessarily large.
                        analysis.diagnostics.pop("raw_detections", None)
                        analysis.diagnostics.pop("raw_athlete_boxes", None)
                        analysis.diagnostics.pop("rejected_person_boxes", None)
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
                        progress_floor = 60 if athlete_engine != "legacy" else 22
                        progress_span = 8 if athlete_engine != "legacy" else 46
                        percent = min(68, progress_floor + int(progress_span * progress_ratio))
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
                                "athlete_engine": athlete_engine,
                                "device": device,
                                "tracking_fps": tracking_fps,
                                "requested_tracking_fps": requested_tracking_fps,
                            }
                            detail["label"] = self._tracking_label(
                                backend=progress_backend,
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

            identity_aliases = self._consolidate_roster_tracks(track_summaries)
            if identity_aliases:
                self._rewrite_tracking_identities(tracking_path, identity_aliases)
                for _period, sample in all_samples:
                    sample.player_key = self._identity_alias(
                        sample.player_key,
                        identity_aliases,
                    )
                for _period, span in all_spans:
                    span.player_key = self._identity_alias(
                        span.player_key,
                        identity_aliases,
                    )
                for _period, candidate in all_events:
                    candidate.player_key = self._identity_alias(
                        candidate.player_key,
                        identity_aliases,
                    )
                    candidate.recipient_key = self._identity_alias(
                        candidate.recipient_key,
                        identity_aliases,
                    )

            if self.analysis_mode == "sample":
                truth = TrackingGroundTruth.objects.filter(match=self.match).first()
                if truth is not None and truth.file:
                    try:
                        ground_truth_evaluation = evaluate_tracking(
                            tracking_path,
                            truth.file.path,
                            timestamp_tolerance_ms=max(
                                45,
                                int(round(600.0 / max(tracking_fps, 1.0))),
                            ),
                        )
                    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                        ground_truth_evaluation = {
                            "status": "error",
                            "verdict": "fail",
                            "issues": [f"Évaluation vérité terrain impossible: {exc}"],
                        }

            _save_local_artifact(
                self.run,
                AnalysisArtifact.Kind.TRACKING,
                f"tracking-{self.run.pk}.ndjson",
                tracking_path,
                metadata={
                    "backend": backend,
                    "athlete_engine": athlete_engine,
                    "gsr": gsr_audit,
                    "tracking_fps": tracking_fps,
                    "requested_tracking_fps": requested_tracking_fps,
                    "coordinate_systems": ["image_normalized", "pitch_meters"],
                    "analysis_mode": self.analysis_mode,
                    "windows": [self._window_payload(window) for window in windows],
                },
            )

        if native_live_requested:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

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
            tracking_fps=tracking_fps,
            requested_tracking_fps=requested_tracking_fps,
            model_classes=model_classes,
            tracker_name=tracker_name,
            profile_name=str(getattr(provider, "profile", backend)),
            image_size=int(getattr(provider, "image_size", 0)),
            detector_confidence=float(getattr(provider, "confidence", 0.0)),
            ball_confidence=float(getattr(provider, "ball_confidence", 0.0)),
            tracker_frame_rate=int(getattr(provider, "tracker_frame_rate", 0)),
            team_calibration=(
                provider.team_calibration_diagnostics()
                if hasattr(provider, "team_calibration_diagnostics")
                else {}
            ),
            strict_validation=bool(
                self.config.get("native_gsr_strict_validation", False)
            ),
            frame_series=diagnostic_frame_series,
            period_counts=period_diagnostic_counts,
            period_frame_series=period_diagnostic_series,
            period_team_observations=period_team_observations,
        )
        diagnostics["athlete_engine"] = athlete_engine
        diagnostics["gsr"] = {
            key: value
            for key, value in gsr_audit.items()
            if key != "manifest"
        }
        diagnostics["gsr_missing_frames"] = int(
            diagnostic_counts["gsr_missing_frames"]
        )
        diagnostics["ground_truth"] = ground_truth_evaluation
        if ground_truth_evaluation.get("verdict") == "fail":
            diagnostics["issues"] = [
                *ground_truth_evaluation.get("issues", []),
                *diagnostics["issues"],
            ]
            diagnostics["verdict"] = "fail"
            diagnostics.setdefault("hard_blockers", []).append(
                "Le test annoté de 2 minutes n’atteint pas les seuils de qualité."
            )
        elif (
            self.analysis_mode == "sample"
            and bool(self.config.get("native_gsr_require_ground_truth", False))
            and ground_truth_evaluation.get("status") != "evaluated"
        ):
            issue = "Une vérité terrain CSV est requise pour valider ce profil."
            diagnostics["issues"].insert(0, issue)
            diagnostics.setdefault("hard_blockers", []).append(issue)
            diagnostics["verdict"] = "fail"
        if athlete_engine != "legacy":
            missing_ratio = diagnostic_counts["gsr_missing_frames"] / max(
                diagnostic_counts["frames"], 1
            )
            if missing_ratio > 0.05:
                diagnostics["issues"].insert(
                    0,
                    "Le résultat GSR ne couvre pas au moins 95 % des images demandées.",
                )
                diagnostics["verdict"] = "fail"
        return {
            "analysis_mode": self.analysis_mode,
            "backend": backend,
            "athlete_engine": athlete_engine,
            "gsr": gsr_audit,
            "tracking_fps": tracking_fps,
            "requested_tracking_fps": requested_tracking_fps,
            "samples": all_samples,
            "spans": all_spans,
            "events": all_events,
            "tracks": track_summaries,
            "camera": camera_summary,
            "windows": [self._window_payload(window) for window in windows],
            "diagnostics": diagnostics,
            "previews": preview_artifacts,
        }

    def _build_legacy_provider(
        self,
        backend: str,
        device: str,
        tracking_fps: float,
        *,
        ball_only: bool = False,
    ):
        ball_class_ids = self.config.get("yolo_ball_class_ids", [])
        profile = str(self.config.get("yolo_profile", "main_py"))
        provider_name = (
            "native_gsr"
            if backend == "yolo" and profile == "native_gsr" and not ball_only
            else backend
        )
        return build_provider(
            provider_name,
            model_path=self.config.get("yolo_model_path", ""),
            device=device,
            profile=profile,
            confidence=float(self.config.get("yolo_confidence", 0.30)),
            ball_confidence=float(self.config.get("yolo_ball_confidence", 0.12)),
            ball_tiled_recovery=bool(
                self.config.get("yolo_ball_tiled_recovery", False)
            ),
            ball_recovery_interval_frames=int(
                self.config.get("yolo_ball_recovery_interval_frames", 12)
            ),
            ball_recovery_image_size=int(
                self.config.get("yolo_ball_recovery_image_size", 960)
            ),
            ball_recovery_overlap=float(
                self.config.get("yolo_ball_recovery_overlap", 0.15)
            ),
            image_size=int(self.config.get("yolo_image_size", 1280)),
            tracking_fps=tracking_fps,
            tracker_name=str(self.config.get("yolo_tracker", "bytetrack")),
            tracker_low_confidence=float(
                self.config.get("yolo_track_low_confidence", 0.10)
            ),
            tracker_new_confidence=float(
                self.config.get("yolo_new_track_confidence", 0.25)
            ),
            tracker_match_threshold=float(
                self.config.get("yolo_track_match_threshold", 0.80)
            ),
            tracker_buffer_seconds=float(
                self.config.get("yolo_track_buffer_seconds", 5.0)
            ),
            player_class_ids=(
                [] if ball_only else self.config.get("yolo_player_class_ids", [])
            ),
            goalkeeper_class_ids=(
                [] if ball_only else self.config.get("yolo_goalkeeper_class_ids", [])
            ),
            referee_class_ids=(
                [] if ball_only else self.config.get("yolo_referee_class_ids", [])
            ),
            ball_class_ids=ball_class_ids,
            inference_class_ids=ball_class_ids if ball_only and backend == "yolo" else None,
            home_team_cluster=str(self.config.get("home_team_cluster", "B")),
            native_gsr_reid_model_path=str(
                self.config.get("native_gsr_reid_model_path", "")
            ),
            native_gsr_max_gap_seconds=float(
                self.config.get("native_gsr_max_gap_seconds", 3.0)
            ),
            native_gsr_global_max_gap_seconds=float(
                self.config.get("native_gsr_global_max_gap_seconds", 7_200.0)
            ),
            native_gsr_reid_backend=str(
                self.config.get("native_gsr_reid_backend", "auto")
            ),
            native_gsr_reid_model_name=str(
                self.config.get("native_gsr_reid_model_name", "osnet_x0_25")
            ),
            native_gsr_jersey_engine=str(
                self.config.get("native_gsr_jersey_engine", "auto")
            ),
            native_gsr_jersey_model_path=str(
                self.config.get("native_gsr_jersey_model_path", "")
            ),
            native_gsr_jersey_device=str(
                self.config.get("native_gsr_jersey_device", "cpu")
            ),
            native_gsr_jersey_interval_frames=int(
                self.config.get("native_gsr_jersey_interval_frames", 12)
            ),
            native_gsr_jersey_minimum_box_height=int(
                self.config.get("native_gsr_jersey_minimum_box_height", 72)
            ),
            native_gsr_jersey_max_crops_per_frame=int(
                self.config.get("native_gsr_jersey_max_crops_per_frame", 4)
            ),
            native_gsr_pitch_model_path=str(
                self.config.get("native_gsr_pitch_model_path", "")
            ),
            native_gsr_pitch_schema_path=str(
                self.config.get("native_gsr_pitch_schema_path", "")
            ),
            native_gsr_pitch_confidence=float(
                self.config.get("native_gsr_pitch_confidence", 0.20)
            ),
            native_gsr_pitch_interval_frames=int(
                self.config.get("native_gsr_pitch_interval_frames", 10)
            ),
            native_gsr_pitch_hold_frames=int(
                self.config.get("native_gsr_pitch_hold_frames", 20)
            ),
            native_gsr_pitch_expected_landmarks=int(
                self.config.get("native_gsr_pitch_expected_landmarks", 97)
            ),
            native_gsr_roster=list(self.config.get("native_gsr_roster") or []),
        )

    def _build_external_gsr_provider(
        self,
        *,
        athlete_engine: str,
        backend: str,
        device: str,
        tracking_fps: float,
        athlete_tracking_fps: float,
        windows: list[dict],
        work_dir: Path,
        live_preview_path: Path,
        tracking_duration_ms: int,
        estimated_frames: int,
        operation: str,
    ) -> tuple[ExternalGSRVisionProvider, dict]:
        executor = ExternalGSRExecutor(
            engine=athlete_engine,
            command=list(self.config.get("gsr_runner_command") or []),
            timeout_seconds=int(self.config.get("gsr_timeout_seconds", 43_200)),
            frame_tolerance_ms=int(self.config.get("gsr_frame_tolerance_ms", 120)),
            home_team_cluster=str(self.config.get("home_team_cluster", "B")),
        )

        def external_progress(payload: dict[str, Any]) -> None:
            try:
                stage_progress = max(0.0, min(100.0, float(payload.get("progress", 0.0))))
            except (TypeError, ValueError):
                stage_progress = 0.0
            progress = min(59, 22 + int(37 * stage_progress / 100.0))
            elapsed = max(0.0, float(payload.get("elapsed_seconds", 0.0) or 0.0))
            eta = payload.get("eta_seconds")
            try:
                eta = float(eta) if eta is not None else None
            except (TypeError, ValueError):
                eta = None
            processed_video_ms = max(
                0,
                min(
                    tracking_duration_ms,
                    int(payload.get("processed_video_ms", 0) or 0),
                ),
            )
            detail = {
                "stage": "external_gsr",
                "stage_progress": round(stage_progress, 2),
                "processed_video_ms": processed_video_ms,
                "total_video_ms": tracking_duration_ms,
                "frames_processed": int(payload.get("frames_processed", 0) or 0),
                "frames_total_estimate": estimated_frames,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta) if eta is not None else None,
                "speed_x": float(payload.get("speed_x", 0.0) or 0.0),
                "backend": backend,
                "athlete_engine": athlete_engine,
                "device": device,
                "tracking_fps": tracking_fps,
                "label": str(payload.get("label") or "").strip()
                or self._tracking_label(
                    backend=athlete_engine,
                    device=device,
                    stage_progress=stage_progress,
                    processed_ms=processed_video_ms,
                    total_ms=tracking_duration_ms,
                    frames_processed=int(payload.get("frames_processed", 0) or 0),
                    frames_total=estimated_frames,
                    speed_x=float(payload.get("speed_x", 0.0) or 0.0),
                    eta_seconds=eta,
                    operation=operation,
                ),
            }
            self._save_live_progress(progress, detail)

        store, audit = executor.prepare(
            work_dir=work_dir,
            video_path=self.video.file.path,
            run_id=str(self.run.pk),
            match_id=str(self.match.pk),
            windows=windows,
            tracking_fps=athlete_tracking_fps,
            precomputed_result=str(self.config.get("gsr_precomputed_result", "")),
            progress_callback=external_progress,
            cancel_callback=self._check_cancelled,
            live_preview_path=live_preview_path,
        )
        ball_backend = str(self.config.get("gsr_ball_backend", "yolo")).strip().lower()
        ball_provider = (
            None
            if ball_backend == "none"
            else self._build_legacy_provider(
                ball_backend,
                device,
                tracking_fps,
                ball_only=True,
            )
        )
        result_fps = max(0.1, float(store.fps or athlete_tracking_fps))
        effective_tolerance_ms = max(
            int(self.config.get("gsr_frame_tolerance_ms", 120)),
            math.ceil(500.0 / result_fps) + 5,
        )
        provider = ExternalGSRVisionProvider(
            store,
            tolerance_ms=effective_tolerance_ms,
            ball_provider=ball_provider,
        )
        audit["ball_backend"] = ball_backend
        audit["requested_athlete_tracking_fps"] = athlete_tracking_fps
        audit["athlete_tracking_fps"] = result_fps
        audit["frame_tolerance_ms"] = effective_tolerance_ms
        _save_json_artifact(
            self.run,
            AnalysisArtifact.Kind.TRACKING,
            f"gsr-audit-{self.run.pk}.json",
            audit,
            metadata={
                "artifact_type": "gsr_audit",
                "schema": audit["schema"],
                "engine": athlete_engine,
            },
        )
        return provider, audit

    def _show_live_tracking(
        self,
        frame,
        analysis: FrameAnalysis,
        *,
        elapsed_ms: int,
        total_ms: int,
        period_number: int,
        window_index: int,
        window_count: int,
        team_labels: dict[str, str],
    ) -> bool:
        """Show the same fluid native diagnostic style as the standalone main.py."""

        import cv2

        preview = frame.copy()
        _frame_height, frame_width = preview.shape[:2]
        debug_overlay = bool(getattr(self, "_live_debug_overlay", False))
        if debug_overlay:
            raw_colors = {
                "player": (255, 0, 255),
                "goalkeeper": (255, 120, 0),
                "referee": (255, 255, 0),
                "ball": (0, 255, 255),
            }
            for detection in analysis.diagnostics.get("raw_detections", []):
                box = detection.get("bbox") or []
                if len(box) < 4:
                    continue
                role = str(detection.get("role", "player"))
                x1, y1, x2, y2 = (int(value) for value in box[:4])
                color = raw_colors.get(role, (255, 0, 255))
                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2 if role == "ball" else 1,
                )

        final_colors = {
            "home": (70, 220, 120),
            "away": (70, 130, 255),
            "unknown": (200, 200, 200),
            "ball": (255, 255, 255),
            "goalkeeper": (255, 175, 50),
            "referee": (255, 220, 70),
        }
        active_history: set[str] = set()
        for obj in analysis.objects:
            x1, y1, x2, y2 = (int(value) for value in obj.bbox_xyxy)
            track_number = obj.track_id.rsplit("-", 1)[-1]
            identity = (
                f"N{obj.shirt_number} ID {track_number}"
                if obj.shirt_number is not None
                else f"ID {track_number}"
            )
            roster_name = str(obj.metadata.get("roster_player_name") or "").strip()
            if roster_name:
                identity = f"{identity} {roster_name}"
            if obj.role == ObjectRole.BALL:
                color = final_colors["ball"]
                label = "BALLON"
                thickness = 3
            elif obj.role == ObjectRole.GOALKEEPER:
                color = final_colors["goalkeeper"]
                label = f"GB {identity}"
                thickness = 2
            elif obj.role == ObjectRole.REFEREE:
                color = final_colors["referee"]
                label = f"ARBITRE/JUGE {identity}"
                thickness = 2
            else:
                color = final_colors.get(obj.team_key or "unknown", final_colors["unknown"])
                team_label = team_labels.get(obj.team_key or "", "EQUIPE ?")
                label = f"J {team_label} {identity}"
                thickness = 2

            cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                preview,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

            if obj.role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}:
                active_history.add(obj.track_id)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                history = self._live_track_history.setdefault(obj.track_id, [])
                history.append(center)
                if len(history) > 5:
                    del history[:-5]
                if len(history) >= 2:
                    deltas = [
                        (
                            history[index][0] - history[index - 1][0],
                            history[index][1] - history[index - 1][1],
                        )
                        for index in range(1, len(history))
                    ]
                    velocity_x = sum(item[0] for item in deltas) / len(deltas)
                    velocity_y = sum(item[1] for item in deltas) / len(deltas)
                    speed = math.hypot(velocity_x, velocity_y)
                    if speed >= 1.0:
                        scale = min(6.0, 80.0 / max(speed, 1.0))
                        arrow_tip = (
                            int(center[0] + velocity_x * scale),
                            int(center[1] + velocity_y * scale),
                        )
                        cv2.arrowedLine(
                            preview,
                            center,
                            arrow_tip,
                            (0, 0, 255),
                            2,
                            tipLength=0.30,
                        )

        self._live_track_history = {
            key: value
            for key, value in self._live_track_history.items()
            if key in active_history
        }
        elapsed_s = max(0.0, min(float(total_ms), float(elapsed_ms))) / 1000.0
        total_s = max(0.0, float(total_ms)) / 1000.0
        raw_players = int(analysis.diagnostics.get("raw_athlete_detections", 0))
        tracked_players = len(analysis.athletes)
        raw_balls = int(analysis.diagnostics.get("raw_ball_detections", 0))
        identity_diagnostics = (
            analysis.diagnostics.get("native_identity")
            or analysis.diagnostics.get("team_calibration")
            or {}
        )
        team_status = str(
            identity_diagnostics.get("status", "collecting")
        ).upper()
        reid_backend = str(
            identity_diagnostics.get("appearance_backend", "none")
        ).upper()
        reid_status = (
            "ACTIF" if identity_diagnostics.get("deep_reid_ready") else "FALLBACK"
        )
        jersey_tracklets = int(identity_diagnostics.get("jersey_tracklets", 0))
        canonical_tracks = int(identity_diagnostics.get("canonical_tracks", 0))
        stitched = int(identity_diagnostics.get("fragments_stitched", 0))
        duplicates = int(identity_diagnostics.get("duplicates_removed", 0))
        ball_source = (
            str(analysis.ball.metadata.get("ball_engine", "inconnue"))
            .replace("yolo_", "")
            .upper()
            if analysis.ball
            else "-"
        )
        operation = "REFERENCE 40 S" if self.analysis_mode == "reference" else "TEST 2 MIN"
        header_1 = (
            f"{operation} {elapsed_s:05.1f}/{total_s:.0f}s"
            f" | MT{period_number} SEQ={window_index}/{window_count}"
            f" | VIDEO={self._duration_label(analysis.timestamp_ms / 1000)}"
        )
        header_2 = (
            f"MODE={'BRUT+FINAL' if debug_overlay else 'FINAL'}"
            f" | RAW JOUEURS={raw_players} | FINAUX={tracked_players}"
            f" | ECART={max(0, raw_players - tracked_players)}"
        )
        header_3 = (
            f"RE-ID={reid_backend} {reid_status} | EQUIPES={team_status}"
            f" | OCR NUMEROS={jersey_tracklets}/{canonical_tracks}"
        )
        header_4 = (
            f"BALL RAW={raw_balls} FINAL={'OUI' if analysis.ball else 'NON'}"
            f" SOURCE={ball_source} | FUSIONS={stitched} | DOUBLONS={duplicates}"
            " | D=BRUT | ESC=ARRETER"
        )
        cv2.rectangle(preview, (0, 0), (frame_width, 120), (15, 15, 15), -1)
        for line, y, size in (
            (header_1, 25, 0.56),
            (header_2, 53, 0.52),
            (header_3, 80, 0.49),
            (header_4, 107, 0.46),
        ):
            cv2.putText(
                preview,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        max_width = 1500
        if preview.shape[1] > max_width:
            scale = max_width / preview.shape[1]
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        try:
            cv2.imshow("Football Tracking - LIVE", preview)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            logger.warning(
                "Fenetre OpenCV indisponible; le live navigateur reste actif: %s",
                exc,
            )
            return False
        if key in {ord("d"), ord("D")}:
            self._live_debug_overlay = not debug_overlay
        if key == 27:
            cv2.destroyAllWindows()
            raise AnalysisCancelled()
        return True

    @staticmethod
    def _write_sample_preview(
        frame,
        analysis: FrameAnalysis,
        output_path: Path,
        *,
        team_labels: dict[str, str] | None = None,
    ) -> None:
        import cv2

        raw_preview = frame.copy()
        preview = frame.copy()
        raw_colors = {
            "player": (0, 215, 255),
            "goalkeeper": (255, 175, 50),
            "referee": (255, 220, 70),
            "ball": (255, 0, 255),
        }
        raw_labels = {
            "player": "SOURCE J",
            "goalkeeper": "SOURCE GB",
            "referee": "SOURCE ARB",
            "ball": "SOURCE BALLON",
        }
        for detection in analysis.diagnostics.get("raw_detections", []):
            x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
            role = str(detection.get("role", "player"))
            confidence = float(detection.get("confidence", 0.0))
            color = raw_colors.get(role, (0, 215, 255))
            cv2.rectangle(raw_preview, (x1, y1), (x2, y2), color, 3 if role == "ball" else 2)
            cv2.putText(
                raw_preview,
                f"{raw_labels.get(role, 'YOLO')} {confidence:.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        colors = {
            "home": (70, 220, 120),
            "away": (70, 130, 255),
            "unknown": (190, 190, 190),
            "ball": (255, 255, 255),
            "goalkeeper": (255, 175, 50),
            "referee": (255, 220, 70),
        }
        for obj in analysis.objects:
            x1, y1, x2, y2 = (int(value) for value in obj.bbox_xyxy)
            is_ball = obj.role == ObjectRole.BALL
            if is_ball:
                color = colors["ball"]
            elif obj.role == ObjectRole.GOALKEEPER:
                color = colors["goalkeeper"]
            elif obj.role == ObjectRole.REFEREE:
                color = colors["referee"]
            else:
                color = colors.get(obj.team_key or "unknown", colors["unknown"])
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3 if is_ball else 2)
            track_number = obj.track_id.rsplit("-", 1)[-1]
            identity = (
                f"N{obj.shirt_number} ID {track_number}"
                if obj.shirt_number is not None
                else f"ID {track_number}"
            )
            roster_name = str(obj.metadata.get("roster_player_name") or "").strip()
            if roster_name:
                identity = f"{identity} {roster_name}"
            if is_ball:
                label = "BALLON"
            elif obj.role == ObjectRole.GOALKEEPER:
                label = f"GB {identity}"
            elif obj.role == ObjectRole.REFEREE:
                label = f"ARBITRE/JUGE {identity}"
            else:
                team_label = (team_labels or {}).get(obj.team_key or "", "EQUIPE ?")
                label = f"J {team_label} {identity}"
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
        cv2.rectangle(raw_preview, (8, 8), (760, 58), (20, 20, 20), -1)
        cv2.putText(
            raw_preview,
            "A. SORTIE SOURCE - AVANT NORMALISATION DJANGO",
            (18, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            raw_preview,
            "JAUNE=JOUEUR | BLEU=GB | CYAN=ARBITRE | MAGENTA=BALLON LOCAL",
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(preview, (8, 8), (760, 58), (20, 20, 20), -1)
        cv2.putText(
            preview,
            "B. TRACKING FINAL - DONNEES UTILISEES PAR L'ANALYSE",
            (18, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        teams = team_labels or {"home": "T1", "away": "T2"}
        role_legend = (
            f"VERT=J {teams['home']} | ORANGE=J {teams['away']} | "
            "BLEU=GB | CYAN=ARBITRE/JUGE | BLANC=BALLON"
        )
        cv2.putText(
            preview,
            role_legend,
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        separator = np.full((preview.shape[0], 8, 3), 245, dtype=np.uint8)
        comparison = np.hstack((raw_preview, separator, preview))
        max_width = 2560
        if comparison.shape[1] > max_width:
            scale = max_width / comparison.shape[1]
            comparison = cv2.resize(comparison, None, fx=scale, fy=scale)
        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError("Impossible d’écrire l’aperçu annoté du test rapide.")

    @staticmethod
    def _write_live_preview(
        frame,
        analysis: FrameAnalysis,
        output_path: Path,
        *,
        team_labels: dict[str, str] | None = None,
    ) -> None:
        """Atomically publish a light tracking frame for the browser poller."""

        import cv2

        preview = frame.copy()
        colors = {
            "home": (70, 220, 120),
            "away": (70, 130, 255),
            "unknown": (190, 190, 190),
            "ball": (255, 255, 255),
            "goalkeeper": (255, 175, 50),
            "referee": (255, 220, 70),
        }
        for obj in analysis.objects:
            x1, y1, x2, y2 = (int(value) for value in obj.bbox_xyxy)
            track_number = obj.track_id.rsplit("-", 1)[-1]
            identity = (
                f"N{obj.shirt_number} ID {track_number}"
                if obj.shirt_number is not None
                else f"ID {track_number}"
            )
            roster_name = str(obj.metadata.get("roster_player_name") or "").strip()
            if roster_name:
                identity = f"{identity} {roster_name}"
            if obj.role == ObjectRole.BALL:
                color = colors["ball"]
                label = "BALLON"
            elif obj.role == ObjectRole.GOALKEEPER:
                color = colors["goalkeeper"]
                label = f"GB {identity}"
            elif obj.role == ObjectRole.REFEREE:
                color = colors["referee"]
                label = f"ARBITRE/JUGE {identity}"
            else:
                color = colors.get(obj.team_key or "unknown", colors["unknown"])
                team_label = (team_labels or {}).get(obj.team_key or "", "EQUIPE ?")
                label = f"J {team_label} {identity}"
            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                color,
                3 if obj.role == ObjectRole.BALL else 2,
            )
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
        identity_diagnostics = (
            analysis.diagnostics.get("native_identity")
            or analysis.diagnostics.get("team_calibration")
            or {}
        )
        reid_backend = str(
            identity_diagnostics.get("appearance_backend", "none")
        ).upper()
        team_status = str(
            identity_diagnostics.get("status", "collecting")
        ).upper()
        jersey_tracklets = int(identity_diagnostics.get("jersey_tracklets", 0))
        ball_source = (
            str(analysis.ball.metadata.get("ball_engine", "inconnue"))
            .replace("yolo_", "")
            .upper()
            if analysis.ball
            else "-"
        )
        cv2.rectangle(preview, (8, 8), (1120, 66), (20, 20, 20), -1)
        cv2.putText(
            preview,
            f"TRACKING FINAL - VIDEO {MatchAnalysisRunner._duration_label(analysis.timestamp_ms / 1000)}",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"RE-ID={reid_backend} | EQUIPES={team_status} | OCR={jersey_tracklets} | BALL={ball_source}",
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        max_width = 1280
        if preview.shape[1] > max_width:
            scale = max_width / preview.shape[1]
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        temporary_path = output_path.with_name(f".{output_path.stem}.tmp.jpg")
        if not cv2.imwrite(
            str(temporary_path),
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 76],
        ):
            raise RuntimeError("Impossible d’écrire l’aperçu live du tracking.")
        temporary_path.replace(output_path)

    @staticmethod
    def _bbox_iou(first, second) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _team_code(team) -> str:
        words = [word for word in str(team.name).replace("-", " ").split() if word]
        if len(words) >= 2:
            return "".join(word[0] for word in words[:4]).upper()
        return str(team.short_name or team.name)[:4].upper()

    def _tracking_windows(self, periods: list[MatchPeriod]) -> list[dict]:
        if self.analysis_mode not in {"reference", "sample"}:
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
            5_000,
            int(float(self.config.get("sample_window_seconds", 5)) * 1_000),
        )
        windows_per_half = max(
            1,
            min(4, int(self.config.get("sample_windows_per_half", 4))),
        )
        positions = (
            [0.50]
            if windows_per_half == 1
            else (
                [0.25, 0.70]
                if windows_per_half == 2
                else (
                    [0.18, 0.50, 0.78]
                    if windows_per_half == 3
                    else [0.10, 0.36, 0.64, 0.90]
                )
            )
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
        tracking_fps: float = 0.0,
        requested_tracking_fps: float = 0.0,
        model_classes: dict[str, str] | None = None,
        tracker_name: str = "",
        profile_name: str = "",
        image_size: int = 0,
        detector_confidence: float = 0.0,
        ball_confidence: float = 0.0,
        tracker_frame_rate: int = 0,
        team_calibration: dict | None = None,
        strict_validation: bool = False,
        frame_series: dict[str, list[int]] | None = None,
        period_counts: dict[int, Counter] | None = None,
        period_frame_series: dict[int, dict[str, list[int]]] | None = None,
        period_team_observations: dict[int, Counter] | None = None,
    ) -> dict:
        frames = max(int(counts["frames"]), 0)
        raw_athlete_observations = int(
            counts["raw_athlete_detections"]
            if "raw_athlete_detections" in counts
            else counts["athlete_observations"]
        )
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
        frame_series = frame_series or {}
        period_counts = period_counts or {}
        period_frame_series = period_frame_series or {}
        period_team_observations = period_team_observations or {}

        def value(scope: Counter, key: str, fallback: int = 0) -> int:
            return int(scope[key]) if key in scope else int(fallback)

        def distribution(values: list[int]) -> dict[str, int]:
            ordered = sorted(int(item) for item in values)
            if not ordered:
                return {"p10": 0, "median": 0, "p90": 0, "minimum": 0, "maximum": 0}

            def percentile(ratio: float) -> int:
                return ordered[int(round((len(ordered) - 1) * ratio))]

            return {
                "p10": percentile(0.10),
                "median": percentile(0.50),
                "p90": percentile(0.90),
                "minimum": ordered[0],
                "maximum": ordered[-1],
            }

        def stage_summary(
            label: str,
            scope: Counter,
            series: dict[str, list[int]],
            teams: Counter,
        ) -> dict:
            scope_frames = max(value(scope, "frames"), 1)
            raw = value(scope, "raw_athlete_detections")
            candidates = value(scope, "detector_candidate_athletes", raw)
            tracker_input = value(scope, "tracker_input_athletes", raw)
            tracker_output = value(
                scope,
                "tracker_output_athletes",
                value(scope, "athlete_observations"),
            )
            rescued = value(scope, "detector_rescue_athletes")
            identity_input = value(
                scope,
                "native_identity_input_athletes",
                tracker_output + rescued,
            )
            final = value(scope, "athlete_observations")
            known_teams = value(teams, "home") + value(teams, "away")
            final_distribution = distribution(
                list(series.get("athlete_observations") or [])
            )
            return {
                "label": label,
                "frames": value(scope, "frames"),
                "raw_per_frame": round(raw / scope_frames, 2),
                "candidates_per_frame": round(candidates / scope_frames, 2),
                "tracker_input_per_frame": round(tracker_input / scope_frames, 2),
                "tracker_output_per_frame": round(tracker_output / scope_frames, 2),
                "rescued_per_frame": round(rescued / scope_frames, 2),
                "identity_input_per_frame": round(identity_input / scope_frames, 2),
                "final_per_frame": round(final / scope_frames, 2),
                "dedup_keep_pct": round(100.0 * tracker_input / max(candidates, 1), 2),
                "tracker_confirmation_pct": round(
                    100.0 * tracker_output / max(tracker_input, 1), 2
                ),
                "identity_keep_pct": round(
                    100.0 * final / max(identity_input, 1), 2
                ),
                "end_to_end_pct": round(100.0 * final / max(raw, 1), 2),
                "empty_final_pct": round(
                    100.0
                    * sum(
                        int(item == 0)
                        for item in series.get("athlete_observations") or []
                    )
                    / scope_frames,
                    2,
                ),
                "final_distribution": final_distribution,
                "ball_visibility_pct": round(
                    100.0 * value(scope, "ball_visible_frames") / scope_frames,
                    2,
                ),
                "scene_cut_resets": value(scope, "scene_cut_tracker_resets"),
                "identity_births": value(scope, "native_identity_births"),
                "direct_rejections": value(scope, "native_direct_rejections"),
                "short_stitches": value(scope, "native_short_stitches"),
                "long_stitches": value(scope, "native_long_stitches"),
                "known_team_pct": round(
                    100.0 * known_teams / max(final, 1), 2
                ),
                "home_share_pct": round(
                    100.0 * value(teams, "home") / max(known_teams, 1), 2
                ),
                "away_share_pct": round(
                    100.0 * value(teams, "away") / max(known_teams, 1), 2
                ),
            }

        diagnostics = {
            "frames_analyzed": frames,
            "duration_seconds": round(tracking_duration_ms / 1_000, 1),
            "average_athletes_per_frame": round(
                raw_athlete_observations / max(frames, 1), 2
            ),
            "average_player_detections_per_frame": round(
                raw_athlete_observations / max(frames, 1), 2
            ),
            "average_tracked_athletes_per_frame": round(
                counts["athlete_observations"] / max(frames, 1), 2
            ),
            "tracker_retention_pct": round(
                100.0
                * counts["athlete_observations"]
                / max(raw_athlete_observations, 1),
                2,
            ),
            "tracker_dropped_athletes": int(counts["tracker_dropped_athletes"]),
            "ball_visibility_pct": round(
                100.0 * counts["ball_visible_frames"] / max(frames, 1), 2
            ),
            "pitch_metric_coverage_pct": round(
                100.0 * counts["pitch_metric_frames"] / max(frames, 1), 2
            ),
            "field_live_pct": round(100.0 * counts["field_frames"] / max(frames, 1), 2),
            "playable_candidate_pct": round(100.0 * playable_frames / max(frames, 1), 2),
            "tracks": track_count,
            "tracks_per_minute": round(track_count / duration_minutes, 2),
            "home_team_share_pct": round(home_share_pct, 2),
            "away_team_share_pct": round(away_share_pct, 2),
            "unknown_team_observations": int(team_observations["unknown"]),
            "raw_referee_detections": int(counts["raw_referee_detections"]),
            "raw_ball_detections": int(counts["raw_ball_detections"]),
            "raw_other_detections": int(counts["raw_other_detections"]),
            "rejected_person_detections": int(counts["rejected_person_detections"]),
            "duplicate_person_detections": int(counts["duplicate_person_detections"]),
            "tracking_fps": round(float(tracking_fps), 2),
            "requested_tracking_fps": round(float(requested_tracking_fps), 2),
            "model_classes": model_classes or {},
            "tracker": tracker_name,
            "profile": profile_name,
            "image_size": int(image_size),
            "detector_confidence": round(float(detector_confidence), 3),
            "ball_confidence": round(float(ball_confidence), 3),
            "tracker_frame_rate": int(tracker_frame_rate),
            "team_calibration": team_calibration or {},
            "issues": [],
        }
        total_stage = stage_summary(
            "Total", counts, frame_series, team_observations
        )
        stage_breakdown = [total_stage]
        for period_number in sorted(period_counts):
            stage_breakdown.append(
                stage_summary(
                    f"MT{period_number}",
                    period_counts[period_number],
                    period_frame_series.get(period_number, {}),
                    period_team_observations.get(period_number, Counter()),
                )
            )
        diagnostics["stage_breakdown"] = stage_breakdown
        diagnostics["tracker_input_per_frame"] = total_stage[
            "tracker_input_per_frame"
        ]
        diagnostics["tracker_output_per_frame"] = total_stage[
            "tracker_output_per_frame"
        ]
        diagnostics["detector_rescue_per_frame"] = total_stage[
            "rescued_per_frame"
        ]
        diagnostics["native_identity_input_per_frame"] = total_stage[
            "identity_input_per_frame"
        ]
        diagnostics["tracker_confirmation_pct"] = total_stage[
            "tracker_confirmation_pct"
        ]
        diagnostics["native_identity_keep_pct"] = total_stage[
            "identity_keep_pct"
        ]
        diagnostics["empty_final_frames_pct"] = total_stage[
            "empty_final_pct"
        ]
        diagnostics["final_player_distribution"] = total_stage[
            "final_distribution"
        ]
        diagnostics["athlete_confidence_bins"] = {
            "below_activation": value(
                counts, "athlete_confidence_below_activation"
            ),
            "activation_to_strong": value(
                counts, "athlete_confidence_activation_to_strong"
            ),
            "strong": value(counts, "athlete_confidence_strong"),
        }
        diagnostics["ball_selection_reasons"] = {
            key.removeprefix("ball_reason_"): int(count)
            for key, count in counts.items()
            if key.startswith("ball_reason_")
        }

        failures: list[str] = []
        warnings: list[str] = []
        if frames < 50:
            failures.append("Trop peu d’images ont été analysées.")
        if diagnostics["average_player_detections_per_frame"] < 6:
            failures.append("Moins de 6 joueurs sont détectés en moyenne par image.")
        elif diagnostics["average_tracked_athletes_per_frame"] < 6:
            failures.append(
                "YOLO détecte les joueurs, mais le tracker n’en conserve pas 6 par image."
            )
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
        calibration = diagnostics["team_calibration"]
        if profile_name == "native_gsr":
            hard_blockers: list[str] = []
            if calibration.get("status") != "ready":
                warnings.append(
                    "Native GSR n’a pas encore assez de tracklets pour stabiliser les deux équipes."
                )
            if calibration.get("appearance_backend") == "histogram":
                target = hard_blockers if strict_validation else warnings
                target.append(
                    "Le Re-ID profond est absent : les liaisons longues utilisent seulement couleur et texture."
                )
            if calibration.get("appearance_error"):
                warnings.append(str(calibration["appearance_error"]))
            jersey = calibration.get("jersey") or {}
            if not jersey.get("ready"):
                target = hard_blockers if strict_validation else warnings
                target.append(
                    "L’OCR des numéros de maillot n’est pas prêt : aucun checkpoint/ moteur OCR valide."
                )
            elif int(jersey.get("eligible_crops", 0)) >= 20 and int(
                calibration.get("jersey_tracklets", 0)
            ) == 0:
                warnings.append(
                    "L’OCR a vu assez de maillots mais n’a stabilisé aucun numéro."
                )
            if int(calibration.get("roster_entries", 0)) == 0:
                target = hard_blockers if strict_validation else warnings
                target.append(
                    "Les effectifs numérotés sont absents : l’identité roster ne peut pas être résolue."
                )
            pitch = calibration.get("pitch") or {}
            if not pitch.get("ready"):
                target = hard_blockers if strict_validation else warnings
                target.append(
                    "Le modèle de points terrain et son schéma 97 points ne sont pas prêts."
                )
            elif int(pitch.get("successful_calibrations", 0)) == 0:
                hard_blockers.append(
                    "Le modèle terrain a tourné mais aucune homographie fiable n’a été validée."
                )
            diagnostics["hard_blockers"] = hard_blockers
            failures.extend(hard_blockers)
            diagnostics["identity_flow"] = {
                "raw_scoped_ids": int(calibration.get("raw_tracks", 0)),
                "raw_track_segments": int(
                    calibration.get("raw_track_segments", 0)
                ),
                "canonical_births": int(
                    calibration.get(
                        "new_identity_observations",
                        calibration.get("canonical_tracks", 0),
                    )
                ),
                "direct_assignments": int(
                    calibration.get("direct_assignments", 0)
                ),
                "short_stitches": int(calibration.get("short_stitches", 0)),
                "long_stitches": int(calibration.get("long_stitches", 0)),
                "direct_rejections": int(
                    calibration.get("direct_track_rejections", 0)
                ),
                "direct_rejection_reasons": dict(
                    calibration.get("direct_rejection_reasons") or {}
                ),
                "scene_cut_resets": int(
                    calibration.get("scene_cut_tracker_resets", 0)
                ),
                "duplicates_removed": int(
                    calibration.get("duplicates_removed", 0)
                ),
            }
        diagnostics["issues"] = failures + warnings
        diagnostics["verdict"] = "fail" if failures else ("warning" if warnings else "pass")
        return diagnostics

    @staticmethod
    def _effective_tracking_fps(
        *,
        backend: str,
        requested_fps: float,
        native_fps: float,
        minimum_yolo_fps: float,
    ) -> float:
        effective = max(0.5, float(requested_fps))
        if backend.strip().lower() == "yolo":
            effective = max(effective, max(1.0, float(minimum_yolo_fps)))
        if native_fps > 0:
            effective = min(effective, float(native_fps))
        return round(effective, 3)

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
        engine_names = {
            "tracklab": "TRACKLAB + SN-GAMESTATE",
            "winner2025": "SOCCERNETGSR WINNER 2025",
        }
        engine = engine_names.get(backend, backend.upper())
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
            global_identity = obj.metadata.get("identity_scope") == "match"
            if (
                obj.track_id != "ball"
                and not global_identity
                and not obj.track_id.startswith(period_prefix)
            ):
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
            if obj.pitch_x is not None and obj.pitch_y is not None:
                analysis.coordinate_space = "pitch_meters"
            elif projector is not None:
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
                    "identity_confidence_sum": 0.0,
                    "samples": 0,
                    "team_votes": Counter(),
                    "shirt_votes": Counter(),
                    "role_votes": Counter(),
                    "roster_player_votes": Counter(),
                    "points": [],
                    "engine": obj.metadata.get("gsr_engine", "legacy"),
                    "source_track_id": obj.metadata.get("gsr_source_track_id"),
                    "identity_evidence": {},
                },
            )
            summary["end_ms"] = analysis.timestamp_ms
            summary["confidence_sum"] += obj.confidence
            identity_components = [obj.confidence]
            for key in (
                "reid_confidence",
                "role_confidence",
                "team_confidence",
                "jersey_confidence",
            ):
                value = obj.metadata.get(key)
                if value is not None:
                    identity_components.append(max(0.0, min(1.0, float(value))))
            summary["identity_confidence_sum"] += min(identity_components)
            summary["samples"] += 1
            summary["role_votes"][str(obj.role)] += max(0.05, float(obj.confidence))
            if obj.team_key:
                team_confidence = obj.metadata.get("team_confidence")
                summary["team_votes"][obj.team_key] += max(
                    0.05,
                    float(obj.confidence if team_confidence is None else team_confidence),
                )
            if obj.shirt_number is not None:
                jersey_confidence = obj.metadata.get("jersey_confidence")
                summary["shirt_votes"][obj.shirt_number] += max(
                    0.05,
                    float(obj.confidence if jersey_confidence is None else jersey_confidence),
                )
            roster_player_id = obj.metadata.get("roster_player_id")
            if roster_player_id is not None:
                jersey_confidence = obj.metadata.get("jersey_confidence")
                summary["roster_player_votes"][int(roster_player_id)] += max(
                    0.05,
                    float(obj.confidence if jersey_confidence is None else jersey_confidence),
                )
            summary["identity_evidence"].update(
                {
                    key: obj.metadata.get(key)
                    for key in (
                        "identity_engine",
                        "reid_backend",
                        "jersey_backend",
                        "roster_player_name",
                        "identity_windows",
                    )
                    if obj.metadata.get(key) not in {None, ""}
                }
            )
            if len(summary["points"]) < 2_000:
                summary["points"].append(
                    {
                        "t": analysis.timestamp_ms,
                        "x": obj.pitch_x if obj.pitch_x is not None else obj.image_x,
                        "y": obj.pitch_y if obj.pitch_y is not None else obj.image_y,
                        "space": "pitch_meters" if obj.pitch_x is not None else "image_normalized",
                    }
                )

    @staticmethod
    def _identity_alias(value: str | None, aliases: dict[str, str]) -> str | None:
        if value is None:
            return None
        resolved = str(value)
        visited: set[str] = set()
        while resolved in aliases and resolved not in visited:
            visited.add(resolved)
            resolved = aliases[resolved]
        return resolved

    @staticmethod
    def _timestamps_overlap(
        first: list[int],
        second: list[int],
        *,
        tolerance_ms: int = 120,
    ) -> bool:
        first = sorted(first)
        second = sorted(second)
        left = right = 0
        while left < len(first) and right < len(second):
            difference = first[left] - second[right]
            if abs(difference) <= tolerance_ms:
                return True
            if difference < 0:
                left += 1
            else:
                right += 1
        return False

    @classmethod
    def _consolidate_roster_tracks(cls, summaries: dict[str, dict]) -> dict[str, str]:
        """Merge non-simultaneous fragments with the same proven roster identity."""

        groups: dict[int, list[str]] = defaultdict(list)
        for uid, summary in summaries.items():
            roster_votes = summary.get("roster_player_votes") or Counter()
            if not roster_votes:
                continue
            player_id, score = roster_votes.most_common(1)[0]
            total = float(sum(roster_votes.values()))
            if float(score) < 0.8 or float(score) / max(total, 1e-9) < 0.72:
                continue
            groups[int(player_id)].append(uid)

        aliases: dict[str, str] = {}
        for uids in groups.values():
            ordered = sorted(uids, key=lambda uid: summaries[uid]["start_ms"])
            if len(ordered) < 2:
                continue
            target_uid = ordered[0]
            for source_uid in ordered[1:]:
                target = summaries[target_uid]
                source = summaries[source_uid]
                target_times = [int(point["t"]) for point in target.get("points", [])]
                source_times = [int(point["t"]) for point in source.get("points", [])]
                simultaneous = bool(
                    target_times
                    and source_times
                    and cls._timestamps_overlap(target_times, source_times)
                )
                if simultaneous:
                    continue
                aliases[source_uid] = target_uid
                target["start_ms"] = min(target["start_ms"], source["start_ms"])
                target["end_ms"] = max(target["end_ms"], source["end_ms"])
                target["confidence_sum"] += source["confidence_sum"]
                target["identity_confidence_sum"] += source.get(
                    "identity_confidence_sum", source["confidence_sum"]
                )
                target["samples"] += source["samples"]
                for key in (
                    "team_votes",
                    "shirt_votes",
                    "role_votes",
                    "roster_player_votes",
                ):
                    target[key].update(source.get(key) or {})
                target["identity_evidence"].update(
                    source.get("identity_evidence") or {}
                )
                target["points"] = sorted(
                    [*target.get("points", []), *source.get("points", [])],
                    key=lambda point: int(point["t"]),
                )[:2_000]
                summaries.pop(source_uid, None)
        return aliases

    @classmethod
    def _rewrite_tracking_identities(
        cls,
        tracking_path: Path,
        aliases: dict[str, str],
    ) -> None:
        rewritten = tracking_path.with_suffix(".identities.ndjson")
        with tracking_path.open("r", encoding="utf-8") as source, rewritten.open(
            "w", encoding="utf-8"
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                payload = json.loads(line)
                frame = payload.get("frame") or {}
                for obj in frame.get("objects") or []:
                    obj["track_id"] = cls._identity_alias(
                        obj.get("track_id"), aliases
                    )
                    obj["player_key"] = cls._identity_alias(
                        obj.get("player_key"), aliases
                    )
                possession = payload.get("possession") or {}
                possession["player_key"] = cls._identity_alias(
                    possession.get("player_key"), aliases
                )
                target.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        rewritten.replace(tracking_path)

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
        roster_players = {
            player.pk: player
            for player in self.match.home_team.players.filter(active=True)
        }
        roster_players.update(
            {
                player.pk: player
                for player in self.match.away_team.players.filter(active=True)
            }
        )
        for uid, summary in result["tracks"].items():
            team_key = self._winner(summary["team_votes"])
            shirt_number = self._winner(summary["shirt_votes"])
            role = self._winner(summary.get("role_votes", Counter())) or summary["role"]
            roster_player_id = self._winner(
                summary.get("roster_player_votes", Counter())
            )
            player = roster_players.get(roster_player_id)
            if player is None and team_key and shirt_number is not None:
                resolved_team = self.team_by_key.get(team_key)
                candidates = [
                    candidate
                    for candidate in roster_players.values()
                    if resolved_team is not None
                    and candidate.team_id == resolved_team.pk
                    and candidate.shirt_number == shirt_number
                ]
                player = candidates[0] if len(candidates) == 1 else None
            samples = max(int(summary["samples"]), 1)
            tracks.append(
                Track(
                    match=self.match,
                    analysis_run=self.run,
                    track_uid=uid,
                    role=role if role in Track.Role.values else Track.Role.OTHER,
                    team=self.team_by_key.get(team_key),
                    player=player,
                    predicted_shirt_number=shirt_number,
                    identity_confidence=round(
                        summary.get("identity_confidence_sum", summary["confidence_sum"])
                        / samples,
                        4,
                    ),
                    video_start_ms=summary["start_ms"],
                    video_end_ms=summary["end_ms"],
                    metadata={
                        "samples": samples,
                        "points": summary["points"],
                        "engine": summary.get("engine", "legacy"),
                        "source_track_id": summary.get("source_track_id"),
                        "identity_evidence": summary.get("identity_evidence", {}),
                        "identity_auto_resolved": player is not None,
                    },
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
