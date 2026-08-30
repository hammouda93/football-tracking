from __future__ import annotations

import json
import logging
import subprocess
import tempfile
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
        provider = build_provider(
            backend,
            model_path=self.config.get("yolo_model_path", ""),
            device=self.config.get("device", "cpu"),
            confidence=float(self.config.get("yolo_confidence", 0.30)),
            image_size=int(self.config.get("yolo_image_size", 1280)),
            team_colors={
                "home": self.match.home_team.primary_color,
                "away": self.match.away_team.primary_color,
            },
        )
        tracking_fps = max(0.5, float(self.config.get("tracking_fps", 10.0)))
        period_duration = sum(
            max(0, period.video_end_ms - period.video_start_ms) for period in periods
        )
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

        with tempfile.TemporaryDirectory(prefix="football-tracking-") as temp_dir:
            tracking_path = Path(temp_dir) / "tracking.ndjson"
            with tracking_path.open("w", encoding="utf-8") as tracking_file:
                for period in periods:
                    self._check_cancelled()
                    provider.reset()
                    camera = CameraStabilizer()
                    projector = self._projector(period)
                    ball_engine = BallInPlayEngine()
                    period_samples: list[PossessionSample] = []
                    period_prefix = f"p{period.number}-"
                    last_progress_update = -1
                    for timestamp_ms, frame in iter_frames(
                        metadata.path,
                        start_ms=period.video_start_ms,
                        end_ms=period.video_end_ms,
                        target_fps=tracking_fps,
                    ):
                        analysis = provider.analyze_frame(frame, timestamp_ms)
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
                        tracking_file.write(
                            json.dumps(
                                {
                                    "period": period.number,
                                    "match_time_ms": self._match_time(period, timestamp_ms),
                                    "frame": analysis.to_dict(),
                                    "possession": sample.to_dict(),
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )
                        current_processed = processed_ms + timestamp_ms - period.video_start_ms
                        percent = 22 + int(46 * current_processed / max(period_duration, 1))
                        if percent != last_progress_update:
                            self._stage(AnalysisRun.Stage.TRACKING, percent)
                            last_progress_update = percent
                    processed_ms += max(0, period.video_end_ms - period.video_start_ms)
                    spans = ball_engine.compress(
                        period_samples,
                        max_gap_ms=max(1_000, int(2_500 / tracking_fps)),
                    )
                    for span in spans:
                        span.end_ms = min(period.video_end_ms, span.end_ms)
                    all_spans.extend((period, span) for span in spans)
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
        return {
            "backend": backend,
            "tracking_fps": tracking_fps,
            "samples": all_samples,
            "spans": all_spans,
            "events": all_events,
            "tracks": track_summaries,
            "camera": camera_summary,
        }

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
        self._stats(result)
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
            "video": metadata.to_dict(),
            "quality": {"score": quality.score, "grade": quality.grade, **quality.metrics},
            "periods": [self._period_payload(period) for period in periods],
            "backend": result["backend"],
            "tracking_fps": result["tracking_fps"],
            "camera": result["camera"],
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
            result["backend"] == "heuristic"
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
