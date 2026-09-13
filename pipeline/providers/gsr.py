from __future__ import annotations

from pipeline.gsr.contract import GSRFrameStore
from pipeline.types import FrameAnalysis, ObjectRole

from .base import VisionProvider


class ExternalGSRVisionProvider(VisionProvider):
    """Use external GSR athletes and the existing local ball detector together."""

    def __init__(
        self,
        store: GSRFrameStore,
        *,
        tolerance_ms: int,
        ball_provider: VisionProvider | None = None,
    ):
        self.store = store
        self.tolerance_ms = max(1, int(tolerance_ms))
        self.ball_provider = ball_provider
        self.profile = f"gsr:{store.engine}"
        self.tracker_frame_rate = round(store.fps)
        self.image_size = 0
        self.confidence = 0.0
        self.ball_confidence = float(getattr(ball_provider, "ball_confidence", 0.0))

    def reset(self) -> None:
        if self.ball_provider is not None:
            self.ball_provider.reset()

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        height, width = frame.shape[:2]
        external = self.store.nearest(timestamp_ms, tolerance_ms=self.tolerance_ms)
        if external is None:
            external = FrameAnalysis(
                timestamp_ms=timestamp_ms,
                width=width,
                height=height,
                field_score=1.0,
                diagnostics={
                    "gsr_engine": self.store.engine,
                    "gsr_frame_missing": True,
                    "raw_athlete_detections": 0,
                    "raw_referee_detections": 0,
                    "raw_detections": [],
                },
            )
        elif external.width != width or external.height != height:
            raise RuntimeError(
                "Les dimensions du résultat GSR ne correspondent pas à la vidéo source "
                f"({external.width}x{external.height} contre {width}x{height})."
            )

        local = (
            self.ball_provider.analyze_frame(frame, timestamp_ms)
            if self.ball_provider is not None
            else None
        )
        if local is not None:
            balls = [item for item in local.objects if item.role == ObjectRole.BALL]
            external.objects.extend(balls[:1])
            external.field_score = local.field_score
            external.scene_cut = local.scene_cut
            external.replay_probability = local.replay_probability
            external.diagnostics["raw_ball_detections"] = int(
                local.diagnostics.get("raw_ball_detections", len(balls))
            )
            external.diagnostics["raw_other_detections"] = int(
                local.diagnostics.get("raw_other_detections", 0)
            )
            external.diagnostics["model_classes"] = dict(
                local.diagnostics.get("model_classes") or {}
            )
            external.diagnostics["raw_detections"].extend(
                item
                for item in (local.diagnostics.get("raw_detections") or [])
                if item.get("role") == "ball"
            )
        external.diagnostics.update(
            {
                "tracker": f"gsr:{self.store.engine}",
                "gsr_engine_revision": self.store.engine_revision,
            }
        )
        return external

    def team_calibration_diagnostics(self) -> dict:
        return {
            "status": "external_gsr",
            "source": "tracklet_team_voting",
            "engine": self.store.engine,
            "samples": sum(len(frame.athletes) for frame in self.store.frames),
            "uses_entered_team_colors": False,
        }
