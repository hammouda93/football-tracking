from __future__ import annotations

from pipeline.types import FrameAnalysis

from .base import VisionProvider
from .native_identity import NativeIdentityRefiner
from .yolo import YoloVisionProvider


class NativeGSRVisionProvider(VisionProvider):
    """Windows-native GSR profile: detector + GMC tracker + IDATR-like refinement."""

    def __init__(self, **config):
        robust_config = dict(config)
        robust_config.update(
            {
                "profile": "advanced",
                "tracker_name": "botsort",
                "confidence": float(config.get("confidence", 0.18)),
                "tracker_low_confidence": float(
                    config.get("tracker_low_confidence", 0.05)
                ),
                "tracker_new_confidence": float(
                    config.get("tracker_new_confidence", 0.20)
                ),
                "tracker_match_threshold": float(
                    config.get("tracker_match_threshold", 0.80)
                ),
                "tracker_buffer_seconds": float(
                    config.get("tracker_buffer_seconds", 4.8)
                ),
                "image_size": int(config.get("image_size", 1280)),
            }
        )
        self.base = YoloVisionProvider(**robust_config)
        self.refiner = NativeIdentityRefiner(
            home_team_cluster=str(config.get("home_team_cluster", "B")),
            max_gap_seconds=float(config.get("native_gsr_max_gap_seconds", 3.0)),
            reid_model_path=str(config.get("native_gsr_reid_model_path", "")),
            device=str(config.get("device", "cpu")),
        )
        self.profile = "native_gsr"
        for attribute in (
            "confidence",
            "ball_confidence",
            "image_size",
            "tracking_fps",
            "tracker_frame_rate",
        ):
            setattr(self, attribute, getattr(self.base, attribute))
        self.tracker_name = "botsort+native_reid"

    def reset(self) -> None:
        self.base.reset()
        self.refiner.reset()

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        analysis = self.base.analyze_frame(frame, timestamp_ms)
        before = len(analysis.objects)
        analysis.objects = self.refiner.process(frame, analysis.objects, timestamp_ms)
        identity = self.refiner.diagnostics()
        removed = max(0, before - len(analysis.objects))
        analysis.diagnostics["duplicate_person_detections"] = int(
            analysis.diagnostics.get("duplicate_person_detections", 0)
        ) + removed
        analysis.diagnostics["tracked_athletes"] = len(analysis.athletes)
        analysis.diagnostics["tracker"] = self.tracker_name
        analysis.diagnostics["profile"] = self.profile
        analysis.diagnostics["native_identity"] = identity
        analysis.diagnostics["team_calibration"] = identity
        return analysis

    def team_calibration_diagnostics(self) -> dict:
        return self.refiner.diagnostics()
