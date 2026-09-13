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
        self.adaptive_resizes: list[dict[str, int]] = []

    def reset(self) -> None:
        self.base.reset()
        self.refiner.reset()

    @staticmethod
    def _is_memory_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "not enough memory" in message

    def _reduce_image_size(self) -> bool:
        current = int(self.base.image_size)
        next_size = next((size for size in (960, 768, 640) if size < current), None)
        if next_size is None:
            return False
        self.adaptive_resizes.append({"from": current, "to": next_size})
        self.base.image_size = next_size
        self.image_size = next_size
        try:  # pragma: no cover - depends on the local CUDA runtime
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return True

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        while True:
            try:
                analysis = self.base.analyze_frame(frame, timestamp_ms)
                break
            except RuntimeError as exc:
                if not self._is_memory_error(exc) or not self._reduce_image_size():
                    raise
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
        analysis.diagnostics["adaptive_image_resizes"] = list(self.adaptive_resizes)
        analysis.diagnostics["image_size"] = self.image_size
        analysis.diagnostics["native_identity"] = identity
        analysis.diagnostics["team_calibration"] = identity
        return analysis

    def team_calibration_diagnostics(self) -> dict:
        return self.refiner.diagnostics()
