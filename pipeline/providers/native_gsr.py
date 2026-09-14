from __future__ import annotations

from pipeline.pitch import NativePitchCalibrator
from pipeline.types import FrameAnalysis, ObjectRole

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
                "tracker_name": str(config.get("tracker_name", "bytetrack")),
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
            global_max_gap_seconds=float(
                config.get("native_gsr_global_max_gap_seconds", 7_200.0)
            ),
            reid_model_path=str(config.get("native_gsr_reid_model_path", "")),
            reid_backend=str(config.get("native_gsr_reid_backend", "auto")),
            reid_model_name=str(
                config.get("native_gsr_reid_model_name", "osnet_x0_25")
            ),
            jersey_engine=str(config.get("native_gsr_jersey_engine", "auto")),
            jersey_model_path=str(
                config.get("native_gsr_jersey_model_path", "")
            ),
            jersey_device=str(config.get("native_gsr_jersey_device", "cpu")),
            jersey_interval_frames=int(
                config.get("native_gsr_jersey_interval_frames", 12)
            ),
            jersey_minimum_box_height=int(
                config.get("native_gsr_jersey_minimum_box_height", 72)
            ),
            jersey_max_crops_per_frame=int(
                config.get("native_gsr_jersey_max_crops_per_frame", 4)
            ),
            roster=list(config.get("native_gsr_roster") or []),
            device=str(config.get("device", "cpu")),
        )
        self.pitch = NativePitchCalibrator(
            model_path=str(config.get("native_gsr_pitch_model_path", "")),
            schema_path=str(config.get("native_gsr_pitch_schema_path", "")),
            confidence=float(config.get("native_gsr_pitch_confidence", 0.20)),
            interval_frames=int(
                config.get("native_gsr_pitch_interval_frames", 10)
            ),
            hold_frames=int(config.get("native_gsr_pitch_hold_frames", 20)),
            expected_landmarks=int(
                config.get("native_gsr_pitch_expected_landmarks", 97)
            ),
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
        self.tracker_name = f"{self.base.tracker_name}+native_reid"
        self.adaptive_resizes: list[dict[str, int]] = []
        self.off_pitch_rejections = 0
        self.previous_scene_gray = None
        self.scene_cut_resets = 0

    def reset(self) -> None:
        self.base.reset()
        self.refiner.reset_window()
        self.pitch.reset_shot()
        self.previous_scene_gray = None

    def _detect_scene_cut(self, frame) -> bool:
        if frame is None or not hasattr(frame, "shape"):
            return False
        import cv2

        _height, width = frame.shape[:2]
        scale = min(1.0, 640.0 / max(width, 1))
        working = (
            cv2.resize(frame, None, fx=scale, fy=scale)
            if scale < 1.0
            else frame
        )
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        previous = getattr(self, "previous_scene_gray", None)
        self.previous_scene_gray = gray
        if previous is None or previous.shape != gray.shape:
            return False
        return float(cv2.absdiff(gray, previous).mean() / 255.0) > 0.32

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
        scene_cut = self._detect_scene_cut(frame)
        if scene_cut:
            self.base.reset_tracking_state()
            self.refiner.reset_window(scene_cut=True)
            self.pitch.reset_shot()
            self.scene_cut_resets += 1
        while True:
            try:
                analysis = self.base.analyze_frame(frame, timestamp_ms)
                break
            except RuntimeError as exc:
                if not self._is_memory_error(exc) or not self._reduce_image_size():
                    raise
        analysis.scene_cut = bool(analysis.scene_cut or scene_cut)
        before = len(analysis.objects)
        pitch_engine = getattr(self, "pitch", None)
        pitch_ready = bool(
            pitch_engine and pitch_engine.update(frame, scene_cut=analysis.scene_cut)
        )
        pitch_objects = (
            pitch_engine.project_boxes(analysis.objects) if pitch_ready else 0
        )
        if pitch_objects:
            analysis.coordinate_space = "pitch_meters"
        rejected_this_frame = 0
        if pitch_ready:
            retained = []
            for obj in analysis.objects:
                outside = obj.metadata.get("pitch_outside_distance_m")
                maximum = (
                    6.0
                    if obj.role == str(ObjectRole.REFEREE)
                    else 3.0
                )
                if (
                    outside is not None
                    and obj.role
                    in {
                        str(ObjectRole.PLAYER),
                        str(ObjectRole.GOALKEEPER),
                        str(ObjectRole.REFEREE),
                    }
                    and float(outside) > maximum
                ):
                    self.off_pitch_rejections += 1
                    rejected_this_frame += 1
                    continue
                retained.append(obj)
            analysis.objects = retained
        analysis.objects = self.refiner.process(frame, analysis.objects, timestamp_ms)
        identity = self.refiner.diagnostics()
        pitch = pitch_engine.diagnostics() if pitch_engine else {"backend": "disabled"}
        removed = max(0, before - len(analysis.objects))
        analysis.diagnostics["duplicate_person_detections"] = int(
            analysis.diagnostics.get("duplicate_person_detections", 0)
        ) + removed
        analysis.diagnostics["tracked_athletes"] = len(analysis.athletes)
        analysis.diagnostics["tracker"] = self.tracker_name
        analysis.diagnostics["profile"] = self.profile
        analysis.diagnostics["adaptive_image_resizes"] = list(self.adaptive_resizes)
        analysis.diagnostics["image_size"] = self.image_size
        analysis.diagnostics["scene_cut_tracker_resets"] = int(
            getattr(self, "scene_cut_resets", 0)
        )
        analysis.diagnostics["native_identity"] = identity
        analysis.diagnostics["pitch_calibration"] = pitch
        analysis.diagnostics["rejected_person_detections"] = int(
            analysis.diagnostics.get("rejected_person_detections", 0)
        ) + rejected_this_frame
        analysis.diagnostics["team_calibration"] = {**identity, "pitch": pitch}
        return analysis

    def team_calibration_diagnostics(self) -> dict:
        return {
            **self.refiner.diagnostics(),
            "pitch": self.pitch.diagnostics(),
            "ball": self.base.ball_recovery_diagnostics(),
            "off_pitch_rejections": self.off_pitch_rejections,
            "scene_cut_tracker_resets": self.scene_cut_resets,
        }
