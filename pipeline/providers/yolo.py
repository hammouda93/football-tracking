from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pipeline.types import FrameAnalysis, ObjectRole, TrackedObject

from .base import VisionProvider


ROLE_ALIASES = {
    "player": ObjectRole.PLAYER,
    "players": ObjectRole.PLAYER,
    "goalkeeper": ObjectRole.GOALKEEPER,
    "goalie": ObjectRole.GOALKEEPER,
    "keeper": ObjectRole.GOALKEEPER,
    "referee": ObjectRole.REFEREE,
    "ref": ObjectRole.REFEREE,
    "ball": ObjectRole.BALL,
    "football": ObjectRole.BALL,
    "soccer-ball": ObjectRole.BALL,
}


class YoloVisionProvider(VisionProvider):
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.3,
        image_size: int = 1280,
        team_colors: dict[str, str] | None = None,
        **_: object,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Poids YOLO absents : {model_path}. Consulte models/README.md."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Le backend YOLO exige requirements-ml.txt."
            ) from exc
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        self.team_colors = {
            key: self._hex_to_lab(value) for key, value in (team_colors or {}).items()
        }
        self.previous_gray = None
        self.tracker = self._build_tracker()

    @staticmethod
    def _build_tracker():
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "Le tracking YOLO exige le paquet supervision de requirements-ml.txt."
            ) from exc
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,
            minimum_matching_threshold=0.75,
            frame_rate=30,
        )

    def reset(self) -> None:
        self.previous_gray = None
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        else:
            self.tracker = self._build_tracker()

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        import cv2
        import supervision as sv

        height, width = frame.shape[:2]
        prediction = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        names = prediction.names
        detections = sv.Detections.from_ultralytics(prediction)

        athlete_mask = np.array(
            [self._role(names[int(class_id)]) != ObjectRole.BALL for class_id in detections.class_id],
            dtype=bool,
        ) if len(detections) else np.array([], dtype=bool)
        athletes = detections[athlete_mask] if len(detections) else detections
        tracked_athletes = self.tracker.update_with_detections(athletes)

        objects: list[TrackedObject] = []
        for xyxy, confidence, class_id, tracker_id in zip(
            tracked_athletes.xyxy,
            tracked_athletes.confidence,
            tracked_athletes.class_id,
            tracked_athletes.tracker_id,
        ):
            role = self._role(names[int(class_id)])
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            image_x = ((x1 + x2) / 2.0) / max(width, 1)
            image_y = y2 / max(height, 1)
            team_key = None
            if role == ObjectRole.PLAYER:
                team_key = self._classify_team(frame, (x1, y1, x2, y2))
            track_id = f"athlete-{int(tracker_id)}"
            objects.append(
                TrackedObject(
                    track_id=track_id,
                    role=str(role),
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    team_key=team_key,
                    player_key=track_id,
                    image_x=image_x,
                    image_y=image_y,
                )
            )

        if len(detections):
            for xyxy, confidence, class_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
            ):
                if self._role(names[int(class_id)]) != ObjectRole.BALL:
                    continue
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                objects.append(
                    TrackedObject(
                        track_id="ball",
                        role=str(ObjectRole.BALL),
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        image_x=((x1 + x2) / 2.0) / max(width, 1),
                        image_y=((y1 + y2) / 2.0) / max(height, 1),
                    )
                )

        field_score, scene_cut = self._field_and_cut(frame)
        return FrameAnalysis(
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            field_score=field_score,
            objects=objects,
            scene_cut=scene_cut,
            replay_probability=0.72 if scene_cut and field_score < 0.2 else 0.0,
        )

    def _field_and_cut(self, frame) -> tuple[float, bool]:
        import cv2

        height, width = frame.shape[:2]
        scale = min(1.0, 640.0 / max(width, 1))
        working = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([100, 255, 255]))
        field_score = min(1.0, float(np.count_nonzero(mask) / mask.size) * 1.6)
        scene_cut = False
        if self.previous_gray is not None:
            scene_cut = float(cv2.absdiff(gray, self.previous_gray).mean() / 255.0) > 0.32
        self.previous_gray = gray
        return field_score, scene_cut

    def _classify_team(self, frame, box: tuple[float, float, float, float]) -> str | None:
        if len(self.team_colors) < 2:
            return None
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        left = max(0, min(width - 1, int(x1)))
        right = max(left + 1, min(width, int(x2)))
        top = max(0, min(height - 1, int(y1)))
        bottom = max(top + 1, min(height, int(y1 + (y2 - y1) * 0.62)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        usable = (hsv[:, 1] >= 35) & ~((hsv[:, 0] >= 25) & (hsv[:, 0] <= 100))
        pixels = lab[usable]
        if len(pixels) < 8:
            pixels = lab
        median = np.median(pixels, axis=0)
        return min(
            self.team_colors,
            key=lambda key: float(np.linalg.norm(median - self.team_colors[key])),
        )

    @staticmethod
    def _role(name: str) -> ObjectRole:
        normalized = str(name).strip().lower().replace("_", "-")
        return ROLE_ALIASES.get(normalized, ObjectRole.OTHER)

    @staticmethod
    def _hex_to_lab(value: str):
        import cv2

        value = value.lstrip("#")
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
        pixel = np.uint8([[[blue, green, red]]])
        return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
