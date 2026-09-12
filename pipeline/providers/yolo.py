from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

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
        tracking_fps: float = 10.0,
        tracker_name: str = "botsort",
        tracker_low_confidence: float = 0.10,
        tracker_new_confidence: float = 0.35,
        tracker_match_threshold: float = 0.85,
        tracker_buffer_seconds: float = 5.0,
        player_class_ids: list[int] | tuple[int, ...] | None = None,
        goalkeeper_class_ids: list[int] | tuple[int, ...] | None = None,
        referee_class_ids: list[int] | tuple[int, ...] | None = None,
        ball_class_ids: list[int] | tuple[int, ...] | None = None,
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
        self.tracking_fps = max(1.0, float(tracking_fps))
        self.tracker_name = str(tracker_name or "botsort").strip().lower()
        if self.tracker_name not in {"botsort", "bytetrack"}:
            raise ValueError(
                f"Tracker YOLO inconnu : {self.tracker_name}. "
                "Valeurs acceptées : botsort, bytetrack."
            )
        self.tracker_low_confidence = max(
            0.01, min(float(tracker_low_confidence), float(confidence))
        )
        self.tracker_new_confidence = max(
            self.tracker_low_confidence, float(tracker_new_confidence)
        )
        self.tracker_match_threshold = max(
            0.1, min(0.99, float(tracker_match_threshold))
        )
        self.tracker_buffer_seconds = max(1.0, float(tracker_buffer_seconds))
        self.class_roles: dict[int, ObjectRole] = {}
        self._register_class_ids(player_class_ids, ObjectRole.PLAYER)
        self._register_class_ids(goalkeeper_class_ids, ObjectRole.GOALKEEPER)
        self._register_class_ids(referee_class_ids, ObjectRole.REFEREE)
        self._register_class_ids(ball_class_ids, ObjectRole.BALL)
        self.team_colors = {
            key: self._hex_to_lab(value) for key, value in (team_colors or {}).items()
        }
        self.previous_gray = None
        self.tracker = self._build_tracker()

    def _build_tracker(self):
        if self.tracker_name == "botsort":
            return self._build_botsort()
        return self._build_bytetrack()

    def _build_botsort(self):
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError as exc:
            raise RuntimeError(
                "BoT-SORT indisponible. Mets à jour requirements-ml.txt."
            ) from exc
        return BOTSORT(
            self._botsort_args(),
            # Ultralytics 8.3 scales ``track_buffer`` by frame_rate / 30 while
            # newer releases store it directly in frames. Passing 30 keeps the
            # configured frame count identical on both implementations.
            frame_rate=30,
        )

    def _botsort_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=float(self.confidence),
            track_low_thresh=self.tracker_low_confidence,
            new_track_thresh=self.tracker_new_confidence,
            track_buffer=max(
                15, int(round(self.tracking_fps * self.tracker_buffer_seconds))
            ),
            match_thresh=self.tracker_match_threshold,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )

    def _build_bytetrack(self):
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "Le tracking YOLO exige le paquet supervision de requirements-ml.txt."
            ) from exc
        return sv.ByteTrack(
            track_activation_threshold=self.tracker_new_confidence,
            lost_track_buffer=max(
                15, int(round(self.tracking_fps * self.tracker_buffer_seconds))
            ),
            minimum_matching_threshold=self.tracker_match_threshold,
            frame_rate=max(1, int(round(self.tracking_fps))),
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
            # Keep low-confidence player boxes for the tracker's recovery pass.
            # Reported detector recall still uses ``self.confidence`` below.
            conf=self.tracker_low_confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        names = prediction.names
        detections = sv.Detections.from_ultralytics(prediction)

        roles = (
            [self._role_for(int(class_id), names) for class_id in detections.class_id]
            if len(detections)
            else []
        )
        confidences = (
            [float(value) for value in prediction.boxes.conf.detach().cpu().tolist()]
            if len(prediction.boxes)
            else []
        )
        trackable_roles = {
            ObjectRole.PLAYER,
            ObjectRole.GOALKEEPER,
            ObjectRole.REFEREE,
        }
        trackable_indices = [
            index for index, role in enumerate(roles) if role in trackable_roles
        ]
        tracked_rows = self._update_tracker(
            prediction,
            detections,
            trackable_indices,
            frame,
        )

        objects: list[TrackedObject] = []
        for xyxy, confidence, class_id, tracker_id in tracked_rows:
            role = self._role_for(int(class_id), names)
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
            ball_candidates = []
            for xyxy, confidence, class_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
            ):
                if (
                    self._role_for(int(class_id), names) != ObjectRole.BALL
                    or float(confidence) < self.confidence
                ):
                    continue
                ball_candidates.append((xyxy, confidence))
            if ball_candidates:
                xyxy, confidence = max(
                    ball_candidates, key=lambda candidate: float(candidate[1])
                )
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
            diagnostics={
                "raw_athlete_detections": sum(
                    role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_referee_detections": sum(
                    role == ObjectRole.REFEREE and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_ball_detections": sum(
                    role == ObjectRole.BALL and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_other_detections": sum(
                    role == ObjectRole.OTHER and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_athlete_boxes": [
                    [float(value) for value in xyxy]
                    for xyxy, role, confidence in zip(
                        detections.xyxy, roles, confidences
                    )
                    if role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    and confidence >= self.confidence
                ],
                "tracked_athletes": sum(
                    item.role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    for item in objects
                ),
                "model_classes": {
                    str(key): str(value)
                    for key, value in (
                        names.items() if hasattr(names, "items") else enumerate(names)
                    )
                },
                "tracker": self.tracker_name,
            },
        )

    def _update_tracker(self, prediction, detections, indices, frame):
        if self.tracker_name == "botsort":
            boxes = prediction.boxes[indices] if indices else prediction.boxes[:0]
            boxes = boxes.cpu().numpy()
            tracks = self.tracker.update(boxes, frame)
            return [
                (
                    row[:4],
                    float(row[5]),
                    int(row[6]),
                    int(row[4]),
                )
                for row in tracks
                if len(row) >= 7
            ]

        import supervision as sv

        athlete_mask = np.zeros(len(detections), dtype=bool)
        athlete_mask[indices] = True
        athletes = detections[athlete_mask] if len(detections) else detections
        tracked = self.tracker.update_with_detections(athletes)
        return list(
            zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.class_id,
                tracked.tracker_id,
            )
        )

    def _register_class_ids(
        self,
        class_ids: list[int] | tuple[int, ...] | None,
        role: ObjectRole,
    ) -> None:
        for class_id in class_ids or []:
            self.class_roles[int(class_id)] = role

    def _role_for(self, class_id: int, names) -> ObjectRole:
        explicit = self.class_roles.get(int(class_id))
        if explicit is not None:
            return explicit
        try:
            name = names[int(class_id)]
        except (IndexError, KeyError, TypeError):
            return ObjectRole.OTHER
        return self._role(name)

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
