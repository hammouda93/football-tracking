from __future__ import annotations

import inspect
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
        self.previous_ball_center: tuple[float, float] | None = None
        self.previous_ball_timestamp_ms: int | None = None
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
        return self._instantiate_botsort(BOTSORT, self._botsort_args())

    @staticmethod
    def _instantiate_botsort(tracker_class, args):
        """Build BoT-SORT across Ultralytics constructor versions.

        Older 8.x releases accept ``frame_rate`` and scale ``track_buffer`` by
        it. Newer 8.x releases removed that parameter and consume the buffer in
        frames directly. A value of 30 preserves the configured frame count on
        the older implementation.
        """

        try:
            parameters = inspect.signature(tracker_class).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "frame_rate" in parameters:
            return tracker_class(args, frame_rate=30)
        return tracker_class(args)

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
        self.previous_ball_center = None
        self.previous_ball_timestamp_ms = None
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
        raw_trackable_indices = [
            index for index, role in enumerate(roles) if role in trackable_roles
        ]
        deduplicated_indices = self._deduplicate_indices(
            detections.xyxy,
            confidences,
            raw_trackable_indices,
        )
        trackable_indices: list[int] = []
        rejected_person_boxes: list[list[float]] = []
        for index in deduplicated_indices:
            role = roles[index]
            box = detections.xyxy[index]
            valid_shape = self._valid_person_box(frame.shape, box)
            on_field = role == ObjectRole.REFEREE or self._box_on_field(frame, box)
            if valid_shape and on_field:
                trackable_indices.append(index)
            elif confidences[index] >= self.confidence:
                rejected_person_boxes.append([float(value) for value in box])
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
        objects = self._deduplicate_tracked_objects(objects)

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
            selected_ball = self._select_ball(
                ball_candidates,
                objects,
                frame.shape,
                timestamp_ms,
            )
            if selected_ball is not None:
                xyxy, confidence = selected_ball
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
                    roles[index] in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    and confidences[index] >= self.confidence
                    for index in trackable_indices
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
                    [float(value) for value in detections.xyxy[index]]
                    for index in trackable_indices
                    if roles[index] in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    and confidences[index] >= self.confidence
                ],
                "rejected_person_boxes": rejected_person_boxes,
                "rejected_person_detections": len(rejected_person_boxes),
                "duplicate_person_detections": max(
                    0, len(raw_trackable_indices) - len(deduplicated_indices)
                ),
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

    @staticmethod
    def _box_iou(first, second) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _deduplicate_indices(
        cls,
        boxes,
        confidences: list[float],
        indices: list[int],
        threshold: float = 0.82,
    ) -> list[int]:
        kept: list[int] = []
        for index in sorted(indices, key=lambda item: confidences[item], reverse=True):
            if any(cls._box_iou(boxes[index], boxes[other]) >= threshold for other in kept):
                continue
            kept.append(index)
        return kept

    @classmethod
    def _deduplicate_tracked_objects(
        cls,
        objects: list[TrackedObject],
        threshold: float = 0.80,
    ) -> list[TrackedObject]:
        kept: list[TrackedObject] = []
        for item in sorted(objects, key=lambda obj: obj.confidence, reverse=True):
            if any(
                cls._box_iou(item.bbox_xyxy, other.bbox_xyxy) >= threshold
                for other in kept
            ):
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _valid_person_box(frame_shape, box) -> bool:
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return (
            height >= max(12.0, frame_height * 0.012)
            and height >= width * 1.08
            and width <= frame_width * 0.18
            and height <= frame_height * 0.65
        )

    @staticmethod
    def _box_on_field(frame, box) -> bool:
        import cv2

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        center_x = int((x1 + x2) / 2.0)
        foot_y = int(y2)
        radius_x = max(7, int(width * 0.75))
        radius_y = max(5, int(height * 0.12))
        left = max(0, center_x - radius_x)
        right = min(frame_width, center_x + radius_x + 1)
        top = max(0, foot_y - radius_y)
        bottom = min(frame_height, foot_y + radius_y + 1)
        patch = frame[top:bottom, left:right]
        if patch.size == 0:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 25)
            & (hsv[:, :, 0] <= 100)
            & (hsv[:, :, 1] >= 20)
            & (hsv[:, :, 2] >= 18)
        )
        return float(np.count_nonzero(green) / green.size) >= 0.12

    def _select_ball(self, candidates, athletes, frame_shape, timestamp_ms: int):
        if not candidates:
            return None
        frame_height, frame_width = frame_shape[:2]
        ranked = []
        for box, confidence in candidates:
            x1, y1, x2, y2 = [float(value) for value in box]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if (
                width < 1.0
                or height < 1.0
                or width > frame_width * 0.045
                or height > frame_height * 0.065
            ):
                continue
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            near_player = False
            proximity_bonus = 0.0
            for athlete in athletes:
                px1, py1, px2, py2 = athlete.bbox_xyxy
                player_height = max(1.0, py2 - py1)
                player_point = ((px1 + px2) / 2.0, py2)
                distance = math.hypot(
                    center[0] - player_point[0], center[1] - player_point[1]
                )
                limit = max(45.0, player_height * 2.2)
                if distance <= limit:
                    near_player = True
                    proximity_bonus = max(proximity_bonus, 0.45 * (1.0 - distance / limit))

            temporal_match = False
            temporal_bonus = 0.0
            if (
                self.previous_ball_center is not None
                and self.previous_ball_timestamp_ms is not None
            ):
                elapsed_ms = max(1, timestamp_ms - self.previous_ball_timestamp_ms)
                distance = math.hypot(
                    center[0] - self.previous_ball_center[0],
                    center[1] - self.previous_ball_center[1],
                )
                limit = min(260.0, max(55.0, elapsed_ms * 0.65))
                if elapsed_ms <= 1_500 and distance <= limit:
                    temporal_match = True
                    temporal_bonus = 0.50 * (1.0 - distance / limit)

            if near_player or temporal_match:
                ranked.append(
                    (float(confidence) + proximity_bonus + temporal_bonus, box, confidence, center)
                )

        if not ranked:
            return None
        _, box, confidence, center = max(ranked, key=lambda item: item[0])
        self.previous_ball_center = center
        self.previous_ball_timestamp_ms = timestamp_ms
        return box, confidence

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
        box_height = max(1.0, y2 - y1)
        box_width = max(1.0, x2 - x1)
        left = max(0, min(width - 1, int(x1 + box_width * 0.18)))
        right = max(left + 1, min(width, int(x2 - box_width * 0.18)))
        top = max(0, min(height - 1, int(y1 + box_height * 0.08)))
        bottom = max(top + 1, min(height, int(y1 + box_height * 0.58)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        green = (
            (hsv[:, 0] >= 25)
            & (hsv[:, 0] <= 100)
            & (hsv[:, 1] >= 20)
        )
        usable = ~green & (hsv[:, 2] >= 18)
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
