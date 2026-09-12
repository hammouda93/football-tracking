from __future__ import annotations

import inspect
import math
from collections import Counter
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
        profile: str = "main_py",
        confidence: float = 0.3,
        ball_confidence: float = 0.12,
        image_size: int = 1280,
        tracking_fps: float = 10.0,
        tracker_name: str = "bytetrack",
        tracker_low_confidence: float = 0.10,
        tracker_new_confidence: float = 0.25,
        tracker_match_threshold: float = 0.80,
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
        self.profile = str(profile or "main_py").strip().lower()
        if self.profile not in {"main_py", "advanced"}:
            raise ValueError(
                f"Profil YOLO inconnu : {self.profile}. "
                "Valeurs acceptées : main_py, advanced."
            )
        if self.profile == "main_py":
            confidence = 0.30
            ball_confidence = 0.12
            image_size = 640
            tracker_name = "bytetrack"
            tracker_low_confidence = 0.30
            tracker_new_confidence = 0.25
            tracker_match_threshold = 0.80
        self.confidence = confidence
        self.image_size = image_size
        self.tracking_fps = max(1.0, float(tracking_fps))
        self.tracker_frame_rate = (
            25 if self.profile == "main_py" else max(1, int(round(self.tracking_fps)))
        )
        self.tracker_name = str(tracker_name or "botsort").strip().lower()
        if self.tracker_name not in {"botsort", "bytetrack"}:
            raise ValueError(
                f"Tracker YOLO inconnu : {self.tracker_name}. "
                "Valeurs acceptées : botsort, bytetrack."
            )
        self.tracker_low_confidence = max(
            0.01, min(float(tracker_low_confidence), float(confidence))
        )
        self.ball_confidence = max(
            0.01,
            min(float(ball_confidence), float(confidence)),
        )
        self.tracker_new_confidence = max(
            self.tracker_low_confidence, float(tracker_new_confidence)
        )
        self.tracker_match_threshold = max(
            0.1, min(0.99, float(tracker_match_threshold))
        )
        self.tracker_buffer_seconds = max(1.0, float(tracker_buffer_seconds))
        self.tracker_buffer_frames = (
            30
            if self.profile == "main_py"
            else max(15, int(round(self.tracking_fps * self.tracker_buffer_seconds)))
        )
        self.class_roles: dict[int, ObjectRole] = {}
        self._register_class_ids(player_class_ids, ObjectRole.PLAYER)
        self._register_class_ids(goalkeeper_class_ids, ObjectRole.GOALKEEPER)
        self._register_class_ids(referee_class_ids, ObjectRole.REFEREE)
        self._register_class_ids(ball_class_ids, ObjectRole.BALL)
        self.team_colors = {
            key: self._hex_to_lab(value) for key, value in (team_colors or {}).items()
        }
        self.team_reference_features = {
            key: self._hex_to_team_feature(value)
            for key, value in (team_colors or {}).items()
        }
        self.team_color_samples: list[np.ndarray] = []
        self.team_cluster_centers: np.ndarray | None = None
        self.team_cluster_mapping: dict[int, str] = {}
        self.team_calibration_fits = 0
        self.team_calibration_last_fit = 0
        self.team_calibration_mapping_margin = 0.0
        self.previous_gray = None
        self.previous_ball_center: tuple[float, float] | None = None
        self.previous_ball_timestamp_ms: int | None = None
        self.team_votes: dict[int, Counter[str]] = {}
        self.tracker = self._build_tracker()
        self.official_tracker = self._build_tracker()

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
            track_buffer=getattr(
                self,
                "tracker_buffer_frames",
                max(15, int(round(self.tracking_fps * self.tracker_buffer_seconds))),
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
            lost_track_buffer=self.tracker_buffer_frames,
            minimum_matching_threshold=self.tracker_match_threshold,
            frame_rate=self.tracker_frame_rate,
        )

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_ball_center = None
        self.previous_ball_timestamp_ms = None
        self.team_votes = {}
        for attribute in ("tracker", "official_tracker"):
            tracker = getattr(self, attribute)
            if hasattr(tracker, "reset"):
                tracker.reset()
            else:
                setattr(self, attribute, self._build_tracker())

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        import cv2
        import supervision as sv

        height, width = frame.shape[:2]
        inference_confidence = min(
            self.tracker_low_confidence,
            self.ball_confidence,
        )
        prediction = self.model.predict(
            source=frame,
            # Player tracking keeps its own threshold below. A lower inference
            # threshold is necessary so tiny ball candidates are not discarded
            # by YOLO before the temporal ball selector can inspect them.
            conf=inference_confidence,
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
        raw_person_indices = [
            index for index, role in enumerate(roles) if role in trackable_roles
        ]
        raw_athlete_indices = [
            index
            for index, role in enumerate(roles)
            if role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
        ]
        raw_official_indices = [
            index for index, role in enumerate(roles) if role == ObjectRole.REFEREE
        ]
        tracker_person_indices = [
            index
            for index in raw_person_indices
            if confidences[index] >= self.tracker_low_confidence
        ]
        tracker_athlete_indices = [
            index
            for index in raw_athlete_indices
            if confidences[index] >= self.tracker_low_confidence
        ]
        tracker_official_indices = [
            index
            for index in raw_official_indices
            if confidences[index] >= self.tracker_low_confidence
        ]
        deduplicated_athlete_indices = self._deduplicate_indices(
            detections.xyxy,
            confidences,
            tracker_athlete_indices,
        )
        deduplicated_official_indices = self._deduplicate_indices(
            detections.xyxy,
            confidences,
            tracker_official_indices,
        )
        # main.py did not remove overlapping detections before ByteTrack. Keeping
        # that exact behaviour lets the short reference test isolate the pipeline
        # regression instead of guessing at more thresholds.
        trackable_athlete_indices = (
            tracker_athlete_indices
            if self.profile == "main_py"
            else deduplicated_athlete_indices
        )
        trackable_official_indices = (
            tracker_official_indices
            if self.profile == "main_py"
            else deduplicated_official_indices
        )
        tracked_rows = self._update_tracker(
            prediction,
            detections,
            trackable_athlete_indices,
            frame,
        )
        official_rows = self._update_tracker(
            prediction,
            detections,
            trackable_official_indices,
            frame,
            tracker=self.official_tracker,
        )
        tracked_rows.extend(
            (box, confidence, class_id, int(tracker_id) + 1_000_000)
            for box, confidence, class_id, tracker_id in official_rows
        )

        objects: list[TrackedObject] = []
        for xyxy, confidence, class_id, tracker_id in tracked_rows:
            role = self._role_for(int(class_id), names)
            # main.py sent the ball through ByteTrack but drew and consumed the
            # raw detector ball separately. Preserve that without duplicating it
            # in FrameAnalysis.objects.
            if role == ObjectRole.BALL:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            image_x = ((x1 + x2) / 2.0) / max(width, 1)
            image_y = y2 / max(height, 1)
            team_key = None
            if role == ObjectRole.PLAYER:
                observed_team = self._classify_team(frame, (x1, y1, x2, y2))
                team_key = self._stabilize_team(int(tracker_id), observed_team)
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
        if self.profile != "main_py":
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
                    or float(confidence) < self.ball_confidence
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
                    for index in raw_person_indices
                ),
                "raw_referee_detections": sum(
                    role == ObjectRole.REFEREE and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_ball_detections": sum(
                    role == ObjectRole.BALL and confidence >= self.ball_confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_other_detections": sum(
                    role == ObjectRole.OTHER and confidence >= self.confidence
                    for role, confidence in zip(roles, confidences)
                ),
                "raw_athlete_boxes": [
                    [float(value) for value in detections.xyxy[index]]
                    for index in raw_person_indices
                    if roles[index] in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
                    and confidences[index] >= self.confidence
                ],
                "raw_detections": [
                    {
                        "bbox": [float(value) for value in detections.xyxy[index]],
                        "role": str(roles[index]),
                        "confidence": float(confidences[index]),
                    }
                    for index in range(len(detections))
                    if roles[index]
                    in {
                        ObjectRole.PLAYER,
                        ObjectRole.GOALKEEPER,
                        ObjectRole.REFEREE,
                        ObjectRole.BALL,
                    }
                    and confidences[index]
                    >= (
                        self.ball_confidence
                        if roles[index] == ObjectRole.BALL
                        else self.confidence
                    )
                ],
                "rejected_person_boxes": [],
                "rejected_person_detections": 0,
                "duplicate_person_detections": max(
                    0,
                    len(tracker_person_indices)
                    - len(deduplicated_athlete_indices)
                    - len(deduplicated_official_indices),
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
                "profile": self.profile,
                "image_size": self.image_size,
                "detector_confidence": self.confidence,
                "ball_confidence": self.ball_confidence,
                "team_calibration": self.team_calibration_diagnostics(),
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

    @staticmethod
    def _box_smaller_coverage(first, second) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area > 0 else 0.0

    @classmethod
    def _boxes_are_duplicates(
        cls,
        first,
        second,
        *,
        iou_threshold: float,
        coverage_threshold: float = 0.92,
    ) -> bool:
        return (
            cls._box_iou(first, second) >= iou_threshold
            or cls._box_smaller_coverage(first, second) >= coverage_threshold
        )

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
            if any(
                cls._boxes_are_duplicates(
                    boxes[index],
                    boxes[other],
                    iou_threshold=threshold,
                )
                for other in kept
            ):
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
                cls._boxes_are_duplicates(
                    item.bbox_xyxy,
                    other.bbox_xyxy,
                    iou_threshold=threshold,
                )
                for other in kept
            ):
                continue
            kept.append(item)
        return kept

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
                or width > frame_width * 0.025
                or height > frame_height * 0.040
                or width / height < 0.45
                or width / height > 2.20
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
                limit = max(85.0, player_height * 3.5)
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
                limit = min(360.0, max(85.0, elapsed_ms * 1.10))
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

    def _stabilize_team(self, tracker_id: int, observed_team: str | None) -> str | None:
        votes = self.team_votes.setdefault(int(tracker_id), Counter())
        if observed_team in self.team_colors:
            votes[observed_team] += 1
        if not votes:
            return None
        ranked = votes.most_common(2)
        winner, winner_votes = ranked[0]
        runner_up_votes = ranked[1][1] if len(ranked) > 1 else 0
        total_votes = sum(votes.values())
        required_margin = max(1, int(round(total_votes * 0.12)))
        return winner if winner_votes - runner_up_votes >= required_margin else None

    def _update_tracker(self, prediction, detections, indices, frame, *, tracker=None):
        tracker = tracker or self.tracker
        if self.tracker_name == "botsort":
            boxes = prediction.boxes[indices] if indices else prediction.boxes[:0]
            boxes = boxes.cpu().numpy()
            tracks = tracker.update(boxes, frame)
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
        tracked = tracker.update_with_detections(athletes)
        return [
            (box, confidence, class_id, tracker_id)
            for box, confidence, class_id, tracker_id in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.class_id,
                tracked.tracker_id,
            )
            if tracker_id is not None
        ]

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
        """Classify a jersey against two automatically learned match clusters.

        The match colors are used only once to name the learned clusters home and
        away. Per-frame classification compares the torso with those learned
        broadcast colors, which is much more robust to shadows and compression
        than comparing every crop directly with an exported CSS color.
        """

        feature = self._extract_team_feature(frame, box)
        if feature is None:
            return None
        self._observe_team_feature(feature)
        if self.team_cluster_centers is None or not self.team_cluster_mapping:
            return None
        distances = np.linalg.norm(self.team_cluster_centers - feature, axis=1)
        ranked = np.argsort(distances)
        if len(ranked) < 2 or float(distances[ranked[1]] - distances[ranked[0]]) < 3.0:
            return None
        return self.team_cluster_mapping.get(int(ranked[0]))

    def _extract_team_feature(
        self,
        frame,
        box: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        box_height = max(1.0, y2 - y1)
        box_width = max(1.0, x2 - x1)
        left = max(0, min(width - 1, int(x1 + box_width * 0.25)))
        right = max(left + 1, min(width, int(x2 - box_width * 0.25)))
        top = max(0, min(height - 1, int(y1 + box_height * 0.16)))
        bottom = max(top + 1, min(height, int(y1 + box_height * 0.55)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        green = (
            (hsv[:, 0] >= 25)
            & (hsv[:, 0] <= 100)
            & (hsv[:, 1] >= 20)
        )
        usable = (~green) & (hsv[:, 2] >= 18)
        pixels = hsv[usable]
        if len(pixels) < 8:
            pixels = hsv[hsv[:, 2] >= 18]
        if len(pixels) < 8:
            return None

        hues = pixels[:, 0].astype(np.float64)
        saturations = pixels[:, 1].astype(np.float64)
        values = pixels[:, 2].astype(np.float64)
        angles = hues * (2.0 * math.pi / 180.0)
        # Low-saturation white/gray pixels have an unstable hue, so they carry
        # less angular weight than a saturated red/black jersey.
        hue_weights = np.maximum(0.15, saturations / 255.0)
        mean_sin = float(np.average(np.sin(angles), weights=hue_weights))
        mean_cos = float(np.average(np.cos(angles), weights=hue_weights))
        hue = (math.atan2(mean_sin, mean_cos) * 180.0 / (2.0 * math.pi)) % 180.0
        return self._team_feature(
            hue,
            float(np.median(saturations)),
            float(np.median(values)),
        )

    def _observe_team_feature(self, feature: np.ndarray) -> None:
        self.team_color_samples.append(feature.astype(np.float32))
        if len(self.team_color_samples) > 400:
            self.team_color_samples.pop(0)
        sample_count = len(self.team_color_samples)
        if sample_count < 12:
            return
        if (
            self.team_cluster_centers is not None
            and sample_count - self.team_calibration_last_fit < 40
        ):
            return

        import cv2

        samples = np.asarray(self.team_color_samples, dtype=np.float32)
        cv2.setRNGSeed(42)
        _compactness, labels, centers = cv2.kmeans(
            samples,
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.15),
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        populations = np.bincount(labels.reshape(-1), minlength=2)
        if int(populations.min()) < max(2, int(round(sample_count * 0.12))):
            return
        if float(np.linalg.norm(centers[0] - centers[1])) < 12.0:
            return

        mapping, margin = self._map_team_clusters(centers)
        if len(mapping) != 2:
            return
        self.team_cluster_centers = centers.astype(np.float32)
        self.team_cluster_mapping = mapping
        self.team_calibration_mapping_margin = margin
        self.team_calibration_last_fit = sample_count
        self.team_calibration_fits += 1

    def _map_team_clusters(self, centers: np.ndarray) -> tuple[dict[int, str], float]:
        if not {"home", "away"}.issubset(self.team_reference_features):
            return {}, 0.0
        home = self.team_reference_features["home"]
        away = self.team_reference_features["away"]
        direct = float(np.linalg.norm(centers[0] - home)) + float(
            np.linalg.norm(centers[1] - away)
        )
        swapped = float(np.linalg.norm(centers[0] - away)) + float(
            np.linalg.norm(centers[1] - home)
        )
        if direct <= swapped:
            return {0: "home", 1: "away"}, swapped - direct
        return {0: "away", 1: "home"}, direct - swapped

    def team_calibration_diagnostics(self) -> dict:
        return {
            "method": "automatic_jersey_clusters",
            "status": "ready" if self.team_cluster_centers is not None else "collecting",
            "samples": len(self.team_color_samples),
            "fits": self.team_calibration_fits,
            "mapping": dict(self.team_cluster_mapping),
            "mapping_margin": round(self.team_calibration_mapping_margin, 2),
        }

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

    @staticmethod
    def _team_feature(hue: float, saturation: float, value: float) -> np.ndarray:
        angle = float(hue) * (2.0 * math.pi / 180.0)
        hue_weight = 10.0 + 35.0 * max(0.0, min(1.0, float(saturation) / 255.0))
        return np.asarray(
            [
                math.cos(angle) * hue_weight,
                math.sin(angle) * hue_weight,
                float(saturation) / 4.0,
                float(value) / 8.0,
            ],
            dtype=np.float32,
        )

    @classmethod
    def _hex_to_team_feature(cls, value: str) -> np.ndarray:
        import cv2

        value = value.lstrip("#")
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
        pixel = np.uint8([[[blue, green, red]]])
        hue, saturation, brightness = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        return cls._team_feature(float(hue), float(saturation), float(brightness))
