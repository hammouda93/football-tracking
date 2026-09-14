from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.jersey import JerseyNumberRecognizer, JerseyObservation, WeightedNumberVote
from pipeline.types import ObjectRole, TrackedObject


ATHLETE_ROLES = {str(ObjectRole.PLAYER), str(ObjectRole.GOALKEEPER)}
PERSON_ROLES = {*ATHLETE_ROLES, str(ObjectRole.REFEREE)}


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _cosine(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    first = _normalize(first)
    second = _normalize(second)
    if first.size != second.size or not first.size:
        return 0.0
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bottom_center(box: Iterable[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = [float(value) for value in box]
    return (x1 + x2) / 2.0, y2


def _box_height(box: Iterable[float]) -> float:
    _x1, y1, _x2, y2 = [float(value) for value in box]
    return max(1.0, y2 - y1)


@dataclass(slots=True)
class IdentityState:
    canonical_id: int
    role: str
    first_timestamp_ms: int
    last_timestamp_ms: int
    last_bbox: tuple[float, float, float, float]
    last_pitch: tuple[float, float] | None = None
    appearance: np.ndarray | None = None
    team_feature_sum: np.ndarray | None = None
    team_feature_count: int = 0
    team_votes: Counter = field(default_factory=Counter)
    team_label: str | None = None
    role_votes: Counter = field(default_factory=Counter)
    jersey_vote: WeightedNumberVote = field(default_factory=WeightedNumberVote)
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    roster_player_id: int | None = None
    roster_player_name: str = ""
    roster_resolved_number: int | None = None
    roster_resolved_team: str | None = None
    raw_track_ids: set[str] = field(default_factory=set)
    windows_seen: set[int] = field(default_factory=set)
    observations: int = 0

    @property
    def mean_team_feature(self) -> np.ndarray | None:
        if self.team_feature_sum is None or self.team_feature_count <= 0:
            return None
        return self.team_feature_sum / float(self.team_feature_count)


class NativeAppearanceEncoder:
    """Native Windows appearance encoder with an optional deep Re-ID model.

    A supplied Ultralytics classification/Re-ID checkpoint is used in batches. If it
    is absent or incompatible, deterministic colour/texture descriptors keep the
    pipeline operational and the diagnostics explicitly report the fallback.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        *,
        backend: str = "auto",
        model_name: str = "osnet_x0_25",
    ):
        self.model_path = str(model_path or "").strip()
        self.device = device
        self.requested_backend = str(backend or "auto").strip().lower()
        self.model_name = str(model_name or "osnet_x0_25").strip()
        self.model = None
        self.model_kind = ""
        self.backend = "histogram"
        self.error = ""
        if not self.model_path:
            return
        if not Path(self.model_path).is_file():
            self.error = f"Poids Re-ID absents ou invalides: {self.model_path}"
            return
        try:
            suffixes = "".join(Path(self.model_path).suffixes).lower()
            if suffixes.endswith(".onnx"):
                import cv2

                self.model = cv2.dnn.readNetFromONNX(self.model_path)
                self.model_kind = "onnx"
                self.backend = "onnx_embeddings"
                return
            if self.requested_backend in {"osnet", "torchreid"} or ".pth" in suffixes:
                from pipeline.providers.osnet import OSNetFeatureExtractor

                self.model = OSNetFeatureExtractor(
                    model_path=self.model_path,
                    device=self.device,
                )
                self.model_kind = "osnet"
                self.backend = f"native_{self.model_name}"
                return
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.model_kind = "ultralytics"
            self.backend = "ultralytics_embeddings"
        except Exception as exc:  # pragma: no cover - depends on optional model
            self.error = f"Re-ID indisponible: {exc}"
            self.model = None
            self.model_kind = ""
            self.backend = "histogram"

    @staticmethod
    def _crop(frame, box, *, torso: bool = False):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box]
        box_height = max(1.0, y2 - y1)
        if torso:
            box_width = max(1.0, x2 - x1)
            x1 += box_width * 0.18
            x2 -= box_width * 0.18
            y1 += box_height * 0.12
            y2 = y1 + box_height * 0.46
        else:
            # Remove most of the head/background. Identity comes from kit,
            # silhouette and texture; no explicit skin-colour feature is stored.
            y1 += box_height * 0.06
        left = max(0, min(width - 1, int(round(x1))))
        right = max(left + 1, min(width, int(round(x2))))
        top = max(0, min(height - 1, int(round(y1))))
        bottom = max(top + 1, min(height, int(round(y2))))
        return frame[top:bottom, left:right]

    @staticmethod
    def _histogram_feature(crop) -> np.ndarray:
        import cv2

        if crop.size == 0:
            return np.zeros(190, dtype=np.float32)
        resized = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        hs_hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        ab_hist = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gradients_x, gradients_y, angleInDegrees=True)
        orientation, _ = np.histogram(
            angle,
            bins=18,
            range=(0, 360),
            weights=magnitude,
        )
        spatial = []
        for row in range(2):
            for column in range(2):
                cell = lab[row * 48 : (row + 1) * 48, column * 24 : (column + 1) * 24]
                spatial.extend(np.mean(cell, axis=(0, 1)) / 255.0)
        return _normalize(
            np.concatenate(
                [
                    cv2.normalize(hs_hist, None).reshape(-1),
                    cv2.normalize(ab_hist, None).reshape(-1),
                    _normalize(orientation.astype(np.float32)),
                    np.asarray(spatial, dtype=np.float32),
                ]
            )
        )

    @classmethod
    def team_feature(cls, frame, box) -> np.ndarray | None:
        import cv2

        crop = cls._crop(frame, box, torso=True)
        if crop.size < 24:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        feature = np.asarray(
            [
                np.median(hue) / 179.0,
                np.median(saturation) / 255.0,
                np.median(value) / 255.0,
                np.median(lab[:, :, 1]) / 255.0,
                np.median(lab[:, :, 2]) / 255.0,
                np.mean(value >= 205),
                np.mean(value <= 65),
                np.mean((hue <= 12) | (hue >= 168)),
            ],
            dtype=np.float32,
        )
        return feature

    def encode(self, frame, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        crops = [self._crop(frame, box) for box in boxes]
        if self.model is not None and crops:
            try:  # pragma: no cover - exercised only with an optional checkpoint
                if self.model_kind == "onnx":
                    import cv2

                    tensors = []
                    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                    for crop in crops:
                        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        tensors.append(((rgb - mean) / std).transpose(2, 0, 1))
                    self.model.setInput(np.asarray(tensors, dtype=np.float32))
                    outputs = np.asarray(self.model.forward())
                    outputs = [outputs[index].reshape(-1) for index in range(len(crops))]
                elif self.model_kind == "osnet":
                    outputs = self.model(crops)
                else:
                    outputs = self.model.embed(
                        source=crops,
                        device=self.device,
                        verbose=False,
                    )
                features = []
                for output in outputs:
                    if hasattr(output, "detach"):
                        output = output.detach().cpu().numpy()
                    features.append(_normalize(np.asarray(output)))
                if len(features) == len(crops):
                    return features
                raise ValueError("nombre d'embeddings différent du nombre de joueurs")
            except Exception as exc:
                self.error = f"Re-ID profond désactivé pendant le test: {exc}"
                self.model = None
                self.model_kind = ""
                self.backend = "histogram"
        return [self._histogram_feature(crop) for crop in crops]


class NativeIdentityRefiner:
    """Second-pass identity and team stabilizer inspired by GSR/IDATR.

    The detector/tracker remains replaceable. This layer removes duplicate person
    boxes, aggregates appearance per tracklet, reconnects conservative fragments,
    and learns the two outfield jersey groups from the video only.
    """

    def __init__(
        self,
        *,
        home_team_cluster: str = "B",
        max_gap_seconds: float = 3.0,
        global_max_gap_seconds: float = 7_200.0,
        reid_model_path: str = "",
        reid_backend: str = "auto",
        reid_model_name: str = "osnet_x0_25",
        jersey_engine: str = "auto",
        jersey_model_path: str = "",
        jersey_device: str = "cpu",
        jersey_interval_frames: int = 12,
        jersey_minimum_box_height: int = 72,
        jersey_max_crops_per_frame: int = 4,
        roster: list[dict] | None = None,
        device: str = "cpu",
    ):
        self.home_team_cluster = (
            "A" if str(home_team_cluster).strip().upper() == "A" else "B"
        )
        self.max_gap_ms = max(500, int(float(max_gap_seconds) * 1_000))
        self.global_max_gap_ms = max(
            self.max_gap_ms,
            int(float(global_max_gap_seconds) * 1_000),
        )
        self.encoder = NativeAppearanceEncoder(
            reid_model_path,
            device,
            backend=reid_backend,
            model_name=reid_model_name,
        )
        self.jersey = JerseyNumberRecognizer(
            engine=jersey_engine,
            model_path=jersey_model_path,
            device=jersey_device,
            minimum_box_height=jersey_minimum_box_height,
        )
        self.jersey_interval_frames = max(1, int(jersey_interval_frames))
        self.jersey_max_crops_per_frame = max(
            1, int(jersey_max_crops_per_frame)
        )
        self.jersey_cursor = 0
        self.roster = self._normalize_roster(roster or [])
        self.roster_by_team_number: dict[tuple[str, int], dict] = {}
        self.roster_by_number: defaultdict[int, list[dict]] = defaultdict(list)
        for entry in self.roster:
            key = (entry["team_key"], entry["shirt_number"])
            if key not in self.roster_by_team_number:
                self.roster_by_team_number[key] = entry
            else:
                # A duplicated shirt number inside one team must be resolved by a human.
                self.roster_by_team_number[key] = {}
            self.roster_by_number[entry["shirt_number"]].append(entry)
        self.reset()

    @staticmethod
    def _normalize_roster(rows: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for row in rows:
            try:
                player_id = int(row.get("id"))
                shirt_number = int(row.get("shirt_number"))
            except (TypeError, ValueError):
                continue
            team_key = str(row.get("team_key") or "").strip().lower()
            if team_key not in {"home", "away"} or not 0 <= shirt_number <= 99:
                continue
            normalized.append(
                {
                    "id": player_id,
                    "team_key": team_key,
                    "shirt_number": shirt_number,
                    "name": str(row.get("name") or player_id),
                    "position": str(row.get("position") or "").strip().upper(),
                }
            )
        return normalized

    def reset(self) -> None:
        self.states: dict[int, IdentityState] = {}
        self.raw_to_canonical: dict[str, int] = {}
        self.next_id = 1
        self.frames = 0
        self.raw_tracks_seen: set[str] = set()
        self.duplicates_removed = 0
        self.fragments_stitched = 0
        self.team_observations = 0
        self.team_centers: np.ndarray | None = None
        self.team_last_fit_observations = 0
        self.team_fits = 0
        self.window_index = 0
        self.global_reacquisitions = 0
        self.roster_resolutions = 0

    def reset_window(self) -> None:
        """Reset only the online tracker namespace, preserving match identities."""

        self.raw_to_canonical = {}
        self.window_index += 1

    @staticmethod
    def _same_role_family(first: str, second: str) -> bool:
        if first == second:
            return True
        return first in ATHLETE_ROLES and second in ATHLETE_ROLES

    @classmethod
    def _duplicate(cls, first, second, feature_a, feature_b) -> bool:
        if not cls._same_role_family(first.role, second.role):
            return False
        if _bbox_iou(first.bbox_xyxy, second.bbox_xyxy) < 0.15:
            return False
        ax, ay = _bottom_center(first.bbox_xyxy)
        bx, by = _bottom_center(second.bbox_xyxy)
        average_height = (_box_height(first.bbox_xyxy) + _box_height(second.bbox_xyxy)) / 2.0
        average_width = (
            max(1.0, first.bbox_xyxy[2] - first.bbox_xyxy[0])
            + max(1.0, second.bbox_xyxy[2] - second.bbox_xyxy[0])
        ) / 2.0
        same_feet = abs(ay - by) <= max(3.0, 0.035 * average_height)
        same_column = abs(ax - bx) <= max(4.0, 0.40 * average_width)
        return same_feet and same_column and _cosine(feature_a, feature_b) >= 0.78

    def _deduplicate(self, people, features):
        kept: list[int] = []
        for index in sorted(range(len(people)), key=lambda i: people[i].confidence, reverse=True):
            if any(
                self._duplicate(people[index], people[other], features[index], features[other])
                for other in kept
            ):
                self.duplicates_removed += 1
                continue
            kept.append(index)
        return [people[index] for index in kept], [features[index] for index in kept]

    @staticmethod
    def _motion_similarity(state: IdentityState, obj: TrackedObject, gap_ms: int) -> float:
        previous = _bottom_center(state.last_bbox)
        current = _bottom_center(obj.bbox_xyxy)
        distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        scale = max(1.0, (_box_height(state.last_bbox) + _box_height(obj.bbox_xyxy)) / 2.0)
        normalized_distance = distance / scale
        maximum = 0.85 + 0.80 * max(0.0, gap_ms / 1_000.0)
        return max(0.0, 1.0 - normalized_distance / maximum)

    @staticmethod
    def _pitch_similarity(state: IdentityState, obj: TrackedObject, gap_ms: int) -> float | None:
        if state.last_pitch is None or obj.pitch_x is None or obj.pitch_y is None:
            return None
        distance = math.hypot(
            float(obj.pitch_x) - state.last_pitch[0],
            float(obj.pitch_y) - state.last_pitch[1],
        )
        seconds = max(0.04, gap_ms / 1_000.0)
        plausible_distance = 4.0 + 11.5 * seconds
        if distance > plausible_distance:
            return 0.0
        return max(0.0, 1.0 - distance / plausible_distance)

    def _team_from_feature(self, feature: np.ndarray | None) -> tuple[str | None, float]:
        if feature is None or self.team_centers is None:
            return None, 0.0
        distances = np.linalg.norm(self.team_centers - feature, axis=1)
        ranked = np.argsort(distances)
        margin = float(distances[ranked[1]] - distances[ranked[0]])
        confidence = max(0.0, min(1.0, margin / 0.35))
        home_index = 0 if self.home_team_cluster == "A" else 1
        observed = "home" if int(ranked[0]) == home_index else "away"
        return observed, confidence

    def _link_candidates(
        self,
        objects,
        features,
        team_features,
        jersey_observations,
        timestamp_ms: int,
    ):
        assignments: dict[int, tuple[int, float]] = {}
        used_states: set[int] = set()
        candidates: list[tuple[float, int, int, bool]] = []
        for index, (obj, feature, team_feature, jersey_observation) in enumerate(
            zip(objects, features, team_features, jersey_observations)
        ):
            raw_id = str(obj.track_id)
            known_id = self.raw_to_canonical.get(raw_id)
            if known_id in self.states:
                assignments[index] = (known_id, 1.0)
                used_states.add(known_id)
                continue
            for canonical_id, state in self.states.items():
                gap_ms = timestamp_ms - state.last_timestamp_ms
                if gap_ms <= 0 or gap_ms > self.global_max_gap_ms:
                    continue
                if not self._same_role_family(state.role, obj.role):
                    continue
                appearance = _cosine(state.appearance, feature)
                observed_team, observed_team_confidence = self._team_from_feature(team_feature)
                if (
                    state.team_label
                    and observed_team
                    and observed_team_confidence >= 0.35
                    and state.team_label != observed_team
                ):
                    continue
                observed_number = jersey_observation.number if jersey_observation else None
                if (
                    state.jersey_number is not None
                    and observed_number is not None
                    and jersey_observation.confidence >= 0.55
                    and state.jersey_number != observed_number
                ):
                    continue
                short_gap = gap_ms <= self.max_gap_ms
                if short_gap:
                    minimum_appearance = 0.72 if self.encoder.backend != "histogram" else 0.90
                    if appearance < minimum_appearance:
                        continue
                    motion = self._motion_similarity(state, obj, gap_ms)
                    if motion <= 0:
                        continue
                    pitch = self._pitch_similarity(state, obj, gap_ms)
                    if pitch == 0.0:
                        continue
                    if pitch is None:
                        score = 0.70 * appearance + 0.26 * motion
                        score += 0.04 * observed_team_confidence
                    else:
                        score = 0.58 * appearance + 0.20 * motion
                        score += 0.14 * pitch + 0.08 * observed_team_confidence
                    minimum_score = 0.72 if self.encoder.backend != "histogram" else 0.88
                else:
                    # Long reacquisition is forbidden with the colour histogram
                    # fallback. It requires deep appearance plus another cue.
                    if self.encoder.backend == "histogram" or appearance < 0.82:
                        continue
                    secondary_evidence = 0.0
                    if observed_team and state.team_label == observed_team:
                        secondary_evidence = max(secondary_evidence, observed_team_confidence)
                    if observed_number is not None and observed_number == state.jersey_number:
                        secondary_evidence = max(secondary_evidence, jersey_observation.confidence)
                    if secondary_evidence < 0.28:
                        continue
                    score = 0.82 * appearance + 0.18 * secondary_evidence
                    minimum_score = 0.82
                if score >= minimum_score:
                    candidates.append((score, index, canonical_id, not short_gap))

        for score, index, canonical_id, long_gap in sorted(candidates, reverse=True):
            if index in assignments or canonical_id in used_states:
                continue
            same_object_scores = [
                value
                for value, other_index, other_id, _long_gap in candidates
                if other_index == index and other_id != canonical_id
            ]
            if same_object_scores and score - max(same_object_scores) < 0.035:
                continue
            assignments[index] = (canonical_id, score)
            used_states.add(canonical_id)
            self.fragments_stitched += 1
            if long_gap:
                self.global_reacquisitions += 1
        return assignments

    def _new_state(self, obj, feature, timestamp_ms: int) -> IdentityState:
        canonical_id = self.next_id
        self.next_id += 1
        state = IdentityState(
            canonical_id=canonical_id,
            role=obj.role,
            first_timestamp_ms=timestamp_ms,
            last_timestamp_ms=timestamp_ms,
            last_bbox=tuple(obj.bbox_xyxy),
            appearance=feature.copy(),
        )
        self.states[canonical_id] = state
        return state

    def _update_state(
        self,
        state,
        obj,
        feature,
        team_feature,
        jersey_observation: JerseyObservation | None,
        timestamp_ms: int,
    ) -> None:
        state.role_votes[str(obj.role)] += max(0.05, float(obj.confidence))
        state.role = state.role_votes.most_common(1)[0][0]
        state.last_timestamp_ms = timestamp_ms
        state.last_bbox = tuple(obj.bbox_xyxy)
        if obj.pitch_x is not None and obj.pitch_y is not None:
            state.last_pitch = (float(obj.pitch_x), float(obj.pitch_y))
        state.appearance = (
            feature.copy()
            if state.appearance is None or state.appearance.size != feature.size
            else _normalize(0.88 * state.appearance + 0.12 * feature)
        )
        state.raw_track_ids.add(str(obj.track_id))
        state.windows_seen.add(self.window_index)
        state.observations += 1
        if jersey_observation is not None and obj.role in ATHLETE_ROLES:
            state.jersey_vote.add(
                jersey_observation.number,
                jersey_observation.confidence,
                timestamp_ms,
            )
            state.jersey_number, state.jersey_confidence = state.jersey_vote.result()
        if obj.role == str(ObjectRole.PLAYER) and team_feature is not None:
            if state.team_feature_sum is None:
                state.team_feature_sum = team_feature.astype(np.float32).copy()
            else:
                state.team_feature_sum += team_feature
            state.team_feature_count += 1
            self.team_observations += 1

    def _fit_team_clusters(self) -> None:
        tracklet_features = [
            state.mean_team_feature
            for state in self.states.values()
            if state.mean_team_feature is not None and state.team_feature_count >= 3
        ]
        if len(tracklet_features) < 6 or self.team_observations < 24:
            return
        if (
            self.team_centers is not None
            and self.team_observations - self.team_last_fit_observations < 40
        ):
            return
        import cv2

        samples = np.asarray(tracklet_features, dtype=np.float32)
        cv2.setRNGSeed(42)
        _compactness, labels, centers = cv2.kmeans(
            samples,
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001),
            12,
            cv2.KMEANS_PP_CENTERS,
        )
        populations = np.bincount(labels.reshape(-1), minlength=2)
        if int(populations.min()) < 2:
            return
        if self.team_centers is None:
            order = sorted(
                range(2),
                key=lambda index: (float(centers[index][1]), float(centers[index][0])),
            )
            centers = centers[order]
        else:
            direct = float(np.linalg.norm(centers[0] - self.team_centers[0])) + float(
                np.linalg.norm(centers[1] - self.team_centers[1])
            )
            swapped = float(np.linalg.norm(centers[1] - self.team_centers[0])) + float(
                np.linalg.norm(centers[0] - self.team_centers[1])
            )
            if swapped < direct:
                centers = centers[::-1]
        if float(np.linalg.norm(centers[0] - centers[1])) < 0.10:
            return
        self.team_centers = centers.astype(np.float32)
        self.team_last_fit_observations = self.team_observations
        self.team_fits += 1

    def _team_for_state(self, state: IdentityState) -> tuple[str | None, float]:
        feature = state.mean_team_feature
        if feature is None or self.team_centers is None:
            return state.team_label, 0.0
        distances = np.linalg.norm(self.team_centers - feature, axis=1)
        ranked = np.argsort(distances)
        margin = float(distances[ranked[1]] - distances[ranked[0]])
        confidence = max(0.0, min(1.0, margin / 0.35))
        if confidence < 0.15:
            return state.team_label, confidence
        cluster = int(ranked[0])
        home_index = 0 if self.home_team_cluster == "A" else 1
        observed = "home" if cluster == home_index else "away"
        state.team_votes[observed] += max(0.1, confidence)
        total = float(sum(state.team_votes.values()))
        winner, votes = state.team_votes.most_common(1)[0]
        if total >= 2.5 and votes / max(total, 1e-9) >= 0.72:
            if state.team_label is None:
                state.team_label = winner
            elif winner == state.team_label or votes >= 2.0 * state.team_votes[state.team_label]:
                state.team_label = winner
        return state.team_label, confidence

    def _resolve_roster(self, state: IdentityState) -> None:
        if state.jersey_number is None:
            return
        if (
            state.roster_player_id is not None
            and state.roster_resolved_number == state.jersey_number
            and (
                state.team_label is None
                or state.roster_resolved_team == state.team_label
            )
        ):
            return
        state.roster_player_id = None
        state.roster_player_name = ""
        state.roster_resolved_number = None
        state.roster_resolved_team = None
        entry = None
        if state.team_label:
            entry = self.roster_by_team_number.get((state.team_label, state.jersey_number))
        elif len(self.roster_by_number[state.jersey_number]) == 1:
            entry = self.roster_by_number[state.jersey_number][0]
        if not entry:
            return
        state.roster_player_id = int(entry["id"])
        state.roster_player_name = str(entry["name"])
        state.roster_resolved_number = int(entry["shirt_number"])
        state.roster_resolved_team = str(entry["team_key"])
        if state.team_label is None:
            state.team_label = str(entry["team_key"])
        self.roster_resolutions += 1

    def process(self, frame, objects: list[TrackedObject], timestamp_ms: int) -> list[TrackedObject]:
        self.frames += 1
        people = [obj for obj in objects if obj.role in PERSON_ROLES]
        passthrough = [obj for obj in objects if obj.role not in PERSON_ROLES]
        if not people:
            return objects
        boxes = [tuple(obj.bbox_xyxy) for obj in people]
        features = self.encoder.encode(frame, boxes)
        people, features = self._deduplicate(people, features)
        team_features = [
            self.encoder.team_feature(frame, obj.bbox_xyxy)
            if obj.role == str(ObjectRole.PLAYER)
            else None
            for obj in people
        ]
        read_jerseys = self.jersey.model is not None and self.frames % self.jersey_interval_frames == 0
        jersey_observations: list[JerseyObservation | None] = [None] * len(people)
        if read_jerseys:
            eligible = [
                index
                for index, obj in enumerate(people)
                if obj.role in ATHLETE_ROLES
                and _box_height(obj.bbox_xyxy)
                >= int(getattr(self.jersey, "minimum_box_height", 24))
                and (
                    self.raw_to_canonical.get(str(obj.track_id)) not in self.states
                    or self.states[self.raw_to_canonical[str(obj.track_id)]].jersey_number
                    is None
                )
            ]
            # Rotate a left-to-right list so one close foreground player cannot
            # consume the OCR budget on every frame.
            eligible.sort(key=lambda index: _bottom_center(people[index].bbox_xyxy)[0])
            if eligible:
                start = self.jersey_cursor % len(eligible)
                ordered = eligible[start:] + eligible[:start]
                selected = ordered[: self.jersey_max_crops_per_frame]
                self.jersey_cursor = (start + len(selected)) % len(eligible)
                for index in selected:
                    jersey_observations[index] = self.jersey.recognize(
                        frame,
                        people[index].bbox_xyxy,
                    )
        assignments = self._link_candidates(
            people,
            features,
            team_features,
            jersey_observations,
            timestamp_ms,
        )
        state_by_index = {}
        for index, (obj, feature, team_feature, jersey_observation) in enumerate(
            zip(people, features, team_features, jersey_observations)
        ):
            raw_id = str(obj.track_id)
            self.raw_tracks_seen.add(raw_id)
            assignment = assignments.get(index)
            if assignment is None:
                state = self._new_state(obj, feature, timestamp_ms)
                reid_confidence = 0.0
            else:
                state = self.states[assignment[0]]
                reid_confidence = float(assignment[1])
            self.raw_to_canonical[raw_id] = state.canonical_id
            self._update_state(
                state,
                obj,
                feature,
                team_feature,
                jersey_observation,
                timestamp_ms,
            )
            state_by_index[index] = (state, reid_confidence)

        self._fit_team_clusters()
        for index, obj in enumerate(people):
            state, reid_confidence = state_by_index[index]
            team_label, team_confidence = self._team_for_state(state)
            self._resolve_roster(state)
            if obj.role in ATHLETE_ROLES:
                obj.team_key = team_label
            elif obj.role == str(ObjectRole.REFEREE):
                obj.team_key = None
            obj.track_id = f"native-{state.canonical_id}"
            obj.player_key = obj.track_id if obj.role in ATHLETE_ROLES else None
            obj.shirt_number = state.jersey_number
            obj.metadata.update(
                {
                    "identity_engine": "native_gsr",
                    "identity_scope": "match",
                    "source_track_ids": sorted(state.raw_track_ids),
                    "reid_backend": self.encoder.backend,
                    "reid_confidence": round(reid_confidence, 4),
                    "team_confidence": round(team_confidence, 4),
                    "jersey_backend": self.jersey.backend,
                    "jersey_confidence": round(state.jersey_confidence, 4),
                    "roster_player_id": state.roster_player_id,
                    "roster_player_name": state.roster_player_name,
                    "identity_windows": len(state.windows_seen),
                }
            )
        return [*people, *passthrough]

    def diagnostics(self) -> dict:
        distance = (
            float(np.linalg.norm(self.team_centers[0] - self.team_centers[1]))
            if self.team_centers is not None
            else 0.0
        )
        return {
            "method": "native_gsr_tracklet_aggregation",
            "source": "video_only",
            "status": "ready" if self.team_centers is not None else "collecting",
            "appearance_backend": self.encoder.backend,
            "appearance_error": self.encoder.error,
            "deep_reid_ready": self.encoder.model is not None,
            "frames": self.frames,
            "windows": self.window_index,
            "raw_tracks": len(self.raw_tracks_seen),
            "canonical_tracks": len(self.states),
            "fragments_stitched": self.fragments_stitched,
            "global_reacquisitions": self.global_reacquisitions,
            "duplicates_removed": self.duplicates_removed,
            "team_tracklets": sum(
                state.mean_team_feature is not None for state in self.states.values()
            ),
            "team_observations": self.team_observations,
            "team_fits": self.team_fits,
            "jersey": self.jersey.diagnostics(),
            "jersey_interval_frames": self.jersey_interval_frames,
            "jersey_max_crops_per_frame": self.jersey_max_crops_per_frame,
            "jersey_tracklets": sum(
                state.jersey_number is not None for state in self.states.values()
            ),
            "roster_entries": len(self.roster),
            "roster_resolutions": self.roster_resolutions,
            "mapping_margin": round(distance, 4),
            "home_group": self.home_team_cluster,
            "away_group": "B" if self.home_team_cluster == "A" else "A",
        }
