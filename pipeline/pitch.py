from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class PitchLandmark:
    index: int
    name: str
    pitch_x: float
    pitch_y: float


@dataclass(frozen=True, slots=True)
class PitchCalibrationResult:
    homography: np.ndarray
    visible_landmarks: int
    inliers: int
    inlier_ratio: float
    reprojection_error_m: float


def load_pitch_landmarks(path: str, *, expected_count: int = 97) -> list[PitchLandmark]:
    schema_path = Path(str(path or ""))
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schéma des points terrain absent: {schema_path}")
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    rows = payload.get("landmarks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Le schéma terrain doit contenir une liste 'landmarks'.")
    landmarks: list[PitchLandmark] = []
    seen: set[int] = set()
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Point terrain #{offset} invalide.")
        index = int(row.get("index", offset))
        coordinates = row.get("pitch_xy")
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError(f"Point terrain #{index}: pitch_xy=[x,y] obligatoire.")
        if index in seen:
            raise ValueError(f"Indice terrain dupliqué: {index}")
        x, y = float(coordinates[0]), float(coordinates[1])
        if not all(math.isfinite(value) for value in (x, y)):
            raise ValueError(f"Point terrain #{index}: coordonnées non finies.")
        landmarks.append(PitchLandmark(index, str(row.get("name", index)), x, y))
        seen.add(index)
    if expected_count and len(landmarks) != int(expected_count):
        raise ValueError(
            f"Le checkpoint annonce {expected_count} points; le schéma en contient {len(landmarks)}."
        )
    return sorted(landmarks, key=lambda item: item.index)


class NativePitchCalibrator:
    """ONNX pitch-keypoint adapter with strict, auditable homography checks.

    It accepts either K heatmaps or K rows ``x, y, confidence``. Semantic pitch
    coordinates always come from the checkpoint's own label schema; they are
    never guessed from generic Hough lines.
    """

    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

    def __init__(
        self,
        *,
        model_path: str = "",
        schema_path: str = "",
        confidence: float = 0.20,
        interval_frames: int = 10,
        hold_frames: int = 20,
        expected_landmarks: int = 97,
        input_width: int = 960,
        input_height: int = 540,
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.schema_path = str(schema_path or "").strip()
        self.confidence = max(0.01, min(0.99, float(confidence)))
        self.interval_frames = max(1, int(interval_frames))
        self.hold_frames = max(0, int(hold_frames))
        self.expected_landmarks = max(4, int(expected_landmarks))
        self.input_width = max(64, int(input_width))
        self.input_height = max(64, int(input_height))
        self.input_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.output_name = ""
        self.net = None
        self.landmarks: list[PitchLandmark] = []
        self.backend = "disabled"
        self.error = ""
        self.frame_index = 0
        self.last_success_frame = -10_000
        self.homography: np.ndarray | None = None
        self.last_result: PitchCalibrationResult | None = None
        self.attempts = 0
        self.successes = 0
        self.projected_frames = 0
        self.scene_resets = 0
        self.maximum_visible_landmarks = 0
        self._initialize()

    def _initialize(self) -> None:
        if not self.model_path and not self.schema_path:
            return
        if not self.model_path or not self.schema_path:
            self.error = "Le modèle terrain et son schéma 97 points sont tous les deux obligatoires."
            return
        model = Path(self.model_path)
        if not model.is_file():
            self.error = f"Poids terrain absents: {model}"
            return
        if model.suffix.lower() != ".onnx":
            self.error = "Le calibrateur terrain natif attend un modèle .onnx."
            return
        try:
            import cv2

            schema_payload = json.loads(Path(self.schema_path).read_text(encoding="utf-8"))
            model_spec = (
                schema_payload.get("model", {})
                if isinstance(schema_payload, dict)
                else {}
            )
            if isinstance(model_spec, dict):
                self.input_width = max(
                    64, int(model_spec.get("input_width", self.input_width))
                )
                self.input_height = max(
                    64, int(model_spec.get("input_height", self.input_height))
                )
                mean = model_spec.get("mean", self.input_mean.tolist())
                std = model_spec.get("std", self.input_std.tolist())
                if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
                    raise ValueError("model.mean/std doivent contenir trois valeurs valides.")
                self.input_mean = np.asarray(mean, dtype=np.float32)
                self.input_std = np.asarray(std, dtype=np.float32)
                self.output_name = str(model_spec.get("output_name") or "").strip()
            self.landmarks = load_pitch_landmarks(
                self.schema_path,
                expected_count=self.expected_landmarks,
            )
            self.net = cv2.dnn.readNetFromONNX(str(model))
            self.backend = "onnx_keypoints"
        except Exception as exc:  # pragma: no cover - optional local checkpoint
            self.error = f"Calibration terrain indisponible: {exc}"
            self.net = None
            self.landmarks = []

    def reset_shot(self) -> None:
        self.homography = None
        self.last_result = None
        self.last_success_frame = -10_000
        self.scene_resets += 1

    @staticmethod
    def _softmax_heatmap(heatmap: np.ndarray) -> tuple[float, float, float]:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        flat_index = int(np.argmax(heatmap))
        row, column = np.unravel_index(flat_index, heatmap.shape)
        score = float(heatmap[row, column])
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))
        return float(column), float(row), score

    def _select_output(self, output) -> np.ndarray:
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if not outputs:
            raise ValueError("le modèle terrain n’a produit aucune sortie")
        ranked = []
        for candidate in outputs:
            values = np.asarray(candidate)
            shape = values.shape
            landmark_axis = (
                len(shape) >= 2
                and (
                    shape[0] == self.expected_landmarks
                    or shape[1] == self.expected_landmarks
                    or (len(shape) >= 3 and shape[-2] == self.expected_landmarks)
                )
            )
            ranked.append((int(landmark_axis), values.size, values))
        return max(ranked, key=lambda item: (item[0], item[1]))[2]

    def _decode(self, output: np.ndarray, frame_width: int, frame_height: int):
        values = np.asarray(output)
        if values.ndim in {3, 4} and values.shape[0] == 1:
            values = values[0]
        observations: list[tuple[int, float, float, float]] = []
        if values.ndim == 3:
            keypoint_count, heatmap_height, heatmap_width = values.shape
            for index in range(min(keypoint_count, len(self.landmarks))):
                x, y, score = self._softmax_heatmap(values[index])
                observations.append(
                    (
                        index,
                        x * frame_width / max(heatmap_width - 1, 1),
                        y * frame_height / max(heatmap_height - 1, 1),
                        score,
                    )
                )
            return observations
        values = np.squeeze(values)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(f"sortie terrain non reconnue: shape={values.shape}")
        for index, row in enumerate(values[: len(self.landmarks)]):
            x, y = float(row[0]), float(row[1])
            score = float(row[2]) if row.shape[0] >= 3 else 1.0
            # Coordinate heads commonly emit normalized or input-pixel positions.
            if -0.01 <= x <= 1.01 and -0.01 <= y <= 1.01:
                x *= frame_width
                y *= frame_height
            else:
                x *= frame_width / self.input_width
                y *= frame_height / self.input_height
            observations.append((index, x, y, score))
        return observations

    @classmethod
    def _valid_projection(cls, homography: np.ndarray, frame_width: int, frame_height: int) -> bool:
        import cv2

        image_points = np.asarray(
            [
                [frame_width * 0.5, frame_height * 0.5],
                [frame_width * 0.25, frame_height * 0.75],
                [frame_width * 0.75, frame_height * 0.75],
            ],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not np.isfinite(projected).all():
            return False
        # Broadcast views may extend outside the field, but not hundreds of metres.
        return bool(
            np.all(projected[:, 0] >= -55)
            and np.all(projected[:, 0] <= cls.PITCH_LENGTH_M + 55)
            and np.all(projected[:, 1] >= -45)
            and np.all(projected[:, 1] <= cls.PITCH_WIDTH_M + 45)
        )

    def _fit(self, observations, frame_width: int, frame_height: int) -> PitchCalibrationResult | None:
        import cv2

        visible = [row for row in observations if math.isfinite(row[3]) and row[3] >= self.confidence]
        self.maximum_visible_landmarks = max(self.maximum_visible_landmarks, len(visible))
        if len(visible) < 4:
            return None
        image = np.asarray([[row[1], row[2]] for row in visible], dtype=np.float32)
        pitch = np.asarray(
            [
                [self.landmarks[row[0]].pitch_x, self.landmarks[row[0]].pitch_y]
                for row in visible
            ],
            dtype=np.float32,
        )
        homography, mask = cv2.findHomography(image, pitch, cv2.RANSAC, 3.0)
        if homography is None or mask is None:
            return None
        inliers = int(mask.reshape(-1).sum())
        inlier_ratio = inliers / max(len(visible), 1)
        projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - pitch, axis=1)
        inlier_errors = errors[mask.reshape(-1).astype(bool)]
        reprojection = float(np.mean(inlier_errors)) if inlier_errors.size else float("inf")
        if (
            inliers < 4
            or inlier_ratio < 0.55
            or not math.isfinite(reprojection)
            or reprojection > 4.0
            or not self._valid_projection(homography, frame_width, frame_height)
        ):
            return None
        return PitchCalibrationResult(
            homography=homography,
            visible_landmarks=len(visible),
            inliers=inliers,
            inlier_ratio=float(inlier_ratio),
            reprojection_error_m=reprojection,
        )

    def update(self, frame, *, scene_cut: bool = False) -> bool:
        import cv2

        self.frame_index += 1
        if scene_cut:
            self.reset_shot()
        should_infer = (
            self.net is not None
            and (
                self.homography is None
                or self.frame_index % self.interval_frames == 0
            )
        )
        if should_infer:
            self.attempts += 1
            try:
                resized = cv2.resize(
                    frame,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = (
                    (rgb - self.input_mean) / self.input_std
                ).transpose(2, 0, 1)[None, ...]
                self.net.setInput(np.ascontiguousarray(tensor, dtype=np.float32))
                if self.output_name:
                    raw_output = self.net.forward(self.output_name)
                else:
                    output_names = list(self.net.getUnconnectedOutLayersNames())
                    raw_output = (
                        self.net.forward(output_names)
                        if len(output_names) > 1
                        else self.net.forward()
                    )
                observations = self._decode(
                    self._select_output(raw_output),
                    frame.shape[1],
                    frame.shape[0],
                )
                result = self._fit(observations, frame.shape[1], frame.shape[0])
                if result is not None:
                    self.homography = result.homography
                    self.last_result = result
                    self.last_success_frame = self.frame_index
                    self.successes += 1
            except Exception as exc:  # pragma: no cover - optional runtime/model failure
                self.error = f"Calibration terrain désactivée pendant l’analyse: {exc}"
                self.net = None
                self.backend = "disabled"
        if self.homography is not None and self.frame_index - self.last_success_frame <= self.hold_frames:
            self.projected_frames += 1
            return True
        self.homography = None
        return False

    def project(self, x: float, y: float) -> tuple[float, float] | None:
        projected = self.project_unbounded(x, y)
        if projected is None:
            return None
        x_m, y_m = projected
        if not (-15 <= x_m <= self.PITCH_LENGTH_M + 15 and -15 <= y_m <= self.PITCH_WIDTH_M + 15):
            return None
        return x_m, y_m

    def project_unbounded(self, x: float, y: float) -> tuple[float, float] | None:
        import cv2

        if self.homography is None:
            return None
        point = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.homography)[0, 0]
        if not np.isfinite(projected).all():
            return None
        x_m, y_m = float(projected[0]), float(projected[1])
        if abs(x_m) > 1_000 or abs(y_m) > 1_000:
            return None
        return x_m, y_m

    @classmethod
    def outside_distance(cls, x: float, y: float) -> float:
        delta_x = max(0.0, -float(x), float(x) - cls.PITCH_LENGTH_M)
        delta_y = max(0.0, -float(y), float(y) - cls.PITCH_WIDTH_M)
        return math.hypot(delta_x, delta_y)

    def project_boxes(self, objects: Iterable) -> int:
        projected = 0
        for obj in objects:
            x1, _y1, x2, y2 = [float(value) for value in obj.bbox_xyxy]
            point = self.project_unbounded((x1 + x2) / 2.0, y2)
            if point is None:
                continue
            outside = self.outside_distance(*point)
            obj.metadata["pitch_outside_distance_m"] = round(outside, 3)
            obj.metadata["pitch_inside"] = outside <= 0.25
            if outside > 15.0:
                continue
            obj.pitch_x, obj.pitch_y = point
            obj.metadata["pitch_engine"] = self.backend
            projected += 1
        return projected

    def diagnostics(self) -> dict:
        result = self.last_result
        return {
            "backend": self.backend,
            "ready": self.net is not None,
            "calibrated": self.homography is not None,
            "error": self.error,
            "expected_landmarks": self.expected_landmarks,
            "input_size": [self.input_width, self.input_height],
            "output_name": self.output_name,
            "maximum_visible_landmarks": self.maximum_visible_landmarks,
            "attempts": self.attempts,
            "successful_calibrations": self.successes,
            "projected_frames": self.projected_frames,
            "scene_resets": self.scene_resets,
            "inliers": result.inliers if result else 0,
            "inlier_ratio": round(result.inlier_ratio, 4) if result else 0.0,
            "reprojection_error_m": (
                round(result.reprojection_error_m, 3) if result else None
            ),
        }
