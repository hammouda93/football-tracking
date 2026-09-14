from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class JerseyObservation:
    number: int
    confidence: float
    backend: str


class WeightedNumberVote:
    """Confidence-weighted, temporally distinct jersey-number vote."""

    def __init__(self) -> None:
        self.scores: Counter[int] = Counter()
        self.observations: Counter[int] = Counter()
        self.timestamps: set[int] = set()

    def add(self, number: int, confidence: float, timestamp_ms: int) -> None:
        number = int(number)
        if number < 0 or number > 99 or timestamp_ms in self.timestamps:
            return
        confidence = max(0.0, min(1.0, float(confidence)))
        if confidence <= 0:
            return
        self.timestamps.add(int(timestamp_ms))
        # Squaring prevents a pile of weak OCR guesses from beating a few clear reads.
        self.scores[number] += confidence * confidence
        self.observations[number] += 1

    def result(
        self,
        *,
        minimum_observations: int = 3,
        minimum_score: float = 0.8,
        minimum_share: float = 0.62,
        minimum_margin: float = 0.18,
    ) -> tuple[int | None, float]:
        if not self.scores:
            return None, 0.0
        ranked = self.scores.most_common(2)
        number, score = ranked[0]
        total = float(sum(self.scores.values()))
        share = float(score / max(total, 1e-9))
        runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = float((score - runner_up) / max(score, 1e-9))
        observations = int(self.observations[number])
        confidence = max(0.0, min(1.0, 0.55 * share + 0.25 * margin + 0.20 * min(1, observations / 5)))
        if (
            observations < minimum_observations
            or score < minimum_score
            or share < minimum_share
            or margin < minimum_margin
        ):
            return None, confidence
        return int(number), confidence


class JerseyNumberRecognizer:
    """Optional native jersey OCR.

    Supported engines are an ONNX 0..99 classifier and EasyOCR. The detector
    weights used for players are deliberately never treated as OCR weights.
    """

    def __init__(
        self,
        *,
        engine: str = "auto",
        model_path: str = "",
        device: str = "cpu",
        minimum_box_height: int = 72,
        input_width: int = 96,
        input_height: int = 128,
    ) -> None:
        self.requested_engine = str(engine or "auto").strip().lower()
        self.model_path = str(model_path or "").strip()
        self.device = str(device or "cpu")
        self.minimum_box_height = max(24, int(minimum_box_height))
        self.input_width = max(16, int(input_width))
        self.input_height = max(16, int(input_height))
        self.backend = "disabled"
        self.error = ""
        self.model = None
        self.calls = 0
        self.reads = 0
        self.accepted = 0
        self._initialize()

    def _initialize(self) -> None:
        engine = self.requested_engine
        if engine not in {"auto", "off", "onnx", "easyocr"}:
            self.error = f"Moteur OCR maillot inconnu: {engine}"
            return
        if engine == "off":
            return
        if self.model_path:
            path = Path(self.model_path)
            if not path.is_file():
                self.error = f"Poids OCR maillot absents: {path}"
                return
            if path.suffix.lower() != ".onnx":
                self.error = "Le classifieur maillot natif doit être un fichier .onnx."
                return
            try:
                import cv2

                self.model = cv2.dnn.readNetFromONNX(str(path))
                self.backend = "onnx_0_99"
                return
            except Exception as exc:  # pragma: no cover - optional local checkpoint
                self.error = f"OCR maillot ONNX indisponible: {exc}"
                return
        if engine == "onnx":
            self.error = "NATIVE_GSR_JERSEY_MODEL_PATH est vide."
            return
        if engine == "easyocr":
            try:
                import easyocr

                use_gpu = self.device.strip().lower() not in {"", "cpu", "none"}
                self.model = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self.backend = "easyocr_digits"
            except Exception as exc:  # pragma: no cover - optional dependency
                self.error = f"EasyOCR indisponible: {exc}"

    @staticmethod
    def _jersey_crop(frame, box: Iterable[float]):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        # Exclude the head and legs; front/back rotation is handled by temporal voting.
        left = max(0, min(width - 1, int(round(x1 + 0.04 * box_width))))
        right = max(left + 1, min(width, int(round(x2 - 0.04 * box_width))))
        top = max(0, min(height - 1, int(round(y1 + 0.10 * box_height))))
        bottom = max(top + 1, min(height, int(round(y1 + 0.62 * box_height))))
        return frame[top:bottom, left:right]

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        values = values - float(np.max(values))
        exp = np.exp(values)
        return exp / max(float(exp.sum()), 1e-9)

    @staticmethod
    def _parse_number(text: str) -> int | None:
        digits = "".join(re.findall(r"\d", str(text)))
        if not digits or len(digits) > 2:
            return None
        value = int(digits)
        return value if 0 <= value <= 99 else None

    def _onnx(self, crop) -> JerseyObservation | None:
        import cv2

        resized = cv2.resize(
            crop,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
        self.model.setInput(tensor)
        output = np.asarray(self.model.forward()).reshape(-1)
        if output.size not in {100, 101}:
            raise ValueError(f"sortie OCR attendue 100/101 classes, reçue {output.size}")
        probabilities = self._softmax(output[:100])
        number = int(np.argmax(probabilities))
        confidence = float(probabilities[number])
        return JerseyObservation(number, confidence, self.backend)

    def _easyocr(self, crop) -> JerseyObservation | None:
        import cv2

        scale = max(2.0, 160.0 / max(crop.shape[0], 1))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results = self.model.readtext(
            enlarged,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
            min_size=5,
            text_threshold=0.35,
            low_text=0.20,
            link_threshold=0.20,
        )
        candidates: list[JerseyObservation] = []
        for _polygon, text, confidence in results:
            number = self._parse_number(text)
            if number is not None:
                candidates.append(
                    JerseyObservation(number, max(0.0, min(1.0, float(confidence))), self.backend)
                )
        return max(candidates, key=lambda item: item.confidence, default=None)

    def recognize(self, frame, box: Iterable[float]) -> JerseyObservation | None:
        self.calls += 1
        _x1, y1, _x2, y2 = [float(value) for value in box]
        if self.model is None or y2 - y1 < self.minimum_box_height:
            return None
        crop = self._jersey_crop(frame, box)
        if crop.size < 48:
            return None
        self.reads += 1
        try:
            observation = (
                self._onnx(crop) if self.backend == "onnx_0_99" else self._easyocr(crop)
            )
        except Exception as exc:  # pragma: no cover - optional runtime/model failure
            self.error = f"OCR maillot désactivé pendant l’analyse: {exc}"
            self.model = None
            self.backend = "disabled"
            return None
        if observation is None or not math.isfinite(observation.confidence):
            return None
        if observation.confidence < 0.18:
            return None
        self.accepted += 1
        return observation

    def diagnostics(self) -> dict:
        return {
            "backend": self.backend,
            "ready": self.model is not None,
            "error": self.error,
            "calls": self.calls,
            "eligible_crops": self.reads,
            "accepted_reads": self.accepted,
        }
