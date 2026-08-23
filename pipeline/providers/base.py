from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pipeline.types import FrameAnalysis


class VisionProvider(ABC):
    @abstractmethod
    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        raise NotImplementedError

    def reset(self) -> None:
        return None


def build_provider(name: str, **config: Any) -> VisionProvider:
    normalized = (name or "heuristic").strip().lower()
    if normalized == "heuristic":
        from .heuristic import HeuristicVisionProvider

        return HeuristicVisionProvider()
    if normalized == "yolo":
        from .yolo import YoloVisionProvider

        return YoloVisionProvider(**config)
    raise ValueError(f"Backend d’analyse inconnu : {name}")
