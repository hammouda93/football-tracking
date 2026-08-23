from __future__ import annotations

import numpy as np

from pipeline.types import FrameAnalysis

from .base import VisionProvider


class HeuristicVisionProvider(VisionProvider):
    """Dependency-light provider for quality, camera and period analysis.

    It deliberately emits no player or ball identities. It lets the interface and
    time segmentation run before local ML weights are installed.
    """

    def __init__(self):
        self.previous_gray = None

    def reset(self) -> None:
        self.previous_gray = None

    def analyze_frame(self, frame, timestamp_ms: int) -> FrameAnalysis:
        import cv2

        height, width = frame.shape[:2]
        working = frame
        scale = min(1.0, 640.0 / max(width, 1))
        if scale < 1:
            working = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        green_mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([100, 255, 255]))
        green_ratio = float(np.count_nonzero(green_mask) / green_mask.size)
        scene_cut = False
        if self.previous_gray is not None:
            scene_cut = float(cv2.absdiff(gray, self.previous_gray).mean() / 255.0) > 0.32
        self.previous_gray = gray
        return FrameAnalysis(
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            field_score=max(0.0, min(1.0, green_ratio * 1.6)),
            objects=[],
            scene_cut=scene_cut,
            replay_probability=0.7 if scene_cut and green_ratio < 0.12 else 0.0,
        )
