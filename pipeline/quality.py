from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Callable

import numpy as np

from .types import FrameSignal, VideoMetadata
from .video import iter_sampled_frames, sample_timestamps


@dataclass(slots=True)
class QualityReport:
    score: float
    grade: str
    metrics: dict
    signals: list[FrameSignal]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "grade": self.grade,
            "metrics": self.metrics,
            "signals": [signal.to_dict() for signal in self.signals],
        }


class VideoQualityAnalyzer:
    def __init__(self, sample_seconds: float = 3.0, max_samples: int = 1_200):
        self.sample_seconds = max(0.5, sample_seconds)
        self.max_samples = max_samples

    def analyze(
        self,
        metadata: VideoMetadata,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> QualityReport:
        import cv2

        # Spread a bounded number of samples over the *whole* recording. This is
        # important when the upload contains a long pre-match or post-match section.
        duration_seconds = max(metadata.duration_ms / 1000.0, self.sample_seconds)
        effective_sample_seconds = max(
            self.sample_seconds,
            duration_seconds / max(self.max_samples, 1),
        )
        timestamps = sample_timestamps(
            start_ms=0,
            end_ms=metadata.duration_ms,
            interval_ms=effective_sample_seconds * 1000.0,
            max_samples=self.max_samples,
        )
        signals: list[FrameSignal] = []
        previous_gray = None
        last_callback_percent = -1
        for timestamp_ms, frame in iter_sampled_frames(metadata.path, timestamps):
            height, width = frame.shape[:2]
            scale = min(1.0, 640.0 / max(width, 1))
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            green_mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([100, 255, 255]))
            green_ratio = float(np.count_nonzero(green_mask) / green_mask.size)
            edges = cv2.Canny(gray, 60, 160)
            line_ratio = float(np.count_nonzero(edges) / edges.size)
            motion_score = 0.0
            scene_cut = False
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                motion_score = float(difference.mean() / 255.0)
                scene_cut = motion_score > 0.32
            previous_gray = gray
            field_score = max(0.0, min(1.0, green_ratio * 1.55 + line_ratio * 1.2))
            signals.append(
                FrameSignal(
                    timestamp_ms=timestamp_ms,
                    field_score=field_score,
                    sharpness=sharpness,
                    brightness=brightness,
                    motion_score=motion_score,
                    scene_cut=scene_cut,
                    line_score=line_ratio,
                    replay_probability=0.7 if scene_cut and field_score < 0.22 else 0.0,
                )
            )
            if progress_callback is not None:
                completed = len(signals)
                callback_percent = int(100 * completed / max(len(timestamps), 1))
                if callback_percent != last_callback_percent:
                    progress_callback(completed, len(timestamps))
                    last_callback_percent = callback_percent
        if not signals:
            return QualityReport(score=0.0, grade="reject", metrics={}, signals=[])

        sharpness = fmean(item.sharpness for item in signals)
        brightness = fmean(item.brightness for item in signals)
        field_score = fmean(item.field_score for item in signals)
        field_presence = sum(item.field_score >= 0.25 for item in signals) / len(signals)
        scene_cut_rate = sum(item.scene_cut for item in signals) / len(signals)
        resolution_score = min(1.0, (metadata.width * metadata.height) / (1920 * 1080))
        sharpness_score = min(1.0, sharpness / 180.0)
        exposure_score = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0)
        stability_score = max(0.0, 1.0 - scene_cut_rate * 4.0)
        score = 100.0 * (
            resolution_score * 0.24
            + sharpness_score * 0.22
            + exposure_score * 0.14
            + field_score * 0.20
            + field_presence * 0.12
            + stability_score * 0.08
        )
        if score >= 78 and metadata.width >= 1920:
            grade = "A"
        elif score >= 60 and metadata.width >= 1280:
            grade = "B"
        elif score >= 40:
            grade = "C"
        else:
            grade = "reject"
        metrics = {
            "resolution": f"{metadata.width}x{metadata.height}",
            "fps": round(metadata.fps, 3),
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "field_score": round(field_score, 4),
            "field_presence": round(field_presence, 4),
            "scene_cut_rate": round(scene_cut_rate, 4),
            "sample_count": len(signals),
            "sample_interval_seconds": round(effective_sample_seconds, 3),
        }
        return QualityReport(score=round(score, 2), grade=grade, metrics=metrics, signals=signals)
