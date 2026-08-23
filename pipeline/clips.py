from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from django.conf import settings

from .types import EventCandidate


@dataclass(slots=True)
class ClipWindow:
    start_ms: int
    end_ms: int
    event_indexes: list[int]
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


class ClipPlanner:
    WINDOWS = {
        "goal": (12_000, 10_000),
        "shot": (9_000, 7_000),
        "pass": (5_000, 5_000),
        "carry": (5_000, 4_000),
        "duel": (5_000, 5_000),
        "aerial_duel": (5_000, 5_000),
        "tackle": (5_000, 5_000),
    }

    def plan(
        self,
        events: list[EventCandidate],
        *,
        video_duration_ms: int,
        merge_gap_ms: int = 2_000,
    ) -> list[ClipWindow]:
        windows: list[ClipWindow] = []
        for index, event in enumerate(events):
            before, after = self.WINDOWS.get(event.event_type, (5_000, 5_000))
            candidate = ClipWindow(
                start_ms=max(0, event.timestamp_ms - before),
                end_ms=min(video_duration_ms, event.timestamp_ms + after),
                event_indexes=[index],
                label=event.event_type,
            )
            if windows and candidate.start_ms <= windows[-1].end_ms + merge_gap_ms:
                windows[-1].end_ms = max(windows[-1].end_ms, candidate.end_ms)
                windows[-1].event_indexes.append(index)
                if candidate.label not in windows[-1].label:
                    windows[-1].label += f"-{candidate.label}"
            else:
                windows.append(candidate)
        return windows


def render_clip(source_path: str, output_path: str, window: ClipWindow) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_seconds = window.start_ms / 1000.0
    duration_seconds = max(0.1, (window.end_ms - window.start_ms) / 1000.0)
    command = [
        settings.FFMPEG_BINARY,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        source_path,
        "-t",
        f"{duration_seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=max(120, int(duration_seconds * 5)))
