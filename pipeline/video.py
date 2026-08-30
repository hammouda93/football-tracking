from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator
from pathlib import Path

from django.conf import settings

from .types import VideoMetadata


class VideoOpenError(RuntimeError):
    pass


def probe_video(path: str | Path) -> VideoMetadata:
    path = str(Path(path).resolve())
    metadata = _probe_with_ffprobe(path)
    if metadata is not None:
        return metadata
    return _probe_with_opencv(path)


def _probe_with_ffprobe(path: str) -> VideoMetadata | None:
    command = [
        settings.FFPROBE_BINARY,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,codec_name,duration:format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator or 0) / max(float(denominator or 1), 1.0)
    duration_seconds = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    frame_count = int(stream.get("nb_frames") or round(duration_seconds * fps))
    return VideoMetadata(
        path=path,
        duration_ms=int(duration_seconds * 1000),
        fps=fps,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        frame_count=frame_count,
        codec=str(stream.get("codec_name") or ""),
    )


def _probe_with_opencv(path: str) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoOpenError(f"Impossible d’ouvrir la vidéo : {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    codec_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((codec_int >> 8 * index) & 0xFF) for index in range(4)).strip("\x00")
    capture.release()
    if fps <= 0:
        raise VideoOpenError("FPS invalide ou illisible.")
    return VideoMetadata(
        path=path,
        duration_ms=int(frame_count / fps * 1000),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        codec=codec,
    )


def iter_frames(
    path: str | Path,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
    target_fps: float | None = None,
) -> Iterator[tuple[int, object]]:
    """Yield ``(timestamp_ms, BGR frame)`` without extracting images to disk."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(f"Impossible d’ouvrir la vidéo : {path}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0, start_ms))
    target_fps = target_fps or native_fps
    interval_ms = 1000.0 / max(target_fps, 0.1)
    next_timestamp = float(start_ms)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
            if timestamp_ms + 1 < next_timestamp:
                continue
            if end_ms is not None and timestamp_ms > end_ms:
                break
            yield timestamp_ms, frame
            next_timestamp = timestamp_ms + interval_ms
    finally:
        capture.release()


def sample_timestamps(
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: float,
    max_samples: int,
) -> list[int]:
    """Return bounded, regularly spaced timestamps without scanning the video."""
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms, int(end_ms))
    interval_ms = max(1.0, float(interval_ms))
    max_samples = max(1, int(max_samples))
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return []
    interval_ms = max(interval_ms, duration_ms / max_samples)
    count = min(max_samples, max(1, math.ceil(duration_ms / interval_ms)))
    return [
        start_ms + int(index * interval_ms)
        for index in range(count)
        if start_ms + int(index * interval_ms) < end_ms
    ]


def iter_sampled_frames(
    path: str | Path,
    timestamps_ms: list[int],
) -> Iterator[tuple[int, object]]:
    """Seek directly to sparse timestamps instead of decoding every prior frame.

    ``iter_frames`` remains the sequential reader used by tracking, where temporal
    continuity matters. Quality control only needs representative still frames and
    would otherwise decode an entire multi-gigabyte match just to retain a few of
    them.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(f"Impossible d’ouvrir la vidéo : {path}")
    try:
        for timestamp_ms in timestamps_ms:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_ms))
            ok, frame = capture.read()
            if not ok:
                continue
            # Use the requested timestamp: some OpenCV/codec combinations report
            # zero or the previous keyframe after a random seek.
            yield timestamp_ms, frame
    finally:
        capture.release()


def video_time_to_match_time(timestamp_ms: int, periods: list) -> tuple[object | None, int | None]:
    for period in periods:
        if period.video_start_ms <= timestamp_ms <= period.video_end_ms:
            relative = timestamp_ms - period.video_start_ms
            return period, period.match_clock_start_ms + relative
    return None, None
