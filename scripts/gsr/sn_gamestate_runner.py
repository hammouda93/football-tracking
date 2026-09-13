#!/usr/bin/env python3
"""Run the official sn-gamestate pipeline on application video windows.

This file deliberately contains no SoccerNet or TrackLab source code.  It is a
small process adapter: it prepares one external video, invokes the separately
installed GPL pipeline, and converts its TrackerState into our versioned NDJSON
contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "football-tracking.gsr/v1"
UPSTREAM = "https://github.com/SoccerNet/sn-gamestate"
WINDOWS_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
MODULE_PROGRESS = {
    "bbox_detector": (10.0, "Détection des joueurs"),
    "reid": (28.0, "Empreintes visuelles Re-ID"),
    "track": (46.0, "Association temporelle TrackLab"),
    "pitch": (58.0, "Détection du terrain"),
    "calibration": (68.0, "Calibration caméra-terrain"),
    "jersey_number_detect": (78.0, "Lecture des numéros de maillot"),
    "tracklet_agg": (86.0, "Vote par piste"),
    "team": (91.0, "Regroupement automatique des équipes"),
    "team_side": (95.0, "Stabilisation des équipes"),
}


@dataclass(frozen=True)
class SegmentMap:
    period: int
    index: int
    source_start_ms: int
    source_end_ms: int
    composite_start_frame: int
    frame_count: int

    @property
    def composite_end_frame(self) -> int:
        return self.composite_start_frame + self.frame_count


def local_path(value: str | os.PathLike[str]) -> Path:
    """Translate a Windows drive path when this runner executes inside WSL."""

    raw = str(value)
    if os.name != "nt":
        match = WINDOWS_PATH.match(raw)
        if match:
            rest = match.group("rest").replace("\\", "/").lstrip("/")
            return Path("/mnt") / match.group("drive").lower() / rest
    return Path(raw).expanduser()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class ProgressWriter:
    def __init__(self, path: Path, total_video_ms: int, total_frames: int):
        self.path = path
        self.total_video_ms = max(1, int(total_video_ms))
        self.total_frames = max(1, int(total_frames))
        self.started = time.monotonic()

    def update(
        self,
        progress: float,
        label: str,
        *,
        processed_video_ms: int = 0,
        frames_processed: int = 0,
    ) -> None:
        progress = max(0.0, min(100.0, float(progress)))
        elapsed = max(0.0, time.monotonic() - self.started)
        if progress > 0.5:
            eta = elapsed * (100.0 - progress) / progress
        else:
            eta = None
        speed_x = max(0, processed_video_ms) / 1_000.0 / elapsed if elapsed > 0 else 0.0
        atomic_json(
            self.path,
            {
                "progress": round(progress, 2),
                "processed_video_ms": max(0, int(processed_video_ms)),
                "total_video_ms": self.total_video_ms,
                "frames_processed": max(0, int(frames_processed)),
                "frames_total": self.total_frames,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta) if eta is not None else None,
                "speed_x": round(speed_x, 3),
                "label": label,
            },
        )


def run_checked(argv: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(argv)
    print(f"[sn-gamestate bridge] {printable}", flush=True)
    completed = subprocess.run(argv, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"La commande s'est arrêtée avec le code {completed.returncode}: {printable}"
        )


def video_info(path: Path) -> tuple[int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Vidéo temporaire illisible : {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Métadonnées vidéo invalides : {path}")
    return width, height, frame_count


def prepare_external_video(
    *,
    source: Path,
    windows: list[dict[str, Any]],
    target_fps: float,
    work_dir: Path,
    ffmpeg: str,
    progress: ProgressWriter,
) -> tuple[Path, list[SegmentMap], int, int]:
    segments: list[Path] = []
    maps: list[SegmentMap] = []
    width = height = 0
    composite_frame = 0
    separator_path = work_dir / "separator.mp4"
    separator_frames = max(1, round(target_fps * 6.0))

    for position, window in enumerate(windows):
        start_ms = int(window["start_ms"])
        end_ms = int(window["end_ms"])
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"Fenêtre GSR invalide : {window}")
        segment_path = work_dir / (
            f"period-{int(window['period'])}-window-{int(window['index'])}.mp4"
        )
        progress.update(
            1.0 + 5.0 * position / max(1, len(windows)),
            f"Préparation vidéo {position + 1}/{len(windows)}",
        )
        run_checked(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_ms / 1_000.0:.3f}",
                "-i",
                str(source),
                "-t",
                f"{(end_ms - start_ms) / 1_000.0:.3f}",
                "-vf",
                f"fps={target_fps:.6f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(segment_path),
            ]
        )
        segment_width, segment_height, frame_count = video_info(segment_path)
        if not segments:
            width, height = segment_width, segment_height
            run_checked(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:r={target_fps:.6f}",
                    "-t",
                    "6",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(separator_path),
                ]
            )
            _, _, separator_frames = video_info(separator_path)
        elif (segment_width, segment_height) != (width, height):
            raise RuntimeError("Les fenêtres extraites n'ont pas les mêmes dimensions.")

        if segments:
            composite_frame += separator_frames
        maps.append(
            SegmentMap(
                period=int(window["period"]),
                index=int(window["index"]),
                source_start_ms=start_ms,
                source_end_ms=end_ms,
                composite_start_frame=composite_frame,
                frame_count=frame_count,
            )
        )
        composite_frame += frame_count
        segments.append(segment_path)

    if len(segments) == 1:
        return segments[0], maps, width, height

    concat_file = work_dir / "concat.txt"
    concat_entries: list[Path] = []
    for index, segment in enumerate(segments):
        if index:
            concat_entries.append(separator_path)
        concat_entries.append(segment)
    concat_file.write_text(
        "".join(
            f"file '{path.as_posix().replace(chr(39), chr(39) * 2)}'\n" for path in concat_entries
        ),
        encoding="utf-8",
    )
    composite = work_dir / "external-video.mp4"
    run_checked(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(composite),
        ]
    )
    return composite, maps, width, height


def detect_device(*, allow_cpu: bool) -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if not allow_cpu:
        raise RuntimeError(
            "sn-gamestate ne voit aucun GPU CUDA. Son pipeline complet sur CPU est "
            "trop lent pour un test utile. Corrige NVIDIA/WSL, ou ajoute --allow-cpu "
            "uniquement pour accepter explicitement cette lenteur."
        )
    return "cpu (autorisé explicitement)"


def run_tracklab(
    *,
    sn_root: Path,
    video_path: Path,
    state_path: Path,
    run_dir: Path,
    progress: ProgressWriter,
    total_video_ms: int,
    total_frames: int,
) -> None:
    argv = [
        sys.executable,
        "-m",
        "tracklab.main",
        "-cn",
        "soccernet",
        "dataset=youtube",
        f"dataset.video_path={video_path.as_posix()}",
        "dataset.nframes=-1",
        "dataset.nvid=1",
        "dataset.eval_set=val",
        "eval_tracking=false",
        "use_wandb=false",
        "use_rich=false",
        "visualization.save_videos=false",
        f"state.save_file={state_path.as_posix()}",
        "state.load_file=null",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    print(f"[sn-gamestate bridge] {' '.join(argv)}", flush=True)
    process = subprocess.Popen(
        argv,
        cwd=sn_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    module_progress = 8.0
    module_label = "Initialisation des modèles SoccerNet"
    batch_fraction = 0.0
    last_update = 0.0
    stream_closed = False
    while process.poll() is None or not stream_closed:
        try:
            line = messages.get(timeout=0.5)
        except queue.Empty:
            line = ""
        if line is None:
            stream_closed = True
        elif line:
            print(line, end="", flush=True)
            lowered = line.lower().replace(" ", "_")
            for module, (value, label) in MODULE_PROGRESS.items():
                if module in lowered:
                    module_progress, module_label = value, label
                    batch_fraction = 0.0
                    break
            matches = re.findall(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", line)
            if matches:
                current, total = (int(item) for item in matches[-1])
                if total > 0:
                    batch_fraction = max(0.0, min(1.0, current / total))
        now = time.monotonic()
        if now - last_update >= 1.0:
            next_values = sorted(
                value for value, _ in MODULE_PROGRESS.values() if value > module_progress
            )
            next_progress = next_values[0] if next_values else 97.0
            current_progress = module_progress + (next_progress - module_progress) * batch_fraction
            progress.update(current_progress, module_label)
            last_update = now
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"TrackLab s'est arrêté avec le code {return_code}.")
    if not state_path.is_file():
        raise RuntimeError(f"TrackLab n'a pas produit le TrackerState attendu : {state_path}")
    progress.update(
        97.0,
        "Conversion du TrackerState",
        processed_video_ms=total_video_ms,
        frames_processed=total_frames,
    )


def load_tracker_state(path: Path):
    import pandas as pd

    with zipfile.ZipFile(path) as archive:
        detection_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith(".pkl") and not name.endswith("_image.pkl")
            ),
            None,
        )
        image_name = next(
            (name for name in archive.namelist() if name.endswith("_image.pkl")),
            None,
        )
        if not detection_name or not image_name:
            raise RuntimeError("TrackerState incomplet : détections ou images absentes.")
        with archive.open(detection_name) as source:
            detections = pd.read_pickle(source)
        with archive.open(image_name) as source:
            images = pd.read_pickle(source)
    return detections, images


def clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_role(row: Any) -> str | None:
    for key in ("role", "role_detection"):
        value = row.get(key)
        if value is None:
            continue
        role = str(value).strip().lower()
        if role in {"player", "goalkeeper", "referee"}:
            return role
    return None


def cluster_name(value: Any) -> str | None:
    number = clean_number(value)
    if number is None:
        return None
    rounded = round(number)
    return {0: "A", 1: "B"}.get(rounded)


def segment_for_frame(frame: int, segments: list[SegmentMap]) -> SegmentMap | None:
    return next(
        (
            segment
            for segment in segments
            if segment.composite_start_frame <= frame < segment.composite_end_frame
        ),
        None,
    )


def source_timestamp(frame: int, segment: SegmentMap, fps: float) -> int:
    offset = frame - segment.composite_start_frame
    timestamp = segment.source_start_ms + round(offset * 1_000.0 / fps)
    return min(timestamp, segment.source_end_ms - 1)


def reverse_goal_side_maps(detections: Any, images: Any, segments: list[SegmentMap]):
    """Infer goalkeeper cluster from each segment's player pitch distribution."""

    result: dict[tuple[int, int], dict[str, str]] = {}
    image_frames = {
        image_id: int(row.get("frame", image_id)) for image_id, row in images.iterrows()
    }
    for segment in segments:
        xs: dict[str, list[float]] = {"A": [], "B": []}
        for _, row in detections.iterrows():
            frame = image_frames.get(row.get("image_id"))
            if frame is None or not (
                segment.composite_start_frame <= frame < segment.composite_end_frame
            ):
                continue
            cluster = cluster_name(row.get("team_cluster"))
            pitch = row.get("bbox_pitch")
            if cluster and isinstance(pitch, dict):
                x = clean_number(pitch.get("x_bottom_middle"))
                if x is not None:
                    xs[cluster].append(x)
        if xs["A"] and xs["B"]:
            mean_a = sum(xs["A"]) / len(xs["A"])
            mean_b = sum(xs["B"]) / len(xs["B"])
            result[(segment.period, segment.index)] = (
                {"left": "A", "right": "B"} if mean_a < mean_b else {"left": "B", "right": "A"}
            )
    return result


def row_object(
    row: Any,
    *,
    width: int,
    height: int,
    segment: SegmentMap,
    goalkeeper_side_map: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any] | None:
    role = normalize_role(row)
    track_id = row.get("track_id")
    track_number = clean_number(track_id)
    bbox = row.get("bbox_ltwh")
    if role is None or track_number is None or bbox is None:
        return None
    try:
        left, top, box_width, box_height = (float(item) for item in list(bbox)[:4])
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width), left))
    y1 = max(0.0, min(float(height), top))
    x2 = max(0.0, min(float(width), left + box_width))
    y2 = max(0.0, min(float(height), top + box_height))
    if x2 <= x1 or y2 <= y1:
        return None

    cluster = cluster_name(row.get("team_cluster"))
    if role == "goalkeeper" and cluster is None:
        side = str(row.get("team") or "").strip().lower()
        cluster = goalkeeper_side_map.get((segment.period, segment.index), {}).get(side)

    jersey = clean_number(row.get("jersey_number"))
    jersey_number = round(jersey) if jersey is not None and 0 <= jersey <= 99 else None
    pitch = row.get("bbox_pitch")
    pitch_xy = None
    if isinstance(pitch, dict):
        pitch_x = clean_number(pitch.get("x_bottom_middle"))
        pitch_y = clean_number(pitch.get("y_bottom_middle"))
        if pitch_x is not None and pitch_y is not None:
            pitch_xy = [pitch_x, pitch_y]

    confidence = clean_number(row.get("bbox_conf"))
    role_confidence = clean_number(row.get("role_confidence"))
    jersey_confidence = clean_number(row.get("jersey_number_confidence", row.get("jn_confidence")))
    payload: dict[str, Any] = {
        "track_id": str(round(track_number)),
        "role": role,
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": max(0.0, min(1.0, confidence if confidence is not None else 1.0)),
        "team_cluster": cluster,
        "shirt_number": jersey_number,
        "pitch_xy": pitch_xy,
        "role_confidence": role_confidence,
        "jersey_confidence": jersey_confidence,
        "metadata": {
            "period": segment.period,
            "window_index": segment.index,
            "tracklab_team_side": (str(row.get("team")) if row.get("team") is not None else None),
        },
    }
    return payload


def convert_state(
    *,
    state_path: Path,
    result_path: Path,
    segments: list[SegmentMap],
    fps: float,
    width: int,
    height: int,
    revision: str,
    device: str,
) -> dict[int, list[dict[str, Any]]]:
    detections, images = load_tracker_state(state_path)
    image_frames = {
        image_id: int(row.get("frame", image_id)) for image_id, row in images.iterrows()
    }
    goalkeeper_side_map = reverse_goal_side_maps(detections, images, segments)
    objects_by_frame: dict[int, list[dict[str, Any]]] = {}
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for _, row in detections.iterrows():
        frame = image_frames.get(row.get("image_id"))
        if frame is None:
            continue
        segment = segment_for_frame(frame, segments)
        if segment is None:
            continue
        item = row_object(
            row,
            width=width,
            height=height,
            segment=segment,
            goalkeeper_side_map=goalkeeper_side_map,
        )
        if item is None:
            continue
        key = (frame, item["track_id"])
        previous = best.get(key)
        if previous is None or item["confidence"] > previous["confidence"]:
            best[key] = item
    for (frame, _), item in best.items():
        objects_by_frame.setdefault(frame, []).append(item)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    metadata = {
        "type": "metadata",
        "schema": SCHEMA,
        "engine": "tracklab",
        "engine_revision": revision,
        "fps": fps,
        "upstream": UPSTREAM,
        "device": device,
        "windows": [asdict(item) for item in segments],
        "team_method": "sn-gamestate global Re-ID embeddings + KMeans",
        "entered_team_colors_used": False,
    }
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for segment in segments:
            for frame in range(
                segment.composite_start_frame,
                segment.composite_end_frame,
            ):
                payload = {
                    "type": "frame",
                    "timestamp_ms": source_timestamp(frame, segment, fps),
                    "frame_index": frame,
                    "width": width,
                    "height": height,
                    "objects": objects_by_frame.get(frame, []),
                    "diagnostics": {
                        "period": segment.period,
                        "window_index": segment.index,
                        "tracklab_frame": frame,
                    },
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(result_path)
    return objects_by_frame


def write_preview(
    video_path: Path,
    preview_path: Path | None,
    objects_by_frame: dict[int, list[dict[str, Any]]],
) -> None:
    if preview_path is None:
        return
    import cv2

    candidate_frame = max(objects_by_frame, key=lambda key: len(objects_by_frame[key]), default=0)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        return
    colors = {"A": (80, 220, 80), "B": (0, 150, 255), None: (255, 210, 0)}
    for item in objects_by_frame.get(candidate_frame, []):
        x1, y1, x2, y2 = (round(value) for value in item["bbox_xyxy"])
        color = colors.get(item.get("team_cluster"), colors[None])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{item['role']} #{item['track_id']}"
        if item.get("team_cluster"):
            label += f" T{item['team_cluster']}"
        if item.get("shirt_number") is not None:
            label += f" N{item['shirt_number']}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = preview_path.with_suffix(".tmp.jpg")
    if cv2.imwrite(str(temporary), image):
        temporary.replace(preview_path)


def revision_of(sn_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(sn_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def preflight(sn_root: Path, *, allow_cpu: bool) -> dict[str, Any]:
    if not (sn_root / "pyproject.toml").is_file():
        raise RuntimeError(f"Dépôt sn-gamestate introuvable : {sn_root}")
    for command in ("git", "ffmpeg"):
        if not shutil.which(command):
            raise RuntimeError(f"Commande requise introuvable : {command}")
    import cv2
    import pandas
    import sn_gamestate
    import tracklab

    device = detect_device(allow_cpu=allow_cpu)
    return {
        "ok": True,
        "sn_gamestate_root": str(sn_root),
        "revision": revision_of(sn_root),
        "tracklab": getattr(tracklab, "__version__", "installed"),
        "sn_gamestate": getattr(sn_gamestate, "__version__", "installed"),
        "pandas": pandas.__version__,
        "opencv": cv2.__version__,
        "device": device,
    }


def execute(args: argparse.Namespace) -> int:
    sn_root = local_path(args.sn_gamestate_root or os.getenv("SN_GAMESTATE_ROOT", ""))
    if not str(sn_root) or str(sn_root) == ".":
        raise RuntimeError("Indique --sn-gamestate-root ou SN_GAMESTATE_ROOT.")
    allow_cpu = bool(args.allow_cpu or os.getenv("SN_GSR_ALLOW_CPU") == "1")
    if args.preflight:
        print(json.dumps(preflight(sn_root, allow_cpu=allow_cpu), ensure_ascii=False, indent=2))
        return 0

    manifest_path = local_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("engine") != "tracklab":
        raise ValueError("Le manifest n'est pas une requête TrackLab GSR v1.")
    source = local_path(manifest["source_video"])
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo source introuvable depuis WSL : {source}")
    output = manifest.get("output") or {}
    result_path = local_path(output["result_ndjson"])
    progress_path = local_path(output["progress_json"])
    preview_value = str(output.get("live_preview_jpg") or "")
    preview_path = local_path(preview_value) if preview_value else None
    windows = list(manifest.get("windows") or [])
    if not windows:
        raise ValueError("Le manifest ne contient aucune fenêtre confirmée.")
    target_fps = max(1.0, float(manifest.get("target_fps") or 5.0))
    total_video_ms = sum(int(item["end_ms"]) - int(item["start_ms"]) for item in windows)
    total_frames = max(1, round(total_video_ms / 1_000.0 * target_fps))
    progress = ProgressWriter(progress_path, total_video_ms, total_frames)
    progress.update(0.0, "Préflight TrackLab + sn-gamestate")
    details = preflight(sn_root, allow_cpu=allow_cpu)
    print(json.dumps(details, ensure_ascii=False), flush=True)

    keep_work = os.getenv("SN_GSR_KEEP_WORK") == "1"
    work_parent = Path(tempfile.mkdtemp(prefix="football-sn-gsr-"))
    try:
        video_path, segments, width, height = prepare_external_video(
            source=source,
            windows=windows,
            target_fps=target_fps,
            work_dir=work_parent,
            ffmpeg=args.ffmpeg,
            progress=progress,
        )
        state_path = work_parent / "tracklab-state.pklz"
        run_tracklab(
            sn_root=sn_root,
            video_path=video_path,
            state_path=state_path,
            run_dir=work_parent / "hydra-run",
            progress=progress,
            total_video_ms=total_video_ms,
            total_frames=total_frames,
        )
        objects_by_frame = convert_state(
            state_path=state_path,
            result_path=result_path,
            segments=segments,
            fps=target_fps,
            width=width,
            height=height,
            revision=details["revision"],
            device=details["device"],
        )
        write_preview(video_path, preview_path, objects_by_frame)
        progress.update(
            100.0,
            "TrackLab + sn-gamestate terminé",
            processed_video_ms=total_video_ms,
            frames_processed=total_frames,
        )
    finally:
        if keep_work:
            print(f"[sn-gamestate bridge] fichiers conservés : {work_parent}", flush=True)
        else:
            shutil.rmtree(work_parent, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--preflight", action="store_true")
    parser.add_argument("--sn-gamestate-root", default="")
    parser.add_argument("--ffmpeg", default=os.getenv("FFMPEG_BINARY", "ffmpeg"))
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> int:
    try:
        return execute(build_parser().parse_args())
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return a useful error
        print(f"Échec du runner sn-gamestate : {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
