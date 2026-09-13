from __future__ import annotations

import bisect
import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.types import FrameAnalysis, ObjectRole, TrackedObject

GSR_SCHEMA = "football-tracking.gsr/v1"
ATHLETE_ROLES = {
    str(ObjectRole.PLAYER),
    str(ObjectRole.GOALKEEPER),
    str(ObjectRole.REFEREE),
}


class GSRContractError(ValueError):
    """Raised when an external engine result cannot be trusted."""


@dataclass(slots=True)
class GSRFrameStore:
    """Validated, timestamp-indexed GSR results.

    The external process stays isolated from Django and may use TrackLab,
    sn-gamestate or the 2025 challenge winner. Only this small JSON contract
    crosses the process boundary.
    """

    engine: str
    engine_revision: str
    fps: float
    frames: list[FrameAnalysis]
    timestamps: list[int]
    metadata: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        home_team_cluster: str = "B",
    ) -> GSRFrameStore:
        result_path = Path(path)
        if not result_path.is_file():
            raise GSRContractError(f"Résultat GSR introuvable : {result_path}")

        records = list(_read_records(result_path))
        if not records:
            raise GSRContractError("Le résultat GSR est vide.")

        metadata_record = next(
            (item for item in records if str(item.get("type", "")) == "metadata"),
            {},
        )
        schema = str(metadata_record.get("schema") or "")
        if schema != GSR_SCHEMA:
            raise GSRContractError(
                f"Schéma GSR incompatible : {schema or 'absent'} (attendu {GSR_SCHEMA})."
            )
        engine = str(metadata_record.get("engine") or "external").strip().lower()
        engine_revision = str(metadata_record.get("engine_revision") or "unknown")
        fps = _positive_float(metadata_record.get("fps"), default=0.0)

        parsed: list[FrameAnalysis] = []
        seen_timestamps: set[int] = set()
        for record in records:
            if str(record.get("type", "frame")) != "frame":
                continue
            frame = _parse_frame(
                record,
                fps=fps,
                home_team_cluster=home_team_cluster,
                engine=engine,
            )
            if frame.timestamp_ms in seen_timestamps:
                raise GSRContractError(f"Timestamp GSR dupliqué : {frame.timestamp_ms} ms.")
            seen_timestamps.add(frame.timestamp_ms)
            parsed.append(frame)

        if not parsed:
            raise GSRContractError(
                "Aucune image GSR exploitable. L’export doit contenir bbox_xyxy, "
                "track_id, role et timestamp_ms pour chaque image."
            )
        parsed.sort(key=lambda item: item.timestamp_ms)
        return cls(
            engine=engine,
            engine_revision=engine_revision,
            fps=fps,
            frames=parsed,
            timestamps=[item.timestamp_ms for item in parsed],
            metadata=dict(metadata_record),
        )

    def nearest(self, timestamp_ms: int, *, tolerance_ms: int) -> FrameAnalysis | None:
        """Return an independent copy of the closest external frame."""

        if not self.timestamps:
            return None
        index = bisect.bisect_left(self.timestamps, int(timestamp_ms))
        candidates = []
        if index < len(self.timestamps):
            candidates.append(index)
        if index:
            candidates.append(index - 1)
        best = min(candidates, key=lambda value: abs(self.timestamps[value] - timestamp_ms))
        if abs(self.timestamps[best] - timestamp_ms) > max(0, int(tolerance_ms)):
            return None
        frame = copy.deepcopy(self.frames[best])
        # Downstream possession and artifacts use the source-video clock that
        # was actually requested, while retaining the GSR timestamp for audit.
        frame.diagnostics["gsr_source_timestamp_ms"] = frame.timestamp_ms
        frame.timestamp_ms = int(timestamp_ms)
        return frame


def _read_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GSRContractError(f"JSON GSR illisible : {exc}") from exc
        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("type", "metadata")
            metadata.setdefault("schema", payload.get("schema"))
            metadata.setdefault("engine", payload.get("engine"))
            metadata.setdefault("engine_revision", payload.get("engine_revision"))
            metadata.setdefault("fps", payload.get("fps"))
            yield metadata
            yield from payload["frames"]
            return
        if isinstance(payload, list):
            yield from payload
            return
        raise GSRContractError(
            "Le JSON GSR doit contenir une liste de frames et les métadonnées du schéma."
        )

    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GSRContractError(
                        f"NDJSON GSR invalide à la ligne {line_number} : {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise GSRContractError(
                        f"La ligne {line_number} du résultat GSR n’est pas un objet."
                    )
                yield record
    except OSError as exc:
        raise GSRContractError(f"Résultat GSR illisible : {exc}") from exc


def _parse_frame(
    payload: dict[str, Any],
    *,
    fps: float,
    home_team_cluster: str,
    engine: str,
) -> FrameAnalysis:
    timestamp_ms = payload.get("timestamp_ms")
    if timestamp_ms is None and payload.get("frame_index") is not None and fps > 0:
        timestamp_ms = round(float(payload["frame_index"]) * 1_000 / fps)
    if timestamp_ms is None:
        raise GSRContractError("Une frame GSR ne contient ni timestamp_ms ni frame_index+fps.")
    try:
        timestamp_ms = int(timestamp_ms)
        width = int(payload["width"])
        height = int(payload["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GSRContractError("Une frame GSR doit déclarer width et height valides.") from exc
    if timestamp_ms < 0 or width <= 0 or height <= 0:
        raise GSRContractError("Timestamp ou dimensions GSR invalides.")

    objects_payload = payload.get("objects") or []
    if not isinstance(objects_payload, list):
        raise GSRContractError("Le champ objects d’une frame GSR doit être une liste.")
    objects = [
        _parse_object(
            item,
            width=width,
            height=height,
            home_team_cluster=home_team_cluster,
            engine=engine,
        )
        for item in objects_payload
    ]
    objects = [item for item in objects if item is not None]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics.update(
        {
            "gsr_engine": engine,
            "raw_athlete_detections": sum(
                item.role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER} for item in objects
            ),
            "raw_referee_detections": sum(item.role == ObjectRole.REFEREE for item in objects),
            "raw_detections": [
                {
                    "bbox": list(item.bbox_xyxy),
                    "role": str(item.role),
                    "confidence": item.confidence,
                }
                for item in objects
            ],
        }
    )
    return FrameAnalysis(
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        field_score=float(payload.get("field_score", 1.0)),
        objects=objects,
        scene_cut=bool(payload.get("scene_cut", False)),
        replay_probability=float(payload.get("replay_probability", 0.0)),
        coordinate_space=str(payload.get("coordinate_space", "image_pixels+pitch_meters")),
        camera=dict(payload.get("camera") or {}),
        diagnostics=diagnostics,
    )


def _parse_object(
    payload: Any,
    *,
    width: int,
    height: int,
    home_team_cluster: str,
    engine: str,
) -> TrackedObject | None:
    if not isinstance(payload, dict):
        raise GSRContractError("Chaque objet GSR doit être un dictionnaire.")
    role = str(payload.get("role") or "").strip().lower()
    if role not in ATHLETE_ROLES:
        # Ball is deliberately owned by the local ball sidecar. Unknown GSR
        # categories are ignored rather than becoming false athletes.
        return None
    track_id = str(payload.get("track_id") or "").strip()
    if not track_id:
        raise GSRContractError("Un athlète GSR ne contient pas de track_id.")

    bbox = payload.get("bbox_xyxy")
    if bbox is None and payload.get("bbox_ltwh") is not None:
        x, y, w, h = _four_numbers(payload["bbox_ltwh"], "bbox_ltwh")
        bbox = [x, y, x + w, y + h]
    x1, y1, x2, y2 = _four_numbers(bbox, "bbox_xyxy")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise GSRContractError(
            f"bbox_xyxy hors image pour la piste {track_id}: {[x1, y1, x2, y2]}."
        )

    pitch = payload.get("pitch_xy") or payload.get("bbox_pitch") or {}
    pitch_x = pitch_y = None
    if isinstance(pitch, dict):
        pitch_x = _optional_float(
            pitch["x"] if pitch.get("x") is not None else pitch.get("x_bottom_middle")
        )
        pitch_y = _optional_float(
            pitch["y"] if pitch.get("y") is not None else pitch.get("y_bottom_middle")
        )
    elif isinstance(pitch, (list, tuple)) and len(pitch) >= 2:
        pitch_x, pitch_y = _optional_float(pitch[0]), _optional_float(pitch[1])

    shirt_number = payload.get("shirt_number", payload.get("jersey_number"))
    try:
        shirt_number = int(shirt_number) if shirt_number is not None else None
    except (TypeError, ValueError):
        shirt_number = None
    if shirt_number is not None and not 0 <= shirt_number <= 99:
        shirt_number = None

    team_key = _map_team_key(payload, home_team_cluster=home_team_cluster)
    confidence = min(1.0, max(0.0, _positive_float(payload.get("confidence"), default=1.0)))
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "gsr_engine": engine,
            "gsr_source_track_id": track_id,
            "team_confidence": _optional_float(payload.get("team_confidence")),
            "role_confidence": _optional_float(payload.get("role_confidence")),
            "jersey_confidence": _optional_float(payload.get("jersey_confidence")),
            "reid_confidence": _optional_float(payload.get("reid_confidence")),
        }
    )
    return TrackedObject(
        track_id=f"gsr-{track_id}",
        role=role,
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=confidence,
        team_key=team_key,
        player_key=f"gsr-{track_id}",
        shirt_number=shirt_number,
        image_x=((x1 + x2) / 2.0) / width,
        image_y=y2 / height,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        metadata=metadata,
    )


def _map_team_key(payload: dict[str, Any], *, home_team_cluster: str) -> str | None:
    value = payload.get("team_key")
    if value is None:
        value = payload.get("team_cluster", payload.get("team"))
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"home", "away"}:
        return normalized
    cluster = {
        "0": "A",
        "1": "B",
        "a": "A",
        "b": "B",
        "left": "A",
        "right": "B",
    }.get(normalized)
    if cluster is None:
        return None
    return "home" if cluster == str(home_team_cluster).upper() else "away"


def _four_numbers(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise GSRContractError(f"{label} doit contenir exactement quatre nombres.")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise GSRContractError(f"{label} contient une valeur non numérique.") from exc


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
