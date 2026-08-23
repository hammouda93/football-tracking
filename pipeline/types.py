from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ObjectRole(StrEnum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"
    OTHER = "other"


class PlayState(StrEnum):
    IN_PLAY = "in_play"
    CONTROLLED = "controlled"
    CONTESTED = "contested"
    LOOSE = "loose"
    OUT = "out"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class VideoMetadata:
    path: str
    duration_ms: int
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimeSpan:
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    label: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameSignal:
    timestamp_ms: int
    field_score: float
    sharpness: float
    brightness: float
    motion_score: float = 0.0
    scene_cut: bool = False
    line_score: float = 0.0
    ball_visible: bool = False
    replay_probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrackedObject:
    track_id: str
    role: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    team_key: str | None = None
    player_key: str | None = None
    shirt_number: int | None = None
    image_x: float | None = None
    image_y: float | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameAnalysis:
    timestamp_ms: int
    width: int
    height: int
    field_score: float
    objects: list[TrackedObject] = field(default_factory=list)
    scene_cut: bool = False
    replay_probability: float = 0.0
    coordinate_space: str = "image_normalized"
    camera: dict[str, Any] = field(default_factory=dict)

    @property
    def ball(self) -> TrackedObject | None:
        return next((item for item in self.objects if item.role == ObjectRole.BALL), None)

    @property
    def athletes(self) -> list[TrackedObject]:
        return [
            item
            for item in self.objects
            if item.role in {ObjectRole.PLAYER, ObjectRole.GOALKEEPER}
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PossessionSample:
    timestamp_ms: int
    state: str
    team_key: str | None
    player_key: str | None
    ball_x: float | None
    ball_y: float | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PossessionSpan:
    start_ms: int
    end_ms: int
    state: str
    team_key: str | None
    player_key: str | None
    confidence: float

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventCandidate:
    timestamp_ms: int
    event_type: str
    team_key: str | None = None
    player_key: str | None = None
    recipient_key: str | None = None
    outcome: str = "unknown"
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    confidence: float = 0.0
    qualifiers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
