from __future__ import annotations

import math
from collections import deque
from statistics import median

from .types import FrameAnalysis, PlayState, PossessionSample, PossessionSpan


class BallInPlayEngine:
    def __init__(
        self,
        *,
        image_control_radius: float = 0.055,
        pitch_control_radius_m: float = 2.6,
        missing_ball_grace_ms: int = 1_500,
        owner_dwell_ms: int = 250,
    ):
        self.image_control_radius = image_control_radius
        self.pitch_control_radius_m = pitch_control_radius_m
        self.missing_ball_grace_ms = missing_ball_grace_ms
        self.owner_dwell_ms = owner_dwell_ms
        self.last_ball_timestamp: int | None = None
        self.last_ball_position: tuple[float, float] | None = None
        self.current_owner: tuple[str | None, str | None] = (None, None)
        self.pending_owner: tuple[str | None, str | None] = (None, None)
        self.pending_since: int | None = None
        self.owner_history: deque[tuple[str | None, str | None]] = deque(maxlen=5)

    def observe(self, frame: FrameAnalysis) -> PossessionSample:
        if frame.scene_cut or frame.replay_probability >= 0.65 or frame.field_score < 0.14:
            return PossessionSample(
                timestamp_ms=frame.timestamp_ms,
                state=PlayState.UNKNOWN,
                team_key=None,
                player_key=None,
                ball_x=None,
                ball_y=None,
                confidence=max(0.45, 1.0 - frame.field_score),
                metadata={"reason": "broadcast_not_live"},
            )

        ball = frame.ball
        if ball is None:
            if (
                self.last_ball_timestamp is not None
                and frame.timestamp_ms - self.last_ball_timestamp <= self.missing_ball_grace_ms
                and self.current_owner != (None, None)
            ):
                return PossessionSample(
                    timestamp_ms=frame.timestamp_ms,
                    state=PlayState.CONTROLLED,
                    team_key=self.current_owner[0],
                    player_key=self.current_owner[1],
                    ball_x=self.last_ball_position[0] if self.last_ball_position else None,
                    ball_y=self.last_ball_position[1] if self.last_ball_position else None,
                    confidence=0.35,
                )
            return PossessionSample(
                timestamp_ms=frame.timestamp_ms,
                state=PlayState.UNKNOWN,
                team_key=None,
                player_key=None,
                ball_x=None,
                ball_y=None,
                confidence=0.15,
            )

        ball_position = self._position(ball)
        if ball_position is None:
            return PossessionSample(
                frame.timestamp_ms,
                PlayState.UNKNOWN,
                None,
                None,
                None,
                None,
                0.1,
            )
        self.last_ball_timestamp = frame.timestamp_ms
        self.last_ball_position = ball_position

        if frame.coordinate_space == "pitch_meters" and not (
            -0.5 <= ball_position[0] <= 105.5 and -0.5 <= ball_position[1] <= 68.5
        ):
            return PossessionSample(
                frame.timestamp_ms,
                PlayState.OUT,
                None,
                None,
                ball_position[0],
                ball_position[1],
                0.76,
                {"reason": "pitch_boundary", "coordinate_space": frame.coordinate_space},
            )

        distances = []
        for athlete in frame.athletes:
            position = self._position(athlete)
            if position is None:
                continue
            distances.append((self._distance(ball_position, position), athlete))
        distances.sort(key=lambda pair: pair[0])
        radius = (
            self.pitch_control_radius_m
            if frame.coordinate_space == "pitch_meters"
            else self.image_control_radius
        )
        if not distances or distances[0][0] > radius:
            return PossessionSample(
                frame.timestamp_ms,
                PlayState.LOOSE,
                None,
                None,
                ball_position[0],
                ball_position[1],
                0.5,
            )

        closest_distance, closest = distances[0]
        if len(distances) > 1:
            second_distance, second = distances[1]
            if (
                second.team_key
                and closest.team_key
                and second.team_key != closest.team_key
                and second_distance <= radius * 1.15
                and second_distance - closest_distance <= radius * 0.25
            ):
                return PossessionSample(
                    frame.timestamp_ms,
                    PlayState.CONTESTED,
                    None,
                    None,
                    ball_position[0],
                    ball_position[1],
                    0.58,
                    {
                        "contenders": [
                            {
                                "team_key": closest.team_key,
                                "player_key": closest.player_key or closest.track_id,
                                "distance": closest_distance,
                            },
                            {
                                "team_key": second.team_key,
                                "player_key": second.player_key or second.track_id,
                                "distance": second_distance,
                            },
                        ],
                        "coordinate_space": frame.coordinate_space,
                    },
                )

        proposed = (closest.team_key, closest.player_key or closest.track_id)
        owner = self._debounce_owner(proposed, frame.timestamp_ms)
        confidence = max(0.25, min(0.98, closest.confidence * (1.0 - closest_distance / radius)))
        return PossessionSample(
            frame.timestamp_ms,
            PlayState.CONTROLLED,
            owner[0],
            owner[1],
            ball_position[0],
            ball_position[1],
            confidence,
            {
                "nearest_distance": closest_distance,
                "coordinate_space": frame.coordinate_space,
            },
        )

    def _debounce_owner(
        self,
        proposed: tuple[str | None, str | None],
        timestamp_ms: int,
    ) -> tuple[str | None, str | None]:
        if self.current_owner == (None, None):
            self.current_owner = proposed
            self.owner_history.append(proposed)
            self.pending_owner = (None, None)
            self.pending_since = None
            return self.current_owner
        if proposed == self.current_owner:
            self.pending_owner = (None, None)
            self.pending_since = None
            return self.current_owner
        if proposed != self.pending_owner:
            self.pending_owner = proposed
            self.pending_since = timestamp_ms
            return self.current_owner if self.current_owner != (None, None) else proposed
        if self.pending_since is not None and timestamp_ms - self.pending_since >= self.owner_dwell_ms:
            self.current_owner = proposed
            self.owner_history.append(proposed)
            self.pending_owner = (None, None)
            self.pending_since = None
        return self.current_owner

    @staticmethod
    def _position(obj) -> tuple[float, float] | None:
        if obj.pitch_x is not None and obj.pitch_y is not None:
            return obj.pitch_x, obj.pitch_y
        if obj.image_x is not None and obj.image_y is not None:
            return obj.image_x, obj.image_y
        return None

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def compress(samples: list[PossessionSample], max_gap_ms: int = 1_000) -> list[PossessionSpan]:
        if not samples:
            return []
        deltas = [
            current.timestamp_ms - previous.timestamp_ms
            for previous, current in zip(samples, samples[1:])
            if 0 < current.timestamp_ms - previous.timestamp_ms <= max_gap_ms
        ]
        sample_interval_ms = int(median(deltas)) if deltas else 0
        spans: list[PossessionSpan] = []
        start = samples[0]
        previous = samples[0]
        confidences = [start.confidence]
        for sample in samples[1:]:
            same_state = (
                sample.state == start.state
                and sample.team_key == start.team_key
                and sample.player_key == start.player_key
                and sample.timestamp_ms - previous.timestamp_ms <= max_gap_ms
            )
            if same_state:
                previous = sample
                confidences.append(sample.confidence)
                continue
            spans.append(
                PossessionSpan(
                    start_ms=start.timestamp_ms,
                    end_ms=(
                        sample.timestamp_ms
                        if sample.timestamp_ms - previous.timestamp_ms <= max_gap_ms
                        else previous.timestamp_ms + sample_interval_ms
                    ),
                    state=start.state,
                    team_key=start.team_key,
                    player_key=start.player_key,
                    confidence=sum(confidences) / len(confidences),
                )
            )
            start = previous = sample
            confidences = [sample.confidence]
        spans.append(
            PossessionSpan(
                start_ms=start.timestamp_ms,
                end_ms=previous.timestamp_ms + sample_interval_ms,
                state=start.state,
                team_key=start.team_key,
                player_key=start.player_key,
                confidence=sum(confidences) / len(confidences),
            )
        )
        return spans
