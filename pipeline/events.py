from __future__ import annotations

import math

from .types import EventCandidate, PlayState, PossessionSample


class EventEngine:
    """Turn stable possession changes into conservative event candidates."""

    def __init__(self, max_transfer_gap_ms: int = 3_000, carry_distance: float = 0.08):
        self.max_transfer_gap_ms = max_transfer_gap_ms
        self.carry_distance = carry_distance

    def detect(self, samples: list[PossessionSample]) -> list[EventCandidate]:
        events: list[EventCandidate] = []
        controlled_runs = self._controlled_runs(samples)
        for index, current in enumerate(controlled_runs):
            self._append_carry(current, events)
            if index == 0:
                continue
            previous = controlled_runs[index - 1]
            gap_ms = current[0].timestamp_ms - previous[-1].timestamp_ms
            if gap_ms > self.max_transfer_gap_ms:
                continue
            previous_owner = (previous[-1].team_key, previous[-1].player_key)
            current_owner = (current[0].team_key, current[0].player_key)
            if previous_owner == current_owner:
                continue
            start = previous[-1]
            end = current[0]
            confidence = min(start.confidence, end.confidence)
            if start.team_key and start.team_key == end.team_key:
                qualifiers = []
                if self._progress(start, end) >= self._progressive_threshold(start):
                    qualifiers.append("progressive_candidate")
                events.append(
                    EventCandidate(
                        timestamp_ms=end.timestamp_ms,
                        event_type="pass",
                        team_key=start.team_key,
                        player_key=start.player_key,
                        recipient_key=end.player_key,
                        outcome="success",
                        start_x=start.ball_x,
                        start_y=start.ball_y,
                        end_x=end.ball_x,
                        end_y=end.ball_y,
                        confidence=confidence,
                        qualifiers=qualifiers,
                    )
                )
            elif start.team_key and end.team_key and start.team_key != end.team_key:
                events.extend(
                    [
                        EventCandidate(
                            timestamp_ms=end.timestamp_ms,
                            event_type="loss",
                            team_key=start.team_key,
                            player_key=start.player_key,
                            outcome="failure",
                            start_x=start.ball_x,
                            start_y=start.ball_y,
                            end_x=end.ball_x,
                            end_y=end.ball_y,
                            confidence=confidence * 0.82,
                        ),
                        EventCandidate(
                            timestamp_ms=end.timestamp_ms,
                            event_type="recovery",
                            team_key=end.team_key,
                            player_key=end.player_key,
                            outcome="success",
                            start_x=end.ball_x,
                            start_y=end.ball_y,
                            confidence=confidence * 0.82,
                        ),
                    ]
                )
        self._append_duels(samples, events)
        self._append_outs(samples, events)
        self._append_shot_candidates(samples, events)
        self._mark_assists(events)
        return self._deduplicate(events)

    def _append_duels(
        self,
        samples: list[PossessionSample],
        events: list[EventCandidate],
    ) -> None:
        index = 0
        while index < len(samples):
            if samples[index].state != PlayState.CONTESTED:
                index += 1
                continue
            start = index
            while index + 1 < len(samples) and samples[index + 1].state == PlayState.CONTESTED:
                index += 1
            end = index
            previous = next(
                (sample for sample in reversed(samples[:start]) if sample.state == PlayState.CONTROLLED),
                None,
            )
            following = next(
                (sample for sample in samples[end + 1 :] if sample.state == PlayState.CONTROLLED),
                None,
            )
            contested = samples[start : end + 1]
            contenders = [
                contender
                for sample in contested
                for contender in sample.metadata.get("contenders", [])
            ]
            winner_key = following.player_key if following else None
            winner_team = following.team_key if following else None
            if not winner_key and contenders:
                winner_key = contenders[0].get("player_key")
                winner_team = contenders[0].get("team_key")
            if winner_key:
                opponent = next(
                    (
                        item.get("player_key")
                        for item in contenders
                        if item.get("player_key") != winner_key
                    ),
                    None,
                )
                confidence = sum(sample.confidence for sample in contested) / len(contested)
                qualifiers = ["ground_candidate"]
                if opponent:
                    qualifiers.append(f"opponent_track:{opponent}")
                events.append(
                    EventCandidate(
                        timestamp_ms=contested[0].timestamp_ms,
                        event_type="duel",
                        team_key=winner_team,
                        player_key=winner_key,
                        outcome="success" if following else "unknown",
                        start_x=contested[0].ball_x,
                        start_y=contested[0].ball_y,
                        end_x=contested[-1].ball_x,
                        end_y=contested[-1].ball_y,
                        confidence=confidence * 0.78,
                        qualifiers=qualifiers,
                    )
                )
                if previous and following and previous.player_key == following.player_key:
                    events.append(
                        EventCandidate(
                            timestamp_ms=following.timestamp_ms,
                            event_type="dribble",
                            team_key=following.team_key,
                            player_key=following.player_key,
                            outcome="success",
                            start_x=previous.ball_x,
                            start_y=previous.ball_y,
                            end_x=following.ball_x,
                            end_y=following.ball_y,
                            confidence=min(previous.confidence, following.confidence) * 0.75,
                            qualifiers=["after_contest"],
                        )
                    )
            index += 1

    @staticmethod
    def _append_outs(samples: list[PossessionSample], events: list[EventCandidate]) -> None:
        previous_state = None
        for sample in samples:
            if (
                sample.state == PlayState.OUT
                and previous_state not in {None, PlayState.OUT}
                and sample.metadata.get("reason") != "broadcast_not_live"
            ):
                events.append(
                    EventCandidate(
                        timestamp_ms=sample.timestamp_ms,
                        event_type="out",
                        outcome="neutral",
                        confidence=sample.confidence * 0.72,
                    )
                )
            previous_state = sample.state

    def _append_shot_candidates(
        self,
        samples: list[PossessionSample],
        events: list[EventCandidate],
    ) -> None:
        for index, sample in enumerate(samples[:-1]):
            if sample.state != PlayState.CONTROLLED or not sample.player_key:
                continue
            trajectory = [sample]
            for following in samples[index + 1 :]:
                if following.timestamp_ms - sample.timestamp_ms > 2_200:
                    break
                if (
                    following.state == PlayState.CONTROLLED
                    and following.player_key != sample.player_key
                ):
                    break
                if following.ball_x is not None and following.ball_y is not None:
                    trajectory.append(following)
            if len(trajectory) < 3:
                continue
            end = trajectory[-1]
            if end.state not in {PlayState.LOOSE, PlayState.OUT}:
                continue
            if None in {sample.ball_x, sample.ball_y, end.ball_x, end.ball_y}:
                continue
            space = sample.metadata.get("coordinate_space", "image_normalized")
            displacement = math.hypot(end.ball_x - sample.ball_x, end.ball_y - sample.ball_y)
            near_goal = (
                end.ball_x <= 10 or end.ball_x >= 95
                if space == "pitch_meters"
                else end.ball_x <= 0.12 or end.ball_x >= 0.88
            )
            threshold = 8.0 if space == "pitch_meters" else 0.12
            if near_goal and displacement >= threshold:
                events.append(
                    EventCandidate(
                        timestamp_ms=sample.timestamp_ms,
                        event_type="shot",
                        team_key=sample.team_key,
                        player_key=sample.player_key,
                        outcome="unknown",
                        start_x=sample.ball_x,
                        start_y=sample.ball_y,
                        end_x=end.ball_x,
                        end_y=end.ball_y,
                        confidence=min(sample.confidence, end.confidence) * 0.68,
                        qualifiers=["trajectory_candidate", f"space:{space}"],
                    )
                )

    @staticmethod
    def _mark_assists(events: list[EventCandidate]) -> None:
        goals = [event for event in events if event.event_type == "goal"]
        for goal in goals:
            candidate = next(
                (
                    event
                    for event in reversed(events)
                    if event.event_type == "pass"
                    and event.team_key == goal.team_key
                    and event.recipient_key == goal.player_key
                    and 0 <= goal.timestamp_ms - event.timestamp_ms <= 12_000
                ),
                None,
            )
            if candidate and "assist" not in candidate.qualifiers:
                candidate.qualifiers.append("assist")

    def _controlled_runs(self, samples: list[PossessionSample]) -> list[list[PossessionSample]]:
        runs: list[list[PossessionSample]] = []
        current: list[PossessionSample] = []
        current_owner = None
        previous_timestamp: int | None = None
        for sample in samples:
            if sample.state != PlayState.CONTROLLED or not sample.player_key:
                if current:
                    runs.append(current)
                    current = []
                    current_owner = None
                previous_timestamp = sample.timestamp_ms
                continue
            owner = (sample.team_key, sample.player_key)
            discontinuity = (
                previous_timestamp is not None
                and sample.timestamp_ms - previous_timestamp > self.max_transfer_gap_ms
            )
            if current and (owner != current_owner or discontinuity):
                runs.append(current)
                current = []
            current.append(sample)
            current_owner = owner
            previous_timestamp = sample.timestamp_ms
        if current:
            runs.append(current)
        return [run for run in runs if run]

    def _append_carry(
        self,
        run: list[PossessionSample],
        events: list[EventCandidate],
    ) -> None:
        if len(run) < 2:
            return
        start, end = run[0], run[-1]
        if None in {start.ball_x, start.ball_y, end.ball_x, end.ball_y}:
            return
        distance = math.hypot(end.ball_x - start.ball_x, end.ball_y - start.ball_y)
        duration = end.timestamp_ms - start.timestamp_ms
        distance_threshold = (
            5.0
            if start.metadata.get("coordinate_space") == "pitch_meters"
            else self.carry_distance
        )
        if distance < distance_threshold or duration < 700:
            return
        events.append(
            EventCandidate(
                timestamp_ms=end.timestamp_ms,
                event_type="carry",
                team_key=end.team_key,
                player_key=end.player_key,
                outcome="success",
                start_x=start.ball_x,
                start_y=start.ball_y,
                end_x=end.ball_x,
                end_y=end.ball_y,
                confidence=min(sample.confidence for sample in run),
                qualifiers=["progressive_candidate"]
                if abs(end.ball_x - start.ball_x) >= self._progressive_threshold(start)
                else [],
                metadata={"duration_ms": duration, "distance": round(distance, 5)},
            )
        )

    @staticmethod
    def _progress(start: PossessionSample, end: PossessionSample) -> float:
        if start.ball_x is None or end.ball_x is None:
            return 0.0
        return abs(end.ball_x - start.ball_x)

    @staticmethod
    def _progressive_threshold(sample: PossessionSample) -> float:
        return 10.0 if sample.metadata.get("coordinate_space") == "pitch_meters" else 0.12

    @staticmethod
    def _deduplicate(events: list[EventCandidate]) -> list[EventCandidate]:
        result: list[EventCandidate] = []
        for event in sorted(events, key=lambda item: item.timestamp_ms):
            duplicate = next(
                (
                    existing
                    for existing in reversed(result[-6:])
                    if existing.event_type == event.event_type
                    and existing.player_key == event.player_key
                    and abs(existing.timestamp_ms - event.timestamp_ms) <= 400
                ),
                None,
            )
            if duplicate is None:
                result.append(event)
            elif event.confidence > duplicate.confidence:
                result[result.index(duplicate)] = event
        return result
