from __future__ import annotations

from collections import defaultdict

from .types import EventCandidate, PlayState, PossessionSpan


def blank_metrics() -> dict[str, float | int]:
    return {
        "possession_pct": 0.0,
        "ball_in_play_seconds": 0.0,
        "possession_seconds": 0.0,
        "passes": 0,
        "passes_completed": 0,
        "pass_accuracy_pct": 0.0,
        "progressive_passes": 0,
        "carries": 0,
        "shots": 0,
        "shots_on_target": 0,
        "goals": 0,
        "assists": 0,
        "duels": 0,
        "duels_won": 0,
        "recoveries": 0,
        "losses": 0,
        "tackles": 0,
        "interceptions": 0,
        "fouls": 0,
        "corners": 0,
        "offsides": 0,
    }


class StatsAggregator:
    def aggregate(
        self,
        events: list[EventCandidate],
        possessions: list[PossessionSpan],
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        team_metrics: dict[str, dict] = defaultdict(blank_metrics)
        player_metrics: dict[str, dict] = defaultdict(blank_metrics)
        controlled_ms: dict[str, int] = defaultdict(int)
        total_controlled_ms = 0
        total_ball_in_play_ms = sum(
            span.duration_ms
            for span in possessions
            if span.state in {PlayState.CONTROLLED, PlayState.CONTESTED, PlayState.LOOSE}
        )
        for span in possessions:
            if span.state != PlayState.CONTROLLED or not span.team_key:
                continue
            duration = span.duration_ms
            total_controlled_ms += duration
            controlled_ms[span.team_key] += duration
        for team_key, duration in controlled_ms.items():
            team_metrics[team_key]["ball_in_play_seconds"] = round(
                total_ball_in_play_ms / 1000.0, 1
            )
            team_metrics[team_key]["possession_seconds"] = round(duration / 1000.0, 1)
            team_metrics[team_key]["possession_pct"] = round(
                100.0 * duration / max(total_controlled_ms, 1),
                2,
            )

        for event in events:
            if event.team_key:
                self._apply_event(team_metrics[event.team_key], event)
            if event.player_key:
                self._apply_event(player_metrics[event.player_key], event)

        for metrics in [*team_metrics.values(), *player_metrics.values()]:
            passes = metrics["passes"]
            metrics["pass_accuracy_pct"] = round(
                100.0 * metrics["passes_completed"] / passes,
                2,
            ) if passes else 0.0
        return dict(team_metrics), dict(player_metrics)

    @staticmethod
    def _apply_event(metrics: dict, event: EventCandidate) -> None:
        successful = event.outcome == "success"
        mapping = {
            "pass": "passes",
            "carry": "carries",
            "shot": "shots",
            "goal": "goals",
            "duel": "duels",
            "aerial_duel": "duels",
            "recovery": "recoveries",
            "loss": "losses",
            "tackle": "tackles",
            "interception": "interceptions",
            "foul": "fouls",
            "corner": "corners",
            "offside": "offsides",
        }
        key = mapping.get(event.event_type)
        if key:
            metrics[key] += 1
        if event.event_type == "pass" and successful:
            metrics["passes_completed"] += 1
        if event.event_type == "pass" and "progressive_candidate" in event.qualifiers:
            metrics["progressive_passes"] += 1
        if event.event_type in {"duel", "aerial_duel"} and successful:
            metrics["duels_won"] += 1
        if event.event_type == "shot" and "on_target" in event.qualifiers:
            metrics["shots_on_target"] += 1
        if event.event_type == "pass" and "assist" in event.qualifiers:
            metrics["assists"] += 1
