from __future__ import annotations

import unittest

from pipeline.ball_in_play import BallInPlayEngine
from pipeline.events import EventEngine
from pipeline.periods import PeriodDetector
from pipeline.stats import StatsAggregator
from pipeline.types import (
    EventCandidate,
    FrameAnalysis,
    FrameSignal,
    ObjectRole,
    PlayState,
    PossessionSample,
    PossessionSpan,
    TrackedObject,
)


def sample(timestamp, state, team=None, player=None, x=None, y=None, confidence=0.9, **metadata):
    return PossessionSample(timestamp, state, team, player, x, y, confidence, metadata)


class PeriodDetectorTests(unittest.TestCase):
    def test_two_field_blocks_become_two_halves(self):
        signals = []
        for minute in range(0, 111):
            active = minute <= 47 or 63 <= minute <= 110
            signals.append(
                FrameSignal(
                    timestamp_ms=minute * 60_000,
                    field_score=0.64 if active else 0.03,
                    sharpness=150,
                    brightness=125,
                )
            )
        result = PeriodDetector(bridge_gap_ms=120_000).detect(signals, 111 * 60_000)
        self.assertEqual(len(result.periods), 2)
        self.assertLess(result.periods[0].end_ms, result.periods[1].start_ms)
        self.assertGreater(result.diagnostics["halftime_gap_ms"], 8 * 60_000)

    def test_short_or_unknown_video_has_reviewable_fallback(self):
        result = PeriodDetector().detect([], 100 * 60_000)
        self.assertTrue(result.requires_review)
        self.assertEqual(len(result.periods), 2)


class BallInPlayTests(unittest.TestCase):
    def test_nearest_player_controls_ball(self):
        frame = FrameAnalysis(
            timestamp_ms=1_000,
            width=1920,
            height=1080,
            field_score=0.8,
            objects=[
                TrackedObject("ball", str(ObjectRole.BALL), (0, 0, 1, 1), 0.95, image_x=0.5, image_y=0.5),
                TrackedObject("p1", str(ObjectRole.PLAYER), (0, 0, 1, 1), 0.9, team_key="home", player_key="p1", image_x=0.515, image_y=0.5),
            ],
        )
        possession = BallInPlayEngine().observe(frame)
        self.assertEqual(possession.state, PlayState.CONTROLLED)
        self.assertEqual(possession.team_key, "home")
        self.assertEqual(possession.player_key, "p1")

    def test_close_opponents_make_a_contested_ball(self):
        frame = FrameAnalysis(
            timestamp_ms=1_000,
            width=1920,
            height=1080,
            field_score=0.8,
            objects=[
                TrackedObject("ball", str(ObjectRole.BALL), (0, 0, 1, 1), 0.95, image_x=0.5, image_y=0.5),
                TrackedObject("p1", str(ObjectRole.PLAYER), (0, 0, 1, 1), 0.9, team_key="home", player_key="p1", image_x=0.51, image_y=0.5),
                TrackedObject("p2", str(ObjectRole.PLAYER), (0, 0, 1, 1), 0.9, team_key="away", player_key="p2", image_x=0.512, image_y=0.5),
            ],
        )
        possession = BallInPlayEngine().observe(frame)
        self.assertEqual(possession.state, PlayState.CONTESTED)
        self.assertEqual(len(possession.metadata["contenders"]), 2)


class EventEngineTests(unittest.TestCase):
    def test_same_team_owner_change_is_a_pass(self):
        samples = [
            sample(0, PlayState.CONTROLLED, "home", "p1", 0.25, 0.5),
            sample(500, PlayState.CONTROLLED, "home", "p1", 0.30, 0.5),
            sample(900, PlayState.CONTROLLED, "home", "p2", 0.48, 0.5),
            sample(1_400, PlayState.CONTROLLED, "home", "p2", 0.51, 0.5),
        ]
        events = EventEngine().detect(samples)
        passes = [event for event in events if event.event_type == "pass"]
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0].player_key, "p1")
        self.assertEqual(passes[0].recipient_key, "p2")

    def test_team_change_creates_loss_and_recovery(self):
        samples = [
            sample(0, PlayState.CONTROLLED, "home", "p1", 0.50, 0.5),
            sample(500, PlayState.CONTROLLED, "away", "p2", 0.51, 0.5),
        ]
        types = {event.event_type for event in EventEngine().detect(samples)}
        self.assertEqual(types, {"loss", "recovery"})

    def test_contested_sequence_creates_duel_and_dribble(self):
        contenders = [
            {"team_key": "home", "player_key": "p1", "distance": 0.01},
            {"team_key": "away", "player_key": "p2", "distance": 0.012},
        ]
        samples = [
            sample(0, PlayState.CONTROLLED, "home", "p1", 0.5, 0.5),
            sample(300, PlayState.CONTESTED, x=0.51, y=0.5, contenders=contenders),
            sample(700, PlayState.CONTROLLED, "home", "p1", 0.53, 0.5),
        ]
        types = {event.event_type for event in EventEngine().detect(samples)}
        self.assertIn("duel", types)
        self.assertIn("dribble", types)

    def test_fast_trajectory_near_goal_is_reviewable_shot(self):
        samples = [
            sample(0, PlayState.CONTROLLED, "home", "p9", 0.70, 0.45, coordinate_space="image_normalized"),
            sample(400, PlayState.CONTROLLED, "home", "p9", 0.78, 0.46, coordinate_space="image_normalized"),
            sample(800, PlayState.LOOSE, x=0.91, y=0.47, coordinate_space="image_normalized"),
        ]
        shots = [event for event in EventEngine().detect(samples) if event.event_type == "shot"]
        self.assertEqual(len(shots), 1)
        self.assertLess(shots[0].confidence, 0.92)


class StatsTests(unittest.TestCase):
    def test_possession_and_pass_accuracy(self):
        events = [
            EventCandidate(1_000, "pass", "home", "p1", outcome="success"),
            EventCandidate(2_000, "pass", "home", "p1", outcome="failure"),
        ]
        spans = [
            PossessionSpan(0, 6_000, PlayState.CONTROLLED, "home", "p1", 0.9),
            PossessionSpan(6_000, 10_000, PlayState.CONTROLLED, "away", "p2", 0.9),
        ]
        teams, players = StatsAggregator().aggregate(events, spans)
        self.assertEqual(teams["home"]["possession_pct"], 60.0)
        self.assertEqual(players["p1"]["pass_accuracy_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
