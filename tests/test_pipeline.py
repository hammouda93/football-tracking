from __future__ import annotations

import sys
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.ball_in_play import BallInPlayEngine
from pipeline.events import EventEngine
from pipeline.periods import PeriodDetector
from pipeline.providers.yolo import YoloVisionProvider
from pipeline.stats import StatsAggregator
from pipeline.runner import MatchAnalysisRunner
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
from pipeline.video import iter_frames, iter_sampled_frames, sample_timestamps


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


class VideoSamplingTests(unittest.TestCase):
    def test_quality_sampling_is_bounded_and_spread_over_full_match(self):
        timestamps = sample_timestamps(
            start_ms=0,
            end_ms=5_400_000,
            interval_ms=15_000,
            max_samples=360,
        )

        self.assertEqual(len(timestamps), 360)
        self.assertEqual(timestamps[:3], [0, 15_000, 30_000])
        self.assertEqual(timestamps[-1], 5_385_000)

    def test_quality_sampling_never_exceeds_limit(self):
        timestamps = sample_timestamps(
            start_ms=0,
            end_ms=7_200_000,
            interval_ms=1_000,
            max_samples=360,
        )

        self.assertEqual(len(timestamps), 360)
        self.assertGreater(timestamps[-1], 7_100_000)

    def test_sparse_reader_decodes_only_requested_frames(self):
        class FakeCapture:
            def __init__(self):
                self.read_count = 0
                self.seek_values = []
                self.released = False

            def isOpened(self):
                return True

            def set(self, _property, value):
                self.seek_values.append(value)
                return True

            def read(self):
                self.read_count += 1
                return True, object()

            def release(self):
                self.released = True

        capture = FakeCapture()
        timestamps = [0, 15_000, 30_000, 45_000]
        fake_cv2 = SimpleNamespace(
            CAP_PROP_POS_MSEC=0,
            VideoCapture=lambda _path: capture,
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            frames = list(iter_sampled_frames("match.mp4", timestamps))

        self.assertEqual([timestamp for timestamp, _ in frames], timestamps)
        self.assertEqual(capture.read_count, len(timestamps))
        self.assertEqual(capture.seek_values, timestamps)
        self.assertTrue(capture.released)

    def test_sequential_reader_falls_back_to_frame_clock_when_msec_is_stuck(self):
        class FakeCapture:
            def __init__(self):
                self.position = 0

            def isOpened(self):
                return True

            def set(self, _property, _value):
                return True

            def get(self, property_id):
                if property_id == 1:
                    return 25.0
                if property_id == 2:
                    return 0.0
                if property_id == 3:
                    return float(self.position)
                return 0.0

            def read(self):
                if self.position >= 26:
                    return False, None
                self.position += 1
                return True, object()

            def release(self):
                return None

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_POS_MSEC=2,
            CAP_PROP_POS_FRAMES=3,
            VideoCapture=lambda _path: FakeCapture(),
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            frames = list(iter_frames("match.mp4", target_fps=5.0))

        timestamps = [timestamp for timestamp, _ in frames]
        self.assertGreater(len(timestamps), 1)
        self.assertEqual(timestamps[:3], [0, 200, 400])

    def test_tracking_label_contains_stage_video_frames_speed_and_eta(self):
        label = MatchAnalysisRunner._tracking_label(
            backend="yolo",
            device="cpu",
            stage_progress=12.5,
            processed_ms=675_000,
            total_ms=5_408_000,
            frames_processed=6_750,
            frames_total=54_080,
            speed_x=0.25,
            eta_seconds=18_932,
        )

        self.assertIn("Tracking 12.5%", label)
        self.assertIn("vidéo 11:15/1h30", label)
        self.assertIn("6 750/54 080 images", label)
        self.assertIn("0.25×", label)
        self.assertIn("reste 5h15", label)
        self.assertIn("YOLO CPU", label)

    def test_yolo_tracking_fps_has_a_temporal_continuity_floor(self):
        effective = MatchAnalysisRunner._effective_tracking_fps(
            backend="yolo",
            requested_fps=2.0,
            native_fps=25.0,
            minimum_yolo_fps=8.0,
        )

        self.assertEqual(effective, 8.0)

    def test_heuristic_tracking_fps_is_not_forced_up(self):
        effective = MatchAnalysisRunner._effective_tracking_fps(
            backend="heuristic",
            requested_fps=2.0,
            native_fps=25.0,
            minimum_yolo_fps=8.0,
        )

        self.assertEqual(effective, 2.0)

    def test_explicit_model_class_ids_override_unknown_names(self):
        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.class_roles = {
            0: ObjectRole.BALL,
            2: ObjectRole.PLAYER,
            3: ObjectRole.REFEREE,
        }

        self.assertEqual(provider._role_for(2, {2: "athlete"}), ObjectRole.PLAYER)
        self.assertEqual(provider._role_for(0, {0: "tiny-object"}), ObjectRole.BALL)

    def test_sample_uses_two_thirty_second_windows_per_half(self):
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {
            "analysis_mode": "sample",
            "sample_window_seconds": 30,
            "sample_windows_per_half": 2,
        }
        periods = [
            SimpleNamespace(number=1, video_start_ms=0, video_end_ms=2_700_000),
            SimpleNamespace(number=2, video_start_ms=3_300_000, video_end_ms=6_000_000),
        ]

        windows = runner._tracking_windows(periods)

        self.assertEqual(len(windows), 4)
        self.assertEqual(sum(item["end_ms"] - item["start_ms"] for item in windows), 120_000)
        self.assertEqual([item["period"].number for item in windows], [1, 1, 2, 2])

    def test_diagnostics_fail_bad_ball_team_and_track_detection(self):
        diagnostics = MatchAnalysisRunner._tracking_diagnostics(
            Counter(
                {
                    "frames": 1_200,
                    "raw_athlete_detections": 14_400,
                    "athlete_observations": 14_400,
                    "ball_visible_frames": 24,
                    "field_frames": 1_100,
                    "state_unknown": 1_176,
                    "state_loose": 24,
                }
            ),
            Counter({"home": 13_900, "away": 500}),
            track_count=300,
            tracking_duration_ms=120_000,
        )

        self.assertEqual(diagnostics["verdict"], "fail")
        self.assertLess(diagnostics["ball_visibility_pct"], 10)
        self.assertGreater(diagnostics["tracks_per_minute"], 80)

    def test_diagnostics_distinguish_yolo_recall_from_tracker_retention(self):
        diagnostics = MatchAnalysisRunner._tracking_diagnostics(
            Counter(
                {
                    "frames": 100,
                    "raw_athlete_detections": 1_100,
                    "athlete_observations": 450,
                    "ball_visible_frames": 40,
                    "field_frames": 100,
                    "state_controlled": 40,
                }
            ),
            Counter({"home": 250, "away": 200}),
            track_count=20,
            tracking_duration_ms=120_000,
        )

        self.assertEqual(diagnostics["average_player_detections_per_frame"], 11.0)
        self.assertEqual(diagnostics["average_tracked_athletes_per_frame"], 4.5)
        self.assertIn("ByteTrack", " ".join(diagnostics["issues"]))


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
