from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from pipeline.ball_in_play import BallInPlayEngine
from pipeline.events import EventEngine
from pipeline.evaluation import evaluate_tracking, validate_ground_truth
from pipeline.jersey import JerseyNumberRecognizer, JerseyObservation, WeightedNumberVote
from pipeline.periods import PeriodDetector
from pipeline.pitch import NativePitchCalibrator, PitchLandmark, load_pitch_landmarks
from pipeline.providers.yolo import YoloVisionProvider
from pipeline.providers.native_gsr import NativeGSRVisionProvider
from pipeline.providers.native_identity import NativeAppearanceEncoder, NativeIdentityRefiner
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

    def test_short_edited_halftime_break_is_detected_near_video_midpoint(self):
        signals = []
        duration_ms = 100 * 60_000
        for second in range(0, 6_001, 10):
            timestamp_ms = second * 1_000
            in_halftime = 2_880_000 < timestamp_ms < 2_940_000
            signals.append(
                FrameSignal(
                    timestamp_ms=timestamp_ms,
                    field_score=0.03 if in_halftime else 0.64,
                    sharpness=150,
                    brightness=125,
                )
            )

        result = PeriodDetector().detect(signals, duration_ms)

        self.assertEqual(result.diagnostics["detection_method"], "central_broadcast_break")
        self.assertEqual(result.periods[0].end_ms, 2_880_000)
        self.assertEqual(result.periods[1].start_ms, 2_940_000)
        self.assertTrue(result.requires_review)


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

    def test_sequential_reader_does_not_accumulate_native_frame_rounding(self):
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
            frames = list(iter_frames("match.mp4", target_fps=8.0))

        timestamps = [timestamp for timestamp, _ in frames]
        self.assertEqual(timestamps, [0, 160, 280, 400, 520, 640, 760, 880, 1_000])

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

    def test_botsort_profile_enables_camera_motion_compensation(self):
        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.confidence = 0.30
        provider.tracker_low_confidence = 0.10
        provider.tracker_new_confidence = 0.35
        provider.tracker_match_threshold = 0.85
        provider.tracker_buffer_seconds = 5.0
        provider.tracking_fps = 8.0

        args = provider._botsort_args()

        self.assertEqual(args.tracker_type, "botsort")
        self.assertEqual(args.gmc_method, "sparseOptFlow")
        self.assertEqual(args.track_low_thresh, 0.10)
        self.assertEqual(args.new_track_thresh, 0.35)
        self.assertEqual(args.track_buffer, 40)

    def test_botsort_rows_are_converted_to_provider_tracks(self):
        class FakeBoxes:
            def __getitem__(self, _indices):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self

        class FakeTracker:
            def update(self, boxes, frame):
                self.received = (boxes, frame)
                return [[10, 20, 30, 60, 17, 0.91, 2, 0]]

        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.tracker_name = "botsort"
        provider.tracker = FakeTracker()
        prediction = SimpleNamespace(boxes=FakeBoxes())
        frame = object()

        tracks = provider._update_tracker(prediction, None, [0], frame)

        self.assertEqual(tracks, [([10, 20, 30, 60], 0.91, 2, 17)])
        self.assertEqual(provider.tracker.received, (prediction.boxes, frame))

    def test_botsort_constructor_supports_old_and_new_ultralytics(self):
        class LegacyBOTSORT:
            def __init__(self, args, frame_rate=30):
                self.args = args
                self.frame_rate = frame_rate

        class CurrentBOTSORT:
            def __init__(self, args):
                self.args = args

        args = object()

        legacy = YoloVisionProvider._instantiate_botsort(LegacyBOTSORT, args)
        current = YoloVisionProvider._instantiate_botsort(CurrentBOTSORT, args)

        self.assertIs(legacy.args, args)
        self.assertEqual(legacy.frame_rate, 30)
        self.assertIs(current.args, args)

    def test_overlapping_person_detections_keep_the_most_confident_box(self):
        boxes = [
            (100, 100, 150, 250),
            (101, 101, 151, 251),
            (300, 100, 350, 250),
            (110, 150, 130, 220),
        ]

        kept = YoloVisionProvider._deduplicate_indices(
            boxes,
            [0.72, 0.91, 0.80, 0.85],
            [0, 1, 2, 3],
        )

        self.assertEqual(kept, [1, 2])

    def test_overlapping_tracker_rows_are_not_drawn_twice(self):
        first = TrackedObject(
            "athlete-1", "player", (100, 100, 150, 250), 0.71
        )
        duplicate = TrackedObject(
            "athlete-2", "player", (101, 101, 151, 251), 0.92
        )
        separate = TrackedObject(
            "athlete-3", "goalkeeper", (300, 100, 350, 250), 0.84
        )

        kept = YoloVisionProvider._deduplicate_tracked_objects(
            [first, duplicate, separate]
        )

        self.assertEqual([item.track_id for item in kept], ["athlete-2", "athlete-3"])

    def test_ball_selection_rejects_isolated_false_positive(self):
        import numpy as np

        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.previous_ball_center = None
        provider.previous_ball_timestamp_ms = None
        provider.ball_confidence = 0.12
        frame = np.full((720, 1280, 3), (45, 145, 45), dtype=np.uint8)
        player = TrackedObject(
            "athlete-1", "player", (100, 100, 150, 250), 0.90
        )
        near_player = ((125, 245, 135, 255), 0.42)
        isolated = ((900, 500, 910, 510), 0.95)

        selected = provider._select_ball(
            [isolated, near_player],
            [player],
            frame,
            10_000,
        )

        self.assertEqual(selected, near_player)
        self.assertIn("near_player", provider.last_ball_selection_reason)

    def test_ball_selection_accepts_unique_loose_ball_on_the_pitch(self):
        import numpy as np

        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.previous_ball_center = None
        provider.previous_ball_timestamp_ms = None
        provider.ball_confidence = 0.12
        frame = np.full((720, 1280, 3), (45, 145, 45), dtype=np.uint8)
        loose_ball = ((620, 330, 629, 339), 0.24)

        selected = provider._select_ball([loose_ball], [], frame, 10_000)

        self.assertEqual(selected, loose_ball)
        self.assertEqual(provider.last_ball_selection_reason, "field_only")
        self.assertGreater(provider.last_ball_field_support, 0.80)

    def test_ball_selection_rejects_loose_candidate_away_from_pitch(self):
        import numpy as np

        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.previous_ball_center = None
        provider.previous_ball_timestamp_ms = None
        provider.ball_confidence = 0.12
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        logo_candidate = ((620, 330, 629, 339), 0.92)

        selected = provider._select_ball([logo_candidate], [], frame, 10_000)

        self.assertIsNone(selected)

    def test_ball_geometry_rejects_large_field_mark(self):
        self.assertFalse(
            YoloVisionProvider._valid_ball_geometry(
                (100, 100, 140, 140),
                (720, 1280, 3),
            )
        )
        self.assertTrue(
            YoloVisionProvider._valid_ball_geometry(
                (100, 100, 110, 110),
                (720, 1280, 3),
            )
        )

    def test_team_codes_use_club_initials(self):
        self.assertEqual(
            MatchAnalysisRunner._team_code(
                SimpleNamespace(name="Stade Tunisien", short_name="STA")
            ),
            "ST",
        )
        self.assertEqual(
            MatchAnalysisRunner._team_code(
                SimpleNamespace(name="Club Sportif Sfaxien", short_name="CSS")
            ),
            "CSS",
        )

    def test_reference_and_live_previews_are_rendered(self):
        import cv2
        import numpy as np

        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        analysis = FrameAnalysis(
            timestamp_ms=12_000,
            width=320,
            height=180,
            field_score=0.8,
            objects=[
                TrackedObject(
                    "athlete-7",
                    "player",
                    (80, 40, 105, 120),
                    0.9,
                    team_key="home",
                )
            ],
            diagnostics={
                "raw_detections": [
                    {
                        "bbox": [80, 40, 105, 120],
                        "role": "player",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = Path(directory) / "comparison.jpg"
            live_path = Path(directory) / "live.jpg"

            MatchAnalysisRunner._write_sample_preview(
                frame,
                analysis,
                comparison_path,
                team_labels={"home": "HOM (T1)", "away": "AWY (T2)"},
            )
            MatchAnalysisRunner._write_live_preview(
                frame,
                analysis,
                live_path,
                team_labels={"home": "HOM (T1)", "away": "AWY (T2)"},
            )

            comparison = cv2.imread(str(comparison_path))
            live = cv2.imread(str(live_path))

        self.assertEqual(comparison.shape[0], 180)
        self.assertGreater(comparison.shape[1], 640)
        self.assertEqual(live.shape[:2], (180, 320))

    def test_team_label_uses_the_track_majority_instead_of_one_frame(self):
        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.team_votes = {}

        for _ in range(8):
            result = provider._stabilize_team(17, "home")
        for _ in range(2):
            result = provider._stabilize_team(17, "away")

        self.assertEqual(result, "home")

    def test_team_colors_are_learned_from_video_without_imported_club_colors(self):
        import cv2
        import numpy as np

        provider = YoloVisionProvider.__new__(YoloVisionProvider)
        provider.home_team_cluster = "B"
        provider.team_color_samples = []
        provider.team_cluster_centers = None
        provider.team_cluster_mapping = {}
        provider.team_calibration_fits = 0
        provider.team_calibration_last_fit = 0
        provider.team_calibration_mapping_margin = 0.0

        frame = np.full((180, 320, 3), (50, 150, 50), dtype=np.uint8)
        cv2.rectangle(frame, (40, 25), (90, 145), (35, 35, 180), -1)
        cv2.rectangle(frame, (210, 25), (260, 145), (245, 245, 245), -1)
        home_box = (40, 25, 90, 145)
        away_box = (210, 25, 260, 145)
        for _ in range(8):
            provider._classify_team(frame, home_box)
            provider._classify_team(frame, away_box)

        self.assertEqual(provider._classify_team(frame, home_box), "home")
        self.assertEqual(provider._classify_team(frame, away_box), "away")
        diagnostics = provider.team_calibration_diagnostics()
        self.assertEqual(diagnostics["status"], "ready")
        self.assertEqual(diagnostics["source"], "video_only")
        self.assertEqual(diagnostics["home_group"], "B")

    def test_native_gsr_removes_duplicate_person_boxes(self):
        import cv2
        import numpy as np

        frame = np.full((180, 320, 3), (45, 145, 45), dtype=np.uint8)
        cv2.rectangle(frame, (80, 25), (120, 150), (30, 30, 190), -1)
        objects = [
            TrackedObject("raw-1", "player", (80, 25, 120, 150), 0.91),
            TrackedObject("raw-2", "player", (82, 30, 122, 150), 0.72),
        ]
        refiner = NativeIdentityRefiner(home_team_cluster="B")

        refined = refiner.process(frame, objects, 1_000)

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].confidence, 0.91)
        self.assertEqual(refiner.diagnostics()["duplicates_removed"], 1)

    def test_native_gsr_rejects_tracker_id_after_strong_team_switch(self):
        import numpy as np

        refiner = NativeIdentityRefiner(home_team_cluster="A")
        first = TrackedObject("raw-1", "player", (80, 25, 120, 150), 0.91)
        state = refiner._new_state(first, np.asarray([1.0, 0.0]), 1_000)
        state.team_label = "home"
        refiner.raw_to_canonical["raw-1"] = state.canonical_id
        refiner.team_centers = np.asarray(
            [
                np.zeros(8, dtype=np.float32),
                np.ones(8, dtype=np.float32),
            ]
        )
        switched = TrackedObject("raw-1", "player", (82, 25, 122, 150), 0.90)

        assignments = refiner._link_candidates(
            [switched],
            [np.asarray([1.0, 0.0])],
            [np.ones(8, dtype=np.float32)],
            [None],
            1_080,
        )

        self.assertEqual(assignments, {})
        self.assertEqual(refiner.direct_track_rejections, 1)

    def test_native_gsr_stitches_a_conservative_tracker_fragment(self):
        import cv2
        import numpy as np

        frame = np.full((180, 320, 3), (45, 145, 45), dtype=np.uint8)
        cv2.rectangle(frame, (80, 25), (120, 150), (30, 30, 190), -1)
        refiner = NativeIdentityRefiner(home_team_cluster="B")
        first = refiner.process(
            frame,
            [TrackedObject("raw-1", "player", (80, 25, 120, 150), 0.91)],
            1_000,
        )[0]
        second = refiner.process(
            frame,
            [TrackedObject("raw-99", "player", (82, 25, 122, 150), 0.89)],
            1_080,
        )[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(refiner.diagnostics()["fragments_stitched"], 1)
        self.assertEqual(refiner.diagnostics()["canonical_tracks"], 1)

    def test_native_gsr_learns_teams_from_tracklet_torsos(self):
        import cv2
        import numpy as np

        frame = np.full((220, 640, 3), (45, 145, 45), dtype=np.uint8)
        boxes = []
        for index in range(6):
            x = 20 + index * 48
            box = (x, 30, x + 28, 180)
            boxes.append((box, "red", f"red-{index}"))
            cv2.rectangle(frame, (x, 30), (x + 28, 180), (25, 25, 190), -1)
        for index in range(6):
            x = 340 + index * 48
            box = (x, 30, x + 28, 180)
            boxes.append((box, "white", f"white-{index}"))
            cv2.rectangle(frame, (x, 30), (x + 28, 180), (240, 240, 240), -1)

        refiner = NativeIdentityRefiner(home_team_cluster="B")
        refined = []
        # Three observations calibrate the two video-only jersey clusters, then
        # three stable votes deliberately lock each tracklet to one team.
        for frame_index in range(6):
            refined = refiner.process(
                frame,
                [
                    TrackedObject(raw_id, "player", box, 0.9)
                    for box, _colour, raw_id in boxes
                ],
                1_000 + frame_index * 80,
            )

        teams = {
            raw_id: obj.team_key
            for (_box, _colour, raw_id), obj in zip(boxes, refined)
        }
        self.assertTrue(all(teams[f"red-{index}"] == "home" for index in range(6)))
        self.assertTrue(all(teams[f"white-{index}"] == "away" for index in range(6)))
        diagnostics = refiner.diagnostics()
        self.assertEqual(diagnostics["status"], "ready")
        self.assertEqual(diagnostics["source"], "video_only")
        self.assertEqual(diagnostics["team_tracklets"], 12)

    def test_native_gsr_preserves_global_state_when_tracker_window_resets(self):
        import cv2
        import numpy as np

        frame = np.full((180, 320, 3), (45, 145, 45), dtype=np.uint8)
        cv2.rectangle(frame, (80, 25), (120, 150), (30, 30, 190), -1)
        refiner = NativeIdentityRefiner(home_team_cluster="B")
        first = refiner.process(
            frame,
            [TrackedObject("raw-1", "player", (80, 25, 120, 150), 0.91)],
            1_000,
        )[0]

        refiner.reset_window()
        second = refiner.process(
            frame,
            [TrackedObject("raw-1", "player", (82, 25, 122, 150), 0.91)],
            1_080,
        )[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(refiner.diagnostics()["canonical_tracks"], 1)
        self.assertEqual(refiner.diagnostics()["windows"], 1)

    def test_native_gsr_votes_jersey_then_resolves_unique_roster_player(self):
        import numpy as np

        class FakeJersey:
            model = object()
            backend = "fake_ocr"

            @staticmethod
            def recognize(_frame, _box):
                return JerseyObservation(10, 0.95, "fake_ocr")

            @staticmethod
            def diagnostics():
                return {
                    "backend": "fake_ocr",
                    "ready": True,
                    "eligible_crops": 3,
                }

        frame = np.full((180, 320, 3), (45, 145, 45), dtype=np.uint8)
        refiner = NativeIdentityRefiner(
            roster=[
                {
                    "id": 7,
                    "team_key": "home",
                    "shirt_number": 10,
                    "name": "Joueur Test",
                }
            ],
            jersey_interval_frames=1,
        )
        refiner.jersey = FakeJersey()
        refined = None
        for frame_index in range(4):
            refined = refiner.process(
                frame,
                [TrackedObject("raw-1", "player", (80, 25, 120, 150), 0.91)],
                1_000 + frame_index * 80,
            )[0]

        self.assertEqual(refined.shirt_number, 10)
        self.assertEqual(refined.metadata["roster_player_id"], 7)
        self.assertEqual(refined.metadata["roster_player_name"], "Joueur Test")

    def test_match_scoped_native_identity_is_not_prefixed_per_window(self):
        class Camera:
            @staticmethod
            def stabilize_point(x, y):
                return x, y

        analysis = FrameAnalysis(
            1_000,
            320,
            180,
            0.8,
            [
                TrackedObject(
                    "native-9",
                    "player",
                    (80, 25, 120, 150),
                    0.91,
                    image_x=0.3125,
                    image_y=0.8333,
                    metadata={"identity_scope": "match"},
                )
            ],
        )
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        summaries = {}

        runner._normalize_objects(analysis, Camera(), None, "p2-w4-", summaries)

        self.assertEqual(analysis.objects[0].track_id, "native-9")
        self.assertIn("native-9", summaries)

    def test_native_live_window_draws_without_changing_analysis(self):
        import numpy as np

        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {"analysis_mode": "sample"}
        runner._live_track_history = {}
        runner._live_debug_overlay = False
        analysis = FrameAnalysis(
            timestamp_ms=18_000,
            width=320,
            height=180,
            field_score=0.8,
            objects=[
                TrackedObject(
                    "p1-w1-athlete-7",
                    "player",
                    (80, 40, 105, 120),
                    0.9,
                    team_key="home",
                )
            ],
            diagnostics={
                "raw_athlete_detections": 1,
                "raw_ball_detections": 0,
                "raw_detections": [],
                "team_calibration": {"status": "ready", "samples": 20},
            },
        )
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("d")):
            visible = runner._show_live_tracking(
                np.zeros((180, 320, 3), dtype=np.uint8),
                analysis,
                elapsed_ms=18_000,
                total_ms=120_000,
                period_number=1,
                window_index=1,
                window_count=2,
                team_labels={"home": "ST (T1)", "away": "CSS (T2)"},
            )

        self.assertTrue(visible)
        self.assertTrue(runner._live_debug_overlay)
        self.assertEqual(len(analysis.objects), 1)

    def test_reference_uses_eight_five_second_windows_across_both_halves(self):
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {
            "analysis_mode": "reference",
            "sample_window_seconds": 5,
            "sample_windows_per_half": 4,
        }
        periods = [
            SimpleNamespace(number=1, video_start_ms=0, video_end_ms=2_700_000),
            SimpleNamespace(number=2, video_start_ms=3_300_000, video_end_ms=6_000_000),
        ]

        windows = runner._tracking_windows(periods)

        self.assertEqual(len(windows), 8)
        self.assertEqual(sum(item["end_ms"] - item["start_ms"] for item in windows), 40_000)
        self.assertEqual(
            [item["period"].number for item in windows],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )

    def test_validation_sample_uses_one_continuous_minute_per_half(self):
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {
            "analysis_mode": "sample",
            "sample_window_seconds": 60,
            "sample_windows_per_half": 1,
        }
        periods = [
            SimpleNamespace(number=1, video_start_ms=0, video_end_ms=2_700_000),
            SimpleNamespace(number=2, video_start_ms=3_300_000, video_end_ms=6_000_000),
        ]

        windows = runner._tracking_windows(periods)

        self.assertEqual(len(windows), 2)
        self.assertEqual(
            sum(item["end_ms"] - item["start_ms"] for item in windows),
            120_000,
        )
        self.assertEqual(
            [item["end_ms"] - item["start_ms"] for item in windows],
            [60_000, 60_000],
        )
        self.assertEqual(
            [item["period"].number for item in windows],
            [1, 2],
        )

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
        self.assertIn("tracker", " ".join(diagnostics["issues"]))

    def test_native_gsr_reports_histogram_fallback_instead_of_claiming_deep_reid(self):
        diagnostics = MatchAnalysisRunner._tracking_diagnostics(
            Counter(
                {
                    "frames": 100,
                    "raw_athlete_detections": 1_000,
                    "athlete_observations": 1_000,
                    "ball_visible_frames": 50,
                    "field_frames": 100,
                    "state_controlled": 50,
                }
            ),
            Counter({"home": 500, "away": 500}),
            track_count=20,
            tracking_duration_ms=120_000,
            profile_name="native_gsr",
            team_calibration={
                "status": "ready",
                "appearance_backend": "histogram",
                "appearance_error": "",
            },
        )

        self.assertEqual(diagnostics["verdict"], "warning")
        self.assertIn("Re-ID profond", " ".join(diagnostics["issues"]))

    def test_pth_reid_uses_bundled_osnet_without_torchreid_package(self):
        fake_module = ModuleType("pipeline.providers.osnet")

        class FakeExtractor:
            def __init__(self, model_path, device):
                self.model_path = model_path
                self.device = device

        fake_module.OSNetFeatureExtractor = FakeExtractor
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
            with patch.dict(sys.modules, {"pipeline.providers.osnet": fake_module}):
                encoder = NativeAppearanceEncoder(
                    checkpoint.name,
                    "0",
                    backend="osnet",
                    model_name="osnet_x0_25",
                )

        self.assertEqual(encoder.model_kind, "osnet")
        self.assertEqual(encoder.backend, "native_osnet_x0_25")
        self.assertEqual(encoder.model.device, "0")

    @patch("pipeline.runner.build_provider")
    def test_native_gsr_profile_selects_native_windows_provider(self, build_provider):
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {"yolo_profile": "native_gsr"}

        runner._build_legacy_provider("yolo", "cpu", 12.5)

        self.assertEqual(build_provider.call_args.args[0], "native_gsr")
        self.assertEqual(build_provider.call_args.kwargs["profile"], "native_gsr")

    @patch("pipeline.runner.build_provider")
    def test_external_gsr_ball_only_keeps_plain_yolo_provider(self, build_provider):
        runner = MatchAnalysisRunner.__new__(MatchAnalysisRunner)
        runner.config = {"yolo_profile": "native_gsr", "yolo_ball_class_ids": [0]}

        runner._build_legacy_provider("yolo", "cpu", 12.5, ball_only=True)

        self.assertEqual(build_provider.call_args.args[0], "yolo")
        self.assertEqual(build_provider.call_args.kwargs["player_class_ids"], [])

    def test_native_gsr_detects_hard_broadcast_cut(self):
        import numpy as np

        provider = NativeGSRVisionProvider.__new__(NativeGSRVisionProvider)
        provider.previous_scene_gray = None
        first = np.zeros((180, 320, 3), dtype=np.uint8)
        second = np.full((180, 320, 3), 255, dtype=np.uint8)

        self.assertFalse(provider._detect_scene_cut(first))
        self.assertTrue(provider._detect_scene_cut(second))

    def test_native_gsr_retries_at_lower_resolution_after_memory_error(self):
        class MemoryLimitedBase:
            image_size = 1280

            def __init__(self):
                self.calls = 0

            def analyze_frame(self, _frame, timestamp_ms):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("CUDA out of memory")
                return FrameAnalysis(timestamp_ms, 320, 180, 0.8, [])

        class IdentityPassthrough:
            @staticmethod
            def process(_frame, objects, _timestamp_ms):
                return objects

            @staticmethod
            def diagnostics():
                return {"status": "collecting"}

        provider = NativeGSRVisionProvider.__new__(NativeGSRVisionProvider)
        provider.base = MemoryLimitedBase()
        provider.refiner = IdentityPassthrough()
        provider.profile = "native_gsr"
        provider.tracker_name = "botsort+native_reid"
        provider.image_size = 1280
        provider.adaptive_resizes = []

        analysis = provider.analyze_frame(None, 1_000)

        self.assertEqual(provider.base.calls, 2)
        self.assertEqual(provider.image_size, 960)
        self.assertEqual(analysis.diagnostics["image_size"], 960)
        self.assertEqual(
            analysis.diagnostics["adaptive_image_resizes"],
            [{"from": 1280, "to": 960}],
        )


class NativeGSRModuleTests(unittest.TestCase):
    def test_weighted_jersey_vote_requires_repeated_distinct_reads(self):
        vote = WeightedNumberVote()
        vote.add(10, 0.95, 1_000)
        self.assertIsNone(vote.result()[0])
        vote.add(10, 0.90, 1_100)
        vote.add(10, 0.92, 1_200)
        vote.add(17, 0.25, 1_300)

        number, confidence = vote.result()

        self.assertEqual(number, 10)
        self.assertGreater(confidence, 0.75)

    def test_jersey_parser_rejects_words_and_more_than_two_digits(self):
        self.assertEqual(JerseyNumberRecognizer._parse_number("# 07"), 7)
        self.assertIsNone(JerseyNumberRecognizer._parse_number("abc"))
        self.assertIsNone(JerseyNumberRecognizer._parse_number("123"))

    def test_pitch_schema_requires_exact_checkpoint_landmark_count(self):
        with tempfile.TemporaryDirectory() as directory:
            schema = Path(directory) / "pitch.json"
            schema.write_text(
                json.dumps(
                    {
                        "landmarks": [
                            {"index": index, "name": str(index), "pitch_xy": [index, 0]}
                            for index in range(4)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "97 points"):
                load_pitch_landmarks(str(schema), expected_count=97)
            self.assertEqual(len(load_pitch_landmarks(str(schema), expected_count=4)), 4)

    def test_pitch_homography_accepts_consistent_semantic_points(self):
        calibrator = NativePitchCalibrator(expected_landmarks=4)
        calibrator.landmarks = [
            PitchLandmark(0, "top-left", 0, 0),
            PitchLandmark(1, "top-right", 105, 0),
            PitchLandmark(2, "bottom-right", 105, 68),
            PitchLandmark(3, "bottom-left", 0, 68),
        ]
        observations = [
            (0, 0, 0, 0.99),
            (1, 1920, 0, 0.99),
            (2, 1920, 1080, 0.99),
            (3, 0, 1080, 0.99),
        ]

        result = calibrator._fit(observations, 1920, 1080)

        self.assertIsNotNone(result)
        self.assertEqual(result.inliers, 4)
        calibrator.homography = result.homography
        center = calibrator.project(960, 540)
        self.assertAlmostEqual(center[0], 52.5, places=1)
        self.assertAlmostEqual(center[1], 34.0, places=1)
        self.assertEqual(calibrator.outside_distance(*center), 0.0)
        self.assertAlmostEqual(calibrator.outside_distance(-3, 72), 5.0)

    def test_roster_identity_consolidates_non_simultaneous_fragments(self):
        def summary(start, player_id):
            return {
                "start_ms": start,
                "end_ms": start + 1_000,
                "confidence_sum": 5.0,
                "identity_confidence_sum": 4.0,
                "samples": 5,
                "team_votes": Counter({"home": 4.0}),
                "shirt_votes": Counter({10: 3.0}),
                "role_votes": Counter({"player": 4.0}),
                "roster_player_votes": Counter({player_id: 3.0}),
                "points": [{"t": start, "x": 1, "y": 2, "space": "pitch_meters"}],
                "identity_evidence": {},
            }

        summaries = {
            "native-1": summary(1_000, 7),
            "native-9": summary(20_000, 7),
        }

        aliases = MatchAnalysisRunner._consolidate_roster_tracks(summaries)

        self.assertEqual(aliases, {"native-9": "native-1"})
        self.assertEqual(set(summaries), {"native-1"})
        self.assertEqual(summaries["native-1"]["samples"], 10)

    def test_roster_identity_never_merges_simultaneous_players(self):
        shared = {
            "start_ms": 1_000,
            "end_ms": 2_000,
            "confidence_sum": 2.0,
            "identity_confidence_sum": 2.0,
            "samples": 2,
            "team_votes": Counter({"home": 2.0}),
            "shirt_votes": Counter({10: 2.0}),
            "role_votes": Counter({"player": 2.0}),
            "roster_player_votes": Counter({7: 2.0}),
            "identity_evidence": {},
        }
        summaries = {
            "native-1": {**shared, "points": [{"t": 1_000}]},
            "native-2": {**shared, "points": [{"t": 1_080}]},
        }

        aliases = MatchAnalysisRunner._consolidate_roster_tracks(summaries)

        self.assertEqual(aliases, {})
        self.assertEqual(len(summaries), 2)

    def test_tracking_evaluation_measures_detection_identity_team_and_pitch(self):
        with tempfile.TemporaryDirectory() as directory:
            truth_path = Path(directory) / "truth.csv"
            truth_path.write_text(
                "timestamp_ms,object_id,role,x1,y1,x2,y2,team,shirt_number,pitch_x,pitch_y\n"
                "1000,p10,player,10,10,30,70,home,10,20,30\n"
                "1100,p10,player,12,10,32,70,home,10,21,30\n",
                encoding="utf-8",
            )
            tracking_path = Path(directory) / "tracking.ndjson"
            rows = []
            for timestamp, bbox, pitch_x in (
                (1_000, [10, 10, 30, 70], 20),
                (1_100, [12, 10, 32, 70], 21),
            ):
                rows.append(
                    json.dumps(
                        {
                            "frame": {
                                "timestamp_ms": timestamp,
                                "objects": [
                                    {
                                        "track_id": "native-1",
                                        "role": "player",
                                        "bbox_xyxy": bbox,
                                        "team_key": "home",
                                        "shirt_number": 10,
                                        "pitch_x": pitch_x,
                                        "pitch_y": 30,
                                    }
                                ],
                            }
                        }
                    )
                )
            tracking_path.write_text("\n".join(rows), encoding="utf-8")

            metadata = validate_ground_truth(truth_path)
            result = evaluate_tracking(tracking_path, truth_path)

        self.assertEqual(metadata["frames"], 2)
        self.assertEqual(result["precision_pct"], 100.0)
        self.assertEqual(result["recall_pct"], 100.0)
        self.assertEqual(result["identity_consistency_pct"], 100.0)
        self.assertEqual(result["idf1_pct"], 100.0)
        self.assertEqual(result["hota_50_pct"], 100.0)
        self.assertEqual(result["team_accuracy_pct"], 100.0)
        self.assertEqual(result["jersey_accuracy_pct"], 100.0)
        self.assertEqual(result["mean_pitch_error_m"], 0.0)

    def test_unreviewed_prediction_draft_is_not_accepted_as_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            truth_path = Path(directory) / "truth.csv"
            truth_path.write_text(
                "timestamp_ms,object_id,role,x1,y1,x2,y2,reviewed\n"
                "1000,p10,player,10,10,30,70,NO\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reviewed=YES"):
                validate_ground_truth(truth_path)


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
