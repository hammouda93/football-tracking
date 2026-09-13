from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from matches.models import AnalysisRun, Match, MatchPeriod, MatchVideo, Team
from pipeline.gsr import ExternalGSRExecutor, GSRContractError, GSRFrameStore
from pipeline.gsr.contract import GSR_SCHEMA
from pipeline.providers.gsr import ExternalGSRVisionProvider
from pipeline.runner import MatchAnalysisRunner
from pipeline.types import FrameAnalysis, TrackedObject, VideoMetadata


def _write_result(
    path: Path,
    *,
    engine: str = "tracklab",
    timestamp_ms: int = 1_000,
) -> None:
    records = [
        {
            "type": "metadata",
            "schema": GSR_SCHEMA,
            "engine": engine,
            "engine_revision": "abc123",
            "fps": 25.0,
        },
        {
            "type": "frame",
            "timestamp_ms": timestamp_ms,
            "width": 1920,
            "height": 1080,
            "objects": [
                {
                    "track_id": "42",
                    "role": "player",
                    "bbox_xyxy": [100, 200, 180, 420],
                    "confidence": 0.91,
                    "team_cluster": "B",
                    "shirt_number": 10,
                    "pitch_xy": [0.0, -8.5],
                },
                {
                    "track_id": "ref-2",
                    "role": "referee",
                    "bbox_ltwh": [400, 180, 50, 190],
                    "confidence": 0.87,
                },
            ],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )


class GSRContractTests(unittest.TestCase):
    def test_loads_identity_team_jersey_and_pitch_without_club_colors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.ndjson"
            _write_result(path)

            store = GSRFrameStore.load(path, home_team_cluster="B")

        player = store.frames[0].objects[0]
        self.assertEqual(store.engine, "tracklab")
        self.assertEqual(player.track_id, "gsr-42")
        self.assertEqual(player.team_key, "home")
        self.assertEqual(player.shirt_number, 10)
        self.assertEqual(player.pitch_x, 0.0)
        self.assertEqual(player.pitch_y, -8.5)

    def test_nearest_frame_is_tolerant_and_returns_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.ndjson"
            _write_result(path)
            store = GSRFrameStore.load(path)

            frame = store.nearest(1_060, tolerance_ms=80)
            missing = store.nearest(1_200, tolerance_ms=80)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.timestamp_ms, 1_060)
        self.assertEqual(frame.diagnostics["gsr_source_timestamp_ms"], 1_000)
        frame.objects.clear()
        self.assertEqual(len(store.frames[0].objects), 2)
        self.assertIsNone(missing)

    def test_rejects_challenge_json_without_image_boxes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "official.json"
            path.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "image_id": 1,
                                "track_id": 7,
                                "attributes": {"role": "player", "team": "left"},
                                "bbox_pitch": {"x_bottom_middle": 1, "y_bottom_middle": 2},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GSRContractError, "liste de frames"):
                GSRFrameStore.load(path)


class _BallProvider:
    ball_confidence = 0.12

    def reset(self):
        return None

    def analyze_frame(self, frame, timestamp_ms):
        return FrameAnalysis(
            timestamp_ms=timestamp_ms,
            width=frame.shape[1],
            height=frame.shape[0],
            field_score=0.8,
            objects=[
                TrackedObject(
                    "local-athlete-must-not-leak",
                    "player",
                    (10, 10, 30, 60),
                    0.8,
                ),
                TrackedObject("ball", "ball", (500, 300, 510, 310), 0.7),
            ],
            diagnostics={
                "raw_ball_detections": 2,
                "raw_detections": [
                    {"bbox": [500, 300, 510, 310], "role": "ball", "confidence": 0.7}
                ],
            },
        )


class GSRProviderTests(unittest.TestCase):
    def test_external_athletes_are_combined_with_only_the_local_ball(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.ndjson"
            _write_result(path)
            store = GSRFrameStore.load(path)
            provider = ExternalGSRVisionProvider(
                store,
                tolerance_ms=80,
                ball_provider=_BallProvider(),
            )

            analysis = provider.analyze_frame(
                np.zeros((1080, 1920, 3), dtype=np.uint8),
                1_000,
            )

        self.assertEqual(len(analysis.athletes), 1)
        self.assertEqual(analysis.athletes[0].track_id, "gsr-42")
        self.assertIsNotNone(analysis.ball)
        self.assertNotIn(
            "local-athlete-must-not-leak",
            {item.track_id for item in analysis.objects},
        )


class GSRExecutorTests(unittest.TestCase):
    def test_precomputed_result_creates_an_auditable_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            result_path = work_dir / "cached-tracklab.ndjson"
            _write_result(result_path)
            executor = ExternalGSRExecutor(
                engine="tracklab",
                command=[],
                timeout_seconds=60,
                frame_tolerance_ms=120,
                home_team_cluster="B",
            )
            period = type("Period", (), {"number": 1})()

            store, audit = executor.prepare(
                work_dir=work_dir,
                video_path="match.mp4",
                run_id="run-1",
                match_id="match-1",
                windows=[{"period": period, "index": 1, "start_ms": 0, "end_ms": 2_000}],
                tracking_fps=12.5,
                precomputed_result=str(result_path),
            )

        self.assertEqual(len(store.frames), 1)
        self.assertEqual(audit["schema"], GSR_SCHEMA)
        self.assertEqual(audit["engine_revision"], "abc123")
        self.assertEqual(audit["manifest"]["windows"][0]["end_ms"], 2_000)


class GSRRunnerIntegrationTests(TestCase):
    def test_runner_keeps_existing_pipeline_and_uses_external_identity(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            override_settings(MEDIA_ROOT=Path(temp_dir) / "media"),
        ):
            home = Team.objects.create(name="Home", short_name="HOM")
            away = Team.objects.create(name="Away", short_name="AWY")
            match = Match.objects.create(home_team=home, away_team=away)
            video = MatchVideo.objects.create(
                match=match,
                file=SimpleUploadedFile("match.mp4", b"video"),
                original_name="match.mp4",
            )
            period = MatchPeriod.objects.create(
                match=match,
                number=1,
                label="MT1",
                video_start_ms=0,
                video_end_ms=1_000,
                confirmed=True,
            )
            result_path = Path(temp_dir) / "result.ndjson"
            _write_result(result_path, timestamp_ms=0)
            run = AnalysisRun.objects.create(
                match=match,
                config={
                    "analysis_mode": "full",
                    "backend": "heuristic",
                    "athlete_engine": "tracklab",
                    "tracking_fps": 1.0,
                    "gsr_precomputed_result": str(result_path),
                    "gsr_ball_backend": "none",
                    "home_team_cluster": "B",
                },
            )
            metadata = VideoMetadata(
                path=video.file.path,
                duration_ms=1_000,
                fps=1.0,
                width=1920,
                height=1080,
                frame_count=1,
            )

            with patch(
                "pipeline.runner.iter_frames",
                return_value=iter([(0, np.zeros((1080, 1920, 3), dtype=np.uint8))]),
            ):
                result = MatchAnalysisRunner(run)._track([period], metadata)

        self.assertEqual(result["athlete_engine"], "tracklab")
        self.assertEqual(result["gsr"]["engine_revision"], "abc123")
        summary = next(iter(result["tracks"].values()))
        self.assertEqual(summary["shirt_votes"][10], 1)
        self.assertEqual(summary["team_votes"]["home"], 1)


if __name__ == "__main__":
    unittest.main()
