from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from scripts.gsr.sn_gamestate_runner import (
    SegmentMap,
    convert_state,
    local_path,
    row_object,
    segment_for_frame,
    source_timestamp,
)


class FakeRows:
    def __init__(self, rows, *, indices=None):
        self.rows = list(rows)
        self.indices = list(indices if indices is not None else range(len(self.rows)))

    def iterrows(self):
        yield from zip(self.indices, self.rows, strict=True)


class SnGamestateRunnerTests(TestCase):
    def setUp(self):
        self.segment = SegmentMap(
            period=2,
            index=1,
            source_start_ms=50_000,
            source_end_ms=60_000,
            composite_start_frame=25,
            frame_count=50,
        )

    def test_wsl_path_translation_preserves_windows_project(self):
        translated = local_path(r"D:\Django_Projects\football-tracking\media\match.mp4")
        if os.name == "nt":
            self.assertEqual(
                str(translated), r"D:\Django_Projects\football-tracking\media\match.mp4"
            )
        else:
            self.assertEqual(
                translated,
                Path("/mnt/d/Django_Projects/football-tracking/media/match.mp4"),
            )

    def test_segment_mapping_uses_source_video_clock(self):
        self.assertIs(segment_for_frame(25, [self.segment]), self.segment)
        self.assertIsNone(segment_for_frame(24, [self.segment]))
        self.assertEqual(source_timestamp(30, self.segment, 5.0), 51_000)

    def test_row_conversion_keeps_tracklab_identity_team_jersey_and_pitch(self):
        item = row_object(
            {
                "track_id": 42.0,
                "role": "player",
                "bbox_ltwh": [100, 200, 50, 120],
                "bbox_conf": 0.93,
                "team_cluster": 1,
                "jersey_number": 10,
                "role_confidence": 0.88,
                "bbox_pitch": {"x_bottom_middle": 12.5, "y_bottom_middle": -7.0},
            },
            width=1920,
            height=1080,
            segment=self.segment,
            goalkeeper_side_map={},
        )
        self.assertIsNotNone(item)
        self.assertEqual(item["track_id"], "42")
        self.assertEqual(item["team_cluster"], "B")
        self.assertEqual(item["shirt_number"], 10)
        self.assertEqual(item["pitch_xy"], [12.5, -7.0])
        self.assertEqual(item["bbox_xyxy"], [100.0, 200.0, 150.0, 320.0])

    def test_converter_writes_every_requested_frame_and_not_separator_frames(self):
        detections = FakeRows(
            [
                {
                    "image_id": 5,
                    "track_id": 8,
                    "role": "referee",
                    "bbox_ltwh": [10, 20, 30, 60],
                    "bbox_conf": 0.9,
                }
            ]
        )
        images = FakeRows([{"frame": 25}], indices=[5])
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.ndjson"
            with mock.patch(
                "scripts.gsr.sn_gamestate_runner.load_tracker_state",
                return_value=(detections, images),
            ):
                convert_state(
                    state_path=Path(directory) / "unused.pklz",
                    result_path=result,
                    segments=[self.segment],
                    fps=5.0,
                    width=1920,
                    height=1080,
                    revision="abc123",
                    device="cuda:test",
                )
            records = [json.loads(line) for line in result.read_text().splitlines()]
        self.assertEqual(records[0]["schema"], "football-tracking.gsr/v1")
        self.assertEqual(records[0]["entered_team_colors_used"], False)
        self.assertEqual(len(records), 51)
        self.assertEqual(records[1]["timestamp_ms"], 50_000)
        self.assertEqual(records[1]["objects"][0]["role"], "referee")
