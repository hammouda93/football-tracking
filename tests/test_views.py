import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.test.utils import override_settings
from django.urls import reverse

from matches.models import AnalysisRun, Match, MatchPeriod, MatchVideo, Team
from pipeline.runner import MatchAnalysisRunner
from pipeline.types import FrameSignal


class DashboardTests(TestCase):
    def test_dashboard_and_match_lab_render(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY", primary_color="#F04438")
        match = Match.objects.create(home_team=home, away_team=away)
        self.assertContains(self.client.get(reverse("dashboard")), "Football Tracking")
        response = self.client.get(reverse("match-detail", kwargs={"pk": match.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Double horloge")

    def test_analysis_status_exposes_live_tracking_detail(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        run = AnalysisRun.objects.create(
            match=match,
            status=AnalysisRun.Status.PROCESSING,
            current_stage=AnalysisRun.Stage.TRACKING,
            progress=22,
            metrics={"live_progress": {"label": "Tracking 0.5% · ETA en calcul"}},
        )

        response = self.client.get(reverse("analysis-status", kwargs={"pk": run.pk}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["progress_detail"]["label"],
            "Tracking 0.5% · ETA en calcul",
        )

    def test_sample_requires_two_confirmed_halves(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)

        response = self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "sample"},
        )

        self.assertRedirects(response, match.get_absolute_url())
        self.assertFalse(match.analysis_runs.exists())

    def test_sample_run_uses_eight_short_windows_and_no_clips(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                match_clock_start_ms=(number - 1) * 2_700_000,
                match_clock_end_ms=number * 2_700_000,
                confirmed=True,
            )

        response = self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "sample"},
        )

        self.assertRedirects(response, match.get_absolute_url())
        run = match.analysis_runs.get()
        self.assertEqual(run.config["analysis_mode"], "sample")
        self.assertEqual(run.config["athlete_engine"], "legacy")
        self.assertEqual(run.config["sample_window_seconds"], 60)
        self.assertEqual(run.config["sample_windows_per_half"], 1)
        self.assertFalse(run.config["render_clips"])
        self.assertEqual(run.config["yolo_profile"], "main_py")
        self.assertEqual(run.config["min_yolo_tracking_fps"], 12.5)
        self.assertEqual(run.config["yolo_tracker"], "bytetrack")
        self.assertEqual(run.config["yolo_track_low_confidence"], 0.30)
        self.assertEqual(run.config["yolo_new_track_confidence"], 0.25)
        self.assertEqual(run.config["yolo_track_match_threshold"], 0.80)
        self.assertEqual(run.config["yolo_ball_confidence"], 0.12)
        self.assertEqual(run.config["yolo_image_size"], 640)
        self.assertEqual(run.config["yolo_player_class_ids"], [2])
        self.assertEqual(run.config["yolo_goalkeeper_class_ids"], [1])
        self.assertEqual(run.config["yolo_ball_class_ids"], [0])
        self.assertEqual(run.config["gsr_tracking_fps"], 5.0)
        self.assertEqual(run.config["home_team_cluster"], "B")

    @override_settings(
        ANALYSIS_ATHLETE_ENGINE="tracklab",
        GSR_RUNNER_COMMAND=[],
        GSR_PRECOMPUTED_RESULT="",
    )
    def test_external_engine_is_blocked_before_queue_when_bridge_is_missing(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                confirmed=True,
            )

        response = self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "reference"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GSR_RUNNER_COMMAND_JSON")
        self.assertFalse(match.analysis_runs.exists())

    def test_reference_run_uses_eight_five_second_windows(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                confirmed=True,
            )

        response = self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "reference"},
        )

        self.assertRedirects(response, match.get_absolute_url())
        run = match.analysis_runs.get()
        self.assertEqual(run.config["analysis_mode"], "reference")
        self.assertEqual(run.config["sample_window_seconds"], 5)
        self.assertEqual(run.config["sample_windows_per_half"], 4)
        self.assertFalse(run.config["render_clips"])

    def test_team_cluster_mapping_can_be_swapped_without_changing_team_colors(self):
        home = Team.objects.create(
            name="Stade Tunisien", short_name="STA", primary_color="#A54840"
        )
        away = Team.objects.create(
            name="Club Sportif Sfaxien", short_name="CSS", primary_color="#F3F2F8"
        )
        match = Match.objects.create(home_team=home, away_team=away)
        sample = AnalysisRun.objects.create(
            match=match,
            status=AnalysisRun.Status.REVIEW,
            config={"analysis_mode": "sample", "home_team_cluster": "B"},
            metrics={
                "diagnostics": {"verdict": "pass", "manual_approved": True}
            },
        )

        response = self.client.post(
            reverse("match-swap-team-clusters", kwargs={"pk": match.pk})
        )

        self.assertRedirects(response, match.get_absolute_url())
        match.refresh_from_db()
        sample.refresh_from_db()
        self.assertEqual(match.home_team_cluster, "A")
        self.assertEqual(home.primary_color, "#A54840")
        self.assertEqual(away.primary_color, "#F3F2F8")
        self.assertTrue(
            sample.metrics["diagnostics"]["team_mapping_changed_since_run"]
        )
        self.assertFalse(sample.metrics["diagnostics"]["manual_approved"])

    def test_live_preview_endpoint_returns_latest_jpeg_without_cache(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        run = AnalysisRun.objects.create(match=match)

        with tempfile.TemporaryDirectory() as media_root, override_settings(
            MEDIA_ROOT=Path(media_root)
        ):
            url = reverse("analysis-live-preview", kwargs={"pk": run.pk})
            self.assertEqual(self.client.get(url).status_code, 404)
            preview_path = (
                Path(media_root)
                / "matches"
                / str(match.pk)
                / "live"
                / f"{run.pk}.jpg"
            )
            preview_path.parent.mkdir(parents=True)
            preview_path.write_bytes(b"jpeg-frame")

            response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"jpeg-frame")
        self.assertIn("no-store", response["Cache-Control"])

    def test_full_analysis_is_blocked_until_sample_passes(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                confirmed=True,
            )

        response = self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "full"},
        )

        self.assertRedirects(response, match.get_absolute_url())
        self.assertFalse(match.analysis_runs.exists())

    def test_manual_sample_validation_unlocks_full_analysis(self):
        home = Team.objects.create(name="Stade Tunisien", short_name="STA")
        away = Team.objects.create(name="Club Sportif Sfaxien", short_name="CSS")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                match_clock_start_ms=(number - 1) * 2_700_000,
                match_clock_end_ms=number * 2_700_000,
                confirmed=True,
            )
        sample = AnalysisRun.objects.create(
            match=match,
            status=AnalysisRun.Status.REVIEW,
            config={"analysis_mode": "sample"},
            metrics={"diagnostics": {"verdict": "fail"}},
        )

        response = self.client.post(
            reverse("analysis-validate-sample", kwargs={"pk": sample.pk}),
            {"confirm": "yes"},
        )

        self.assertRedirects(response, match.get_absolute_url())
        sample.refresh_from_db()
        self.assertTrue(sample.metrics["diagnostics"]["manual_approved"])

        self.client.post(
            reverse("match-start-analysis", kwargs={"pk": match.pk}),
            {"mode": "full"},
        )
        self.assertEqual(match.analysis_runs.count(), 2)
        self.assertTrue(
            match.analysis_runs.filter(config__analysis_mode="full").exists()
        )

    def test_editing_video_periods_invalidates_the_previous_sample(self):
        home = Team.objects.create(name="Stade Tunisien", short_name="STA")
        away = Team.objects.create(name="Club Sportif Sfaxien", short_name="CSS")
        match = Match.objects.create(home_team=home, away_team=away)
        for number, start in ((1, 0), (2, 3_300_000)):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=start + 2_700_000,
                match_clock_start_ms=(number - 1) * 2_700_000,
                match_clock_end_ms=number * 2_700_000,
                confirmed=True,
            )
        sample = AnalysisRun.objects.create(
            match=match,
            status=AnalysisRun.Status.REVIEW,
            config={"analysis_mode": "sample"},
            metrics={
                "diagnostics": {"verdict": "fail", "manual_approved": True}
            },
        )

        response = self.client.post(
            reverse("match-update-periods", kwargs={"pk": match.pk}),
            {
                "p1_start": "00:00.000",
                "p1_end": "45:04.000",
                "p1_clock_start": "00:00.000",
                "p2_start": "48:53.000",
                "p2_end": "01:40:08.000",
                "p2_clock_start": "45:00.000",
            },
        )

        self.assertRedirects(response, match.get_absolute_url())
        second_half = match.periods.get(number=2)
        self.assertEqual(second_half.video_start_ms, 2_933_000)
        self.assertEqual(second_half.match_clock_start_ms, 2_700_000)
        self.assertEqual(second_half.match_clock_end_ms, 5_775_000)
        sample.refresh_from_db()
        self.assertTrue(sample.metrics["diagnostics"]["periods_changed_since_run"])
        self.assertFalse(sample.metrics["diagnostics"]["manual_approved"])

    def test_prepare_recalculates_confirmed_periods_and_invalidates_sample(self):
        home = Team.objects.create(name="Stade Tunisien", short_name="STA")
        away = Team.objects.create(name="Club Sportif Sfaxien", short_name="CSS")
        match = Match.objects.create(home_team=home, away_team=away)
        MatchVideo.objects.create(
            match=match,
            file="matches/test/source.mp4",
            original_name="source.mp4",
        )
        for number, start, end in (
            (1, 0, 2_884_000),
            (2, 2_884_000, 6_000_000),
        ):
            MatchPeriod.objects.create(
                match=match,
                number=number,
                label=f"MT{number}",
                video_start_ms=start,
                video_end_ms=end,
                confirmed=True,
            )
        sample = AnalysisRun.objects.create(
            match=match,
            status=AnalysisRun.Status.REVIEW,
            config={"analysis_mode": "sample"},
            metrics={
                "diagnostics": {
                    "verdict": "warning",
                    "manual_approved": True,
                }
            },
        )
        prepare = AnalysisRun.objects.create(
            match=match,
            config={"analysis_mode": "prepare"},
        )
        signals = [
            FrameSignal(
                timestamp_ms=second * 1_000,
                field_score=(
                    0.03 if 2_880 < second < 2_940 else 0.64
                ),
                sharpness=150,
                brightness=125,
            )
            for second in range(0, 6_001, 10)
        ]

        runner = MatchAnalysisRunner(prepare)
        with patch("pipeline.runner._save_json_artifact"):
            periods = runner._periods(signals, 6_000_000)

        self.assertEqual(periods[1].video_start_ms, 2_940_000)
        self.assertFalse(periods[0].confirmed)
        self.assertEqual(periods[0].source, MatchPeriod.Source.AUTO)
        sample.refresh_from_db()
        self.assertTrue(sample.metrics["diagnostics"]["periods_changed_since_run"])
        self.assertFalse(sample.metrics["diagnostics"]["manual_approved"])

    def test_video_stream_supports_byte_ranges_for_seeking(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY")
        match = Match.objects.create(home_team=home, away_team=away)

        with tempfile.TemporaryDirectory() as media_root:
            with override_settings(MEDIA_ROOT=media_root):
                video = MatchVideo.objects.create(
                    match=match,
                    original_name="match.mp4",
                    size_bytes=16,
                )
                video.file.save(
                    "match.mp4",
                    SimpleUploadedFile("match.mp4", b"0123456789abcdef"),
                )

                response = self.client.get(
                    reverse("match-video-stream", kwargs={"pk": match.pk}),
                    HTTP_RANGE="bytes=4-7",
                )

                self.assertEqual(response.status_code, 206)
                self.assertEqual(response["Accept-Ranges"], "bytes")
                self.assertEqual(response["Content-Range"], "bytes 4-7/16")
                self.assertEqual(response["Content-Length"], "4")
                self.assertEqual(b"".join(response.streaming_content), b"4567")
