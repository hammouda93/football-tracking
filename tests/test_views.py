from django.test import TestCase
from django.urls import reverse

from matches.models import AnalysisRun, Match, MatchPeriod, Team


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

    def test_sample_run_uses_four_short_windows_and_no_clips(self):
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
        self.assertEqual(run.config["sample_window_seconds"], 30)
        self.assertEqual(run.config["sample_windows_per_half"], 2)
        self.assertFalse(run.config["render_clips"])

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
