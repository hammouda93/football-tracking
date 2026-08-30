from django.test import TestCase
from django.urls import reverse

from matches.models import AnalysisRun, Match, Team


class DashboardTests(TestCase):
    def test_dashboard_and_match_lab_render(self):
        home = Team.objects.create(name="Home", short_name="HOM")
        away = Team.objects.create(name="Away", short_name="AWY", primary_color="#F04438")
        match = Match.objects.create(home_team=home, away_team=away)
        self.assertContains(self.client.get(reverse("dashboard")), "Football Tracking")
        response = self.client.get(reverse("match-detail", kwargs={"pk": match.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pistes → joueurs")

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
