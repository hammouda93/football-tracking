from __future__ import annotations

from django.core.management.base import BaseCommand

from matches.models import Event, Match, MatchPeriod, Player, Team, TeamMatchStat
from pipeline.stats import blank_metrics


class Command(BaseCommand):
    help = "Crée un match de démonstration sans vidéo pour prévisualiser l’interface."

    def handle(self, *args, **options):
        home, _ = Team.objects.get_or_create(
            name="Carthage FC",
            defaults={"short_name": "CAR", "primary_color": "#13D58C"},
        )
        away, _ = Team.objects.get_or_create(
            name="Atlas United",
            defaults={"short_name": "ATL", "primary_color": "#FF6B4A"},
        )
        if Match.objects.filter(home_team=home, away_team=away, competition="Demo Lab").exists():
            self.stdout.write("Le match de démonstration existe déjà.")
            return
        for team, prefix in [(home, "C"), (away, "A")]:
            for number, position in [(1, "GK"), (4, "CB"), (6, "DM"), (8, "CM"), (9, "ST")]:
                Player.objects.get_or_create(
                    team=team,
                    shirt_number=number,
                    defaults={"name": f"Joueur {prefix}{number}", "position": position},
                )
        match = Match.objects.create(
            home_team=home,
            away_team=away,
            competition="Demo Lab",
            venue="Stade Olympique",
            status=Match.Status.REVIEW,
            home_score=2,
            away_score=1,
        )
        first = MatchPeriod.objects.create(
            match=match,
            number=1,
            label="1re mi-temps",
            video_start_ms=185_000,
            video_end_ms=3_015_000,
            match_clock_start_ms=0,
            match_clock_end_ms=2_830_000,
            source=MatchPeriod.Source.MANUAL,
            confidence=1.0,
            confirmed=True,
        )
        second = MatchPeriod.objects.create(
            match=match,
            number=2,
            label="2e mi-temps",
            video_start_ms=3_945_000,
            video_end_ms=6_825_000,
            match_clock_start_ms=2_700_000,
            match_clock_end_ms=5_580_000,
            source=MatchPeriod.Source.MANUAL,
            confidence=1.0,
            confirmed=True,
        )
        home_players = list(home.players.order_by("shirt_number"))
        away_players = list(away.players.order_by("shirt_number"))
        demo_events = [
            (first, 310_000, "pass", home, home_players[2], home_players[3], "success", 0.94),
            (first, 742_000, "duel", away, away_players[1], None, "success", 0.78),
            (first, 1_266_000, "shot", home, home_players[4], None, "failure", 0.87),
            (second, 4_376_000, "recovery", away, away_players[2], None, "success", 0.82),
            (second, 5_086_000, "goal", home, home_players[4], None, "success", 0.96),
        ]
        for period, video_ms, event_type, team, player, recipient, outcome, confidence in demo_events:
            Event.objects.create(
                match=match,
                period=period,
                event_type=event_type,
                team=team,
                player=player,
                recipient=recipient,
                video_time_ms=video_ms,
                match_time_ms=period.match_clock_start_ms + video_ms - period.video_start_ms,
                outcome=outcome,
                confidence=confidence,
                review_status=Event.ReviewStatus.PENDING,
                start_x=0.35,
                start_y=0.48,
                end_x=0.61,
                end_y=0.44,
            )
        for team, possession, passes, shots in [(home, 56.4, 412, 13), (away, 43.6, 327, 8)]:
            metrics = blank_metrics()
            metrics.update(
                {
                    "possession_pct": possession,
                    "passes": passes,
                    "passes_completed": int(passes * 0.84),
                    "pass_accuracy_pct": 84.0,
                    "shots": shots,
                    "duels": 91,
                    "duels_won": 48 if team == home else 43,
                }
            )
            TeamMatchStat.objects.create(
                match=match,
                team=team,
                minutes_played=95.2,
                metrics=metrics,
            )
        self.stdout.write(self.style.SUCCESS(f"Match de démonstration créé : {match.pk}"))
