from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from matches.models import AnalysisRun, Match
from pipeline.runner import run_analysis


class Command(BaseCommand):
    help = "Lance immédiatement l’analyse d’un match existant."

    def add_arguments(self, parser):
        parser.add_argument("match_id", help="UUID du match")
        parser.add_argument("--backend", choices=["heuristic", "yolo"], default="heuristic")
        parser.add_argument("--tracking-fps", type=float, default=10.0)
        parser.add_argument("--no-clips", action="store_true")

    def handle(self, *args, **options):
        try:
            match = Match.objects.get(pk=options["match_id"])
        except (Match.DoesNotExist, ValueError) as exc:
            raise CommandError("Match introuvable.") from exc
        run = AnalysisRun.objects.create(
            match=match,
            config={
                "backend": options["backend"],
                "tracking_fps": options["tracking_fps"],
                "min_yolo_tracking_fps": settings.ANALYSIS_MIN_YOLO_TRACKING_FPS,
                "sample_seconds": 1.0,
                "quality_max_samples": settings.ANALYSIS_QUALITY_MAX_SAMPLES,
                "render_clips": not options["no_clips"],
                "device": settings.ANALYSIS_DEVICE,
                "live_window": settings.ANALYSIS_LIVE_WINDOW,
                "yolo_profile": settings.YOLO_PROFILE,
                "yolo_model_path": settings.YOLO_MODEL_PATH,
                "yolo_confidence": settings.YOLO_CONFIDENCE,
                "yolo_ball_confidence": settings.YOLO_BALL_CONFIDENCE,
                "yolo_image_size": settings.YOLO_IMAGE_SIZE,
                "yolo_tracker": settings.YOLO_TRACKER,
                "yolo_track_low_confidence": settings.YOLO_TRACK_LOW_CONFIDENCE,
                "yolo_new_track_confidence": settings.YOLO_NEW_TRACK_CONFIDENCE,
                "yolo_track_match_threshold": settings.YOLO_TRACK_MATCH_THRESHOLD,
                "yolo_track_buffer_seconds": settings.YOLO_TRACK_BUFFER_SECONDS,
                "yolo_player_class_ids": settings.YOLO_PLAYER_CLASS_IDS,
                "yolo_goalkeeper_class_ids": settings.YOLO_GOALKEEPER_CLASS_IDS,
                "yolo_referee_class_ids": settings.YOLO_REFEREE_CLASS_IDS,
                "yolo_ball_class_ids": settings.YOLO_BALL_CLASS_IDS,
            },
        )
        self.stdout.write(f"Analyse {run.pk} créée.")
        completed = run_analysis(run.pk)
        self.stdout.write(self.style.SUCCESS(completed.get_status_display()))
