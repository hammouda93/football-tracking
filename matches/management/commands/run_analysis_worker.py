from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from pipeline.runner import claim_next_analysis, run_analysis


class Command(BaseCommand):
    help = "Traite les analyses vidéo mises en file."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Traite au plus une analyse puis quitte.",
        )
        parser.add_argument(
            "--poll-seconds",
            type=float,
            default=2.0,
            help="Délai entre deux vérifications de la file.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        poll_seconds = max(0.25, options["poll_seconds"])
        self.stdout.write(self.style.SUCCESS("Worker Football Tracking prêt."))
        while True:
            run = claim_next_analysis()
            if run is None:
                if once:
                    self.stdout.write("Aucune analyse en attente.")
                    return
                time.sleep(poll_seconds)
                continue
            self.stdout.write(f"Analyse {run.pk} — {run.match}")
            try:
                completed = run_analysis(run.pk)
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"Échec : {exc}"))
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Analyse terminée avec le statut {completed.get_status_display()}."
                    )
                )
            if once:
                return
