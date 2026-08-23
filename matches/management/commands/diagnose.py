from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Vérifie les dépendances locales vidéo et ML."

    def handle(self, *args, **options):
        checks = []
        for module_name in ["django", "numpy", "cv2", "PIL"]:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                checks.append((False, module_name, str(exc)))
            else:
                checks.append((True, module_name, getattr(module, "__version__", "installé")))
        for binary in [settings.FFMPEG_BINARY, settings.FFPROBE_BINARY]:
            resolved = shutil.which(binary)
            if resolved:
                try:
                    completed = subprocess.run(
                        [binary, "-version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    version = (completed.stdout or completed.stderr).splitlines()[0]
                except (OSError, subprocess.SubprocessError) as exc:
                    checks.append((False, binary, str(exc)))
                else:
                    checks.append((True, binary, version))
            else:
                checks.append((False, binary, "absent du PATH"))
        if settings.ANALYSIS_BACKEND == "yolo":
            for module_name in ["ultralytics", "supervision"]:
                try:
                    module = importlib.import_module(module_name)
                except ImportError as exc:
                    checks.append((False, module_name, str(exc)))
                else:
                    checks.append((True, module_name, getattr(module, "__version__", "installé")))
            model = Path(settings.YOLO_MODEL_PATH)
            checks.append((model.exists(), "poids YOLO", str(model)))
        for ok, label, detail in checks:
            marker = self.style.SUCCESS("OK") if ok else self.style.ERROR("MANQUANT")
            self.stdout.write(f"[{marker}] {label}: {detail}")
        if not all(ok for ok, _, _ in checks):
            self.stdout.write(
                self.style.WARNING(
                    "Le mode diagnostic peut fonctionner partiellement, mais corrige les éléments manquants avant une analyse complète."
                )
            )
