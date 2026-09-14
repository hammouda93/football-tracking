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
        needs_yolo = (
            settings.ANALYSIS_ATHLETE_ENGINE == "legacy"
            and settings.ANALYSIS_BACKEND == "yolo"
        ) or (
            settings.ANALYSIS_ATHLETE_ENGINE != "legacy"
            and settings.GSR_BALL_BACKEND == "yolo"
        )
        if needs_yolo:
            for module_name in ["ultralytics", "supervision"]:
                try:
                    module = importlib.import_module(module_name)
                except ImportError as exc:
                    checks.append((False, module_name, str(exc)))
                else:
                    checks.append((True, module_name, getattr(module, "__version__", "installé")))
            model = Path(settings.YOLO_MODEL_PATH)
            checks.append((model.exists(), "poids YOLO", str(model)))
        if settings.YOLO_PROFILE == "native_gsr":
            try:
                import torch

                cuda_ready = bool(torch.cuda.is_available())
                cuda_detail = (
                    torch.cuda.get_device_name(0) if cuda_ready else "CUDA indisponible"
                )
                checks.append(
                    (
                        cuda_ready or str(settings.ANALYSIS_DEVICE).lower() == "cpu",
                        "calcul Native GSR",
                        cuda_detail,
                    )
                )
            except ImportError as exc:
                checks.append((False, "PyTorch", str(exc)))

            reid_path = Path(settings.NATIVE_GSR_REID_MODEL_PATH or "")
            checks.append(
                (
                    bool(settings.NATIVE_GSR_REID_MODEL_PATH) and reid_path.is_file(),
                    "poids Re-ID OSNet",
                    str(reid_path) if settings.NATIVE_GSR_REID_MODEL_PATH else "non configurés",
                )
            )
            if settings.NATIVE_GSR_REID_BACKEND in {"osnet", "torchreid"} or (
                settings.NATIVE_GSR_REID_MODEL_PATH
                and ".pth" in reid_path.name.lower()
            ):
                try:
                    importlib.import_module("pipeline.providers.osnet")
                except (ImportError, RuntimeError) as exc:
                    checks.append((False, "OSNet natif", str(exc)))
                else:
                    checks.append((True, "OSNet natif", "inférence PyTorch disponible"))

            jersey_engine = settings.NATIVE_GSR_JERSEY_ENGINE
            if jersey_engine == "easyocr":
                try:
                    easyocr = importlib.import_module("easyocr")
                except ImportError as exc:
                    checks.append((False, "OCR maillot", str(exc)))
                else:
                    checks.append(
                        (True, "OCR maillot", getattr(easyocr, "__version__", "EasyOCR"))
                    )
            elif jersey_engine == "onnx":
                jersey_path = Path(settings.NATIVE_GSR_JERSEY_MODEL_PATH or "")
                checks.append(
                    (
                        bool(settings.NATIVE_GSR_JERSEY_MODEL_PATH)
                        and jersey_path.is_file(),
                        "OCR maillot ONNX",
                        str(jersey_path),
                    )
                )
            else:
                checks.append(
                    (False, "OCR maillot", f"moteur '{jersey_engine}' non activé")
                )

            pitch_model = Path(settings.NATIVE_GSR_PITCH_MODEL_PATH or "")
            pitch_schema = Path(settings.NATIVE_GSR_PITCH_SCHEMA_PATH or "")
            pitch_files_ready = (
                bool(settings.NATIVE_GSR_PITCH_MODEL_PATH)
                and bool(settings.NATIVE_GSR_PITCH_SCHEMA_PATH)
                and pitch_model.is_file()
                and pitch_schema.is_file()
            )
            pitch_detail = f"modèle={pitch_model or '—'} ; schéma={pitch_schema or '—'}"
            if pitch_files_ready:
                try:
                    from pipeline.pitch import load_pitch_landmarks

                    points = load_pitch_landmarks(
                        str(pitch_schema),
                        expected_count=settings.NATIVE_GSR_PITCH_EXPECTED_LANDMARKS,
                    )
                    pitch_detail = f"{len(points)} points ; {pitch_model.name}"
                except (OSError, TypeError, ValueError) as exc:
                    pitch_files_ready = False
                    pitch_detail = str(exc)
            checks.append((pitch_files_ready, "calibration terrain", pitch_detail))
            checks.append(
                (
                    bool(settings.YOLO_BALL_TILED_RECOVERY),
                    "récupération ballon",
                    (
                        f"4 tuiles tous les {settings.YOLO_BALL_RECOVERY_INTERVAL_FRAMES} "
                        f"frames à {settings.YOLO_BALL_RECOVERY_IMAGE_SIZE}px"
                        if settings.YOLO_BALL_TILED_RECOVERY
                        else "désactivée"
                    ),
                )
            )
        if settings.ANALYSIS_ATHLETE_ENGINE != "legacy":
            command = settings.GSR_RUNNER_COMMAND
            precomputed = settings.GSR_PRECOMPUTED_RESULT
            if precomputed:
                checks.append(
                    (
                        True,
                        "passerelle GSR",
                        f"résultat pré-calculé configuré ({settings.ANALYSIS_ATHLETE_ENGINE})",
                    )
                )
            elif command:
                executable = command[0]
                resolved = shutil.which(executable) or (
                    executable if Path(executable).is_file() else None
                )
                checks.append(
                    (
                        bool(resolved),
                        "passerelle GSR",
                        f"{settings.ANALYSIS_ATHLETE_ENGINE}: {' '.join(command)}",
                    )
                )
            else:
                checks.append(
                    (
                        False,
                        "passerelle GSR",
                        "GSR_RUNNER_COMMAND_JSON est vide",
                    )
                )
        for ok, label, detail in checks:
            marker = self.style.SUCCESS("OK") if ok else self.style.ERROR("MANQUANT")
            self.stdout.write(f"[{marker}] {label}: {detail}")
        if not all(ok for ok, _, _ in checks):
            self.stdout.write(
                self.style.WARNING(
                    "Le mode diagnostic peut fonctionner partiellement, mais corrige les éléments manquants avant une analyse complète."
                )
            )
