from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contract import GSR_SCHEMA, GSRContractError, GSRFrameStore

GSR_ENGINE_PROFILES: dict[str, dict[str, str]] = {
    "tracklab": {
        "label": "TrackLab + sn-gamestate",
        "upstream": "https://github.com/SoccerNet/sn-gamestate",
        "architecture": "TrackLab modulaire + pipeline SoccerNet GSR",
        "license": "GPL-3.0 (sn-gamestate) / MIT (TrackLab)",
        "benchmark": "Baseline officielle SoccerNet GSR",
    },
    "winner2025": {
        "label": "SoccernetGSR Winner 2025",
        "upstream": "https://github.com/yinmayoo185/SoccernetGSR",
        "architecture": "YOLOX + Deep-EIoU/OSNet + SFR + OCR/CLIP + IDATR",
        "license": "Exécution externe uniquement : aucune licence racine publiée",
        "benchmark": "Vainqueur SoccerNet GSR 2025 · GS-HOTA 63.90",
    },
}


class ExternalGSRExecutor:
    """Run a separately installed GSR engine through a stable file contract."""

    def __init__(
        self,
        *,
        engine: str,
        command: list[str],
        timeout_seconds: int,
        frame_tolerance_ms: int,
        home_team_cluster: str,
    ):
        self.engine = engine.strip().lower()
        if self.engine not in GSR_ENGINE_PROFILES:
            raise GSRContractError(f"Moteur GSR inconnu : {engine}")
        self.command = [str(item) for item in command if str(item).strip()]
        self.timeout_seconds = max(60, int(timeout_seconds))
        self.frame_tolerance_ms = max(1, int(frame_tolerance_ms))
        self.home_team_cluster = home_team_cluster

    def prepare(
        self,
        *,
        work_dir: Path,
        video_path: str,
        run_id: str,
        match_id: str,
        windows: list[dict[str, Any]],
        tracking_fps: float,
        precomputed_result: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
        live_preview_path: Path | None = None,
    ) -> tuple[GSRFrameStore, dict[str, Any]]:
        result_path = work_dir / "gsr-result.ndjson"
        progress_path = work_dir / "gsr-progress.json"
        manifest_path = work_dir / "gsr-request.json"
        engine_log_path = work_dir / "gsr-engine.log"

        manifest = {
            "schema": GSR_SCHEMA,
            "request_type": "athlete_tracking",
            "engine": self.engine,
            "run_id": run_id,
            "match_id": match_id,
            "source_video": str(Path(video_path).resolve()),
            "target_fps": tracking_fps,
            "windows": [
                {
                    "period": int(window["period"].number),
                    "index": int(window["index"]),
                    "start_ms": int(window["start_ms"]),
                    "end_ms": int(window["end_ms"]),
                }
                for window in windows
            ],
            "output": {
                "result_ndjson": str(result_path.resolve()),
                "progress_json": str(progress_path.resolve()),
                "live_preview_jpg": (str(live_preview_path.resolve()) if live_preview_path else ""),
            },
            "requirements": {
                "coordinate_system": "image_pixels+pitch_meters",
                "roles": ["player", "goalkeeper", "referee"],
                "stable_team_clusters": ["A", "B"],
                "ball_owned_by_django_sidecar": True,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        source_result = self._resolve_precomputed(
            precomputed_result,
            run_id=run_id,
            match_id=match_id,
        )
        if source_result:
            result_path = source_result
        else:
            if not self.command:
                raise GSRContractError(
                    f"{GSR_ENGINE_PROFILES[self.engine]['label']} est sélectionné, mais "
                    "GSR_RUNNER_COMMAND_JSON est vide. Installe le moteur dans un "
                    "environnement séparé puis configure sa commande passerelle."
                )
            self._run_process(
                manifest_path=manifest_path,
                progress_path=progress_path,
                engine_log_path=engine_log_path,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        store = GSRFrameStore.load(
            result_path,
            home_team_cluster=self.home_team_cluster,
        )
        if store.engine not in {self.engine, "external"}:
            raise GSRContractError(
                f"Le résultat annonce le moteur {store.engine}, pas {self.engine}."
            )
        audit = {
            "schema": GSR_SCHEMA,
            "engine": self.engine,
            "engine_label": GSR_ENGINE_PROFILES[self.engine]["label"],
            "engine_revision": store.engine_revision,
            "upstream": GSR_ENGINE_PROFILES[self.engine]["upstream"],
            "license_boundary": GSR_ENGINE_PROFILES[self.engine]["license"],
            "benchmark": GSR_ENGINE_PROFILES[self.engine]["benchmark"],
            "frames_imported": len(store.frames),
            "first_timestamp_ms": store.timestamps[0],
            "last_timestamp_ms": store.timestamps[-1],
            "manifest": manifest,
        }
        return store, audit

    def _run_process(
        self,
        *,
        manifest_path: Path,
        progress_path: Path,
        engine_log_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancel_callback: Callable[[], None] | None,
    ) -> None:
        argv = [*self.command, "--manifest", str(manifest_path.resolve())]
        started = time.monotonic()
        last_progress_mtime = -1
        try:
            with engine_log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    argv,
                    cwd=str(manifest_path.parent),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    if cancel_callback:
                        try:
                            cancel_callback()
                        except Exception:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            raise
                    elapsed = time.monotonic() - started
                    if elapsed > self.timeout_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise GSRContractError(
                            f"Le moteur GSR a dépassé {self.timeout_seconds} secondes."
                        )
                    if progress_callback and progress_path.is_file():
                        mtime = progress_path.stat().st_mtime_ns
                        if mtime != last_progress_mtime:
                            try:
                                payload = json.loads(progress_path.read_text(encoding="utf-8"))
                            except (OSError, json.JSONDecodeError):
                                payload = None
                            if isinstance(payload, dict):
                                progress_callback(payload)
                                last_progress_mtime = mtime
                    time.sleep(0.5)
                return_code = process.wait()
        except OSError as exc:
            raise GSRContractError(
                f"Impossible de démarrer le moteur GSR ({argv[0]}) : {exc}"
            ) from exc
        if return_code:
            try:
                tail = engine_log_path.read_text(encoding="utf-8")[-4_000:]
            except OSError:
                tail = ""
            raise GSRContractError(
                f"Le moteur GSR s’est arrêté avec le code {return_code}.\n{tail}"
            )

    @staticmethod
    def _resolve_precomputed(value: str, *, run_id: str, match_id: str) -> Path | None:
        if not value.strip():
            return None
        expanded = value.replace("{run_id}", run_id).replace("{match_id}", match_id)
        path = Path(expanded).expanduser()
        if not path.is_file():
            raise GSRContractError(f"GSR_PRECOMPUTED_RESULT introuvable : {path}")
        return path
