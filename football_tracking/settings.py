import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _csv_ints(name: str, default: str = "") -> list[int]:
    return [int(value.strip()) for value in os.getenv(name, default).split(",") if value.strip()]


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "football-tracking-local-development-key")
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [
    item.strip()
    for item in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if item.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "matches",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "football_tracking.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "football_tracking.wsgi.application"
ASGI_APPLICATION = "football_tracking.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 30},
    }
}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "fr-fr"
TIME_ZONE = os.getenv("TIME_ZONE", "Africa/Tunis")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_621_440
DATA_UPLOAD_MAX_MEMORY_SIZE = 20_971_520

ANALYSIS_BACKEND = os.getenv("ANALYSIS_BACKEND", "heuristic")
ANALYSIS_ATHLETE_ENGINE = os.getenv("ANALYSIS_ATHLETE_ENGINE", "legacy").strip().lower()
if ANALYSIS_ATHLETE_ENGINE not in {"legacy", "tracklab", "winner2025"}:
    raise ValueError("ANALYSIS_ATHLETE_ENGINE doit valoir legacy, tracklab ou winner2025.")
ANALYSIS_SAMPLE_SECONDS = float(os.getenv("ANALYSIS_SAMPLE_SECONDS", "1.0"))
ANALYSIS_QUALITY_MAX_SAMPLES = int(os.getenv("ANALYSIS_QUALITY_MAX_SAMPLES", "360"))
ANALYSIS_TRACKING_FPS = float(os.getenv("ANALYSIS_TRACKING_FPS", "10.0"))
ANALYSIS_MIN_YOLO_TRACKING_FPS = float(os.getenv("ANALYSIS_MIN_YOLO_TRACKING_FPS", "8.0"))
ANALYSIS_DEVICE = os.getenv("ANALYSIS_DEVICE", "cpu")
ANALYSIS_LIVE_WINDOW = (
    os.getenv(
        "ANALYSIS_LIVE_WINDOW",
        "1" if os.name == "nt" and DEBUG else "0",
    )
    == "1"
)
YOLO_PROFILE = os.getenv("YOLO_PROFILE", "main_py").strip().lower()
if YOLO_PROFILE not in {"main_py", "advanced"}:
    raise ValueError("YOLO_PROFILE doit valoir main_py ou advanced.")
_yolo_model_path = Path(
    os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "football-players.pt"))
)
YOLO_MODEL_PATH = str(
    _yolo_model_path if _yolo_model_path.is_absolute() else BASE_DIR / _yolo_model_path
)
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.30"))
YOLO_BALL_CONFIDENCE = float(os.getenv("YOLO_BALL_CONFIDENCE", "0.12"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "1280"))
YOLO_TRACKER = os.getenv("YOLO_TRACKER", "bytetrack")
YOLO_TRACK_LOW_CONFIDENCE = float(os.getenv("YOLO_TRACK_LOW_CONFIDENCE", "0.10"))
YOLO_NEW_TRACK_CONFIDENCE = float(os.getenv("YOLO_NEW_TRACK_CONFIDENCE", "0.25"))
YOLO_TRACK_MATCH_THRESHOLD = float(os.getenv("YOLO_TRACK_MATCH_THRESHOLD", "0.80"))
YOLO_TRACK_BUFFER_SECONDS = float(os.getenv("YOLO_TRACK_BUFFER_SECONDS", "5.0"))
YOLO_PLAYER_CLASS_IDS = _csv_ints("YOLO_PLAYER_CLASS_IDS", "2")
YOLO_GOALKEEPER_CLASS_IDS = _csv_ints("YOLO_GOALKEEPER_CLASS_IDS", "1")
YOLO_REFEREE_CLASS_IDS = _csv_ints("YOLO_REFEREE_CLASS_IDS", "3")
YOLO_BALL_CLASS_IDS = _csv_ints("YOLO_BALL_CLASS_IDS", "0")

# TrackLab/sn-gamestate and SoccernetGSR Winner run in an isolated Python/CUDA
# environment. A JSON argv avoids shell parsing and keeps paths with spaces safe.
try:
    _gsr_command_json = os.getenv("GSR_RUNNER_COMMAND_JSON", "[]").strip() or "[]"
    GSR_RUNNER_COMMAND = json.loads(_gsr_command_json)
except json.JSONDecodeError as exc:
    raise ValueError("GSR_RUNNER_COMMAND_JSON doit être une liste JSON valide.") from exc
if not isinstance(GSR_RUNNER_COMMAND, list) or not all(
    isinstance(item, str) for item in GSR_RUNNER_COMMAND
):
    raise ValueError("GSR_RUNNER_COMMAND_JSON doit être une liste de chaînes.")
GSR_PRECOMPUTED_RESULT = os.getenv("GSR_PRECOMPUTED_RESULT", "")
GSR_TIMEOUT_SECONDS = int(os.getenv("GSR_TIMEOUT_SECONDS", "43200"))
GSR_FRAME_TOLERANCE_MS = int(os.getenv("GSR_FRAME_TOLERANCE_MS", "120"))
GSR_TRACKING_FPS = float(os.getenv("GSR_TRACKING_FPS", "5.0"))
GSR_BALL_BACKEND = os.getenv("GSR_BALL_BACKEND", "yolo").strip().lower()
if GSR_BALL_BACKEND not in {"none", "heuristic", "yolo"}:
    raise ValueError("GSR_BALL_BACKEND doit valoir none, heuristic ou yolo.")

# ``main_py`` is a reproducible control profile. It mirrors the standalone
# script already validated on the source video, so old environment experiments
# cannot silently change the next comparison run. ``advanced`` restores every
# individually configurable value above.
if YOLO_PROFILE == "main_py":
    ANALYSIS_MIN_YOLO_TRACKING_FPS = 12.5
    YOLO_CONFIDENCE = 0.30
    # Keep the validated player path at 0.30 while admitting small, weaker ball
    # candidates for the dedicated temporal selector.
    YOLO_BALL_CONFIDENCE = 0.12
    YOLO_IMAGE_SIZE = 640
    YOLO_TRACKER = "bytetrack"
    YOLO_TRACK_LOW_CONFIDENCE = 0.30
    YOLO_NEW_TRACK_CONFIDENCE = 0.25
    YOLO_TRACK_MATCH_THRESHOLD = 0.80
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg")
FFPROBE_BINARY = os.getenv("FFPROBE_BINARY", "ffprobe")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
