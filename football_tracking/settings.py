from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _csv_ints(name: str, default: str = "") -> list[int]:
    return [
        int(value.strip())
        for value in os.getenv(name, default).split(",")
        if value.strip()
    ]

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
ANALYSIS_SAMPLE_SECONDS = float(os.getenv("ANALYSIS_SAMPLE_SECONDS", "1.0"))
ANALYSIS_QUALITY_MAX_SAMPLES = int(os.getenv("ANALYSIS_QUALITY_MAX_SAMPLES", "360"))
ANALYSIS_TRACKING_FPS = float(os.getenv("ANALYSIS_TRACKING_FPS", "10.0"))
ANALYSIS_MIN_YOLO_TRACKING_FPS = float(
    os.getenv("ANALYSIS_MIN_YOLO_TRACKING_FPS", "8.0")
)
ANALYSIS_DEVICE = os.getenv("ANALYSIS_DEVICE", "cpu")
_yolo_model_path = Path(
    os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "football-players.pt"))
)
YOLO_MODEL_PATH = str(
    _yolo_model_path if _yolo_model_path.is_absolute() else BASE_DIR / _yolo_model_path
)
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.30"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "1280"))
YOLO_PLAYER_CLASS_IDS = _csv_ints("YOLO_PLAYER_CLASS_IDS", "2")
YOLO_GOALKEEPER_CLASS_IDS = _csv_ints("YOLO_GOALKEEPER_CLASS_IDS")
YOLO_REFEREE_CLASS_IDS = _csv_ints("YOLO_REFEREE_CLASS_IDS", "3")
YOLO_BALL_CLASS_IDS = _csv_ints("YOLO_BALL_CLASS_IDS", "0")
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg")
FFPROBE_BINARY = os.getenv("FFPROBE_BINARY", "ffprobe")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
