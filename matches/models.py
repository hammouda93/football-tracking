from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.urls import reverse


def match_video_upload_to(instance: "MatchVideo", filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return f"matches/{instance.match_id}/source/{uuid4().hex}{suffix}"


def event_clip_upload_to(instance: "Event", filename: str) -> str:
    return f"matches/{instance.match_id}/clips/{filename}"


def artifact_upload_to(instance: "AnalysisArtifact", filename: str) -> str:
    return f"matches/{instance.analysis_run.match_id}/artifacts/{filename}"


class Team(models.Model):
    name = models.CharField(max_length=160)
    short_name = models.CharField(max_length=12, blank=True)
    primary_color = models.CharField(max_length=7, default="#19C37D")
    secondary_color = models.CharField(max_length=7, default="#FFFFFF")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Player(models.Model):
    class Position(models.TextChoices):
        GK = "GK", "Gardien"
        CB = "CB", "Défenseur central"
        FB = "FB", "Latéral"
        DM = "DM", "Milieu défensif"
        CM = "CM", "Milieu central"
        AM = "AM", "Milieu offensif"
        W = "W", "Ailier"
        ST = "ST", "Attaquant"
        OTHER = "OTHER", "Autre"

    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="players")
    name = models.CharField(max_length=160)
    shirt_number = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(99)],
    )
    position = models.CharField(max_length=12, choices=Position.choices, default=Position.OTHER)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["team__name", "shirt_number", "name"]

    def __str__(self) -> str:
        number = f"#{self.shirt_number} " if self.shirt_number is not None else ""
        return f"{number}{self.name}"


class Match(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Brouillon"
        UPLOADED = "uploaded", "Vidéo importée"
        QUEUED = "queued", "En attente"
        PROCESSING = "processing", "Analyse en cours"
        REVIEW = "review", "À valider"
        COMPLETED = "completed", "Terminé"
        FAILED = "failed", "Échec"

    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    home_team = models.ForeignKey(Team, on_delete=models.PROTECT, related_name="home_matches")
    away_team = models.ForeignKey(Team, on_delete=models.PROTECT, related_name="away_matches")
    competition = models.CharField(max_length=160, blank=True)
    venue = models.CharField(max_length=160, blank=True)
    kickoff_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.DRAFT)
    home_score = models.PositiveSmallIntegerField(default=0)
    away_score = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.home_team} – {self.away_team}"

    def get_absolute_url(self) -> str:
        return reverse("match-detail", kwargs={"pk": self.pk})


class MatchVideo(models.Model):
    class QualityGrade(models.TextChoices):
        UNKNOWN = "unknown", "Non évaluée"
        A = "A", "A — Analyse complète"
        B = "B", "B — Bonne"
        C = "C", "C — Limitée"
        REJECT = "reject", "Insuffisante"

    match = models.OneToOneField(Match, on_delete=models.CASCADE, related_name="video")
    file = models.FileField(upload_to=match_video_upload_to)
    original_name = models.CharField(max_length=255)
    size_bytes = models.PositiveBigIntegerField(default=0)
    duration_ms = models.PositiveBigIntegerField(default=0)
    fps = models.FloatField(default=0)
    width = models.PositiveIntegerField(default=0)
    height = models.PositiveIntegerField(default=0)
    codec = models.CharField(max_length=64, blank=True)
    quality_grade = models.CharField(
        max_length=16,
        choices=QualityGrade.choices,
        default=QualityGrade.UNKNOWN,
    )
    quality_score = models.FloatField(default=0)
    quality_metrics = models.JSONField(default=dict, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.original_name


class MatchPeriod(models.Model):
    class Source(models.TextChoices):
        AUTO = "auto", "Automatique"
        MANUAL = "manual", "Manuel"

    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="periods")
    number = models.PositiveSmallIntegerField()
    label = models.CharField(max_length=32)
    video_start_ms = models.PositiveBigIntegerField()
    video_end_ms = models.PositiveBigIntegerField()
    match_clock_start_ms = models.PositiveBigIntegerField(default=0)
    match_clock_end_ms = models.PositiveBigIntegerField(default=2_700_000)
    source = models.CharField(max_length=12, choices=Source.choices, default=Source.AUTO)
    confidence = models.FloatField(default=0)
    confirmed = models.BooleanField(default=False)

    class Meta:
        ordering = ["number"]
        constraints = [
            models.UniqueConstraint(fields=["match", "number"], name="unique_match_period")
        ]

    def __str__(self) -> str:
        return f"{self.match} — {self.label}"


class AnalysisRun(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "En attente"
        PROCESSING = "processing", "En cours"
        REVIEW = "review", "À valider"
        COMPLETED = "completed", "Terminé"
        FAILED = "failed", "Échec"
        CANCELLED = "cancelled", "Annulé"

    class Stage(models.TextChoices):
        QUEUED = "queued", "Mise en file"
        PROBE = "probe", "Lecture vidéo"
        QUALITY = "quality", "Contrôle qualité"
        PERIODS = "periods", "Détection des périodes"
        TRACKING = "tracking", "Tracking joueurs et ballon"
        CAMERA = "camera", "Calibration caméra"
        POSSESSION = "possession", "Ball in play et possession"
        EVENTS = "events", "Détection des actions"
        STATS = "stats", "Calcul des statistiques"
        CLIPS = "clips", "Découpage vidéo"
        REPORT = "report", "Rapport final"
        DONE = "done", "Terminé"

    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="analysis_runs")
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.QUEUED)
    current_stage = models.CharField(max_length=24, choices=Stage.choices, default=Stage.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0, validators=[MaxValueValidator(100)])
    config = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Analyse {self.match} ({self.get_status_display()})"


class AnalysisArtifact(models.Model):
    class Kind(models.TextChoices):
        QUALITY = "quality", "Qualité vidéo"
        PERIODS = "periods", "Périodes"
        TRACKING = "tracking", "Tracking"
        CAMERA = "camera", "Caméra"
        EVENTS = "events", "Événements"
        REPORT = "report", "Rapport"
        ANNOTATED_VIDEO = "annotated_video", "Vidéo annotée"

    analysis_run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="artifacts")
    kind = models.CharField(max_length=32, choices=Kind.choices)
    file = models.FileField(upload_to=artifact_upload_to)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Track(models.Model):
    class Role(models.TextChoices):
        PLAYER = "player", "Joueur"
        GOALKEEPER = "goalkeeper", "Gardien"
        REFEREE = "referee", "Arbitre"
        OTHER = "other", "Autre"

    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="tracks")
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.CASCADE,
        related_name="tracks",
        null=True,
        blank=True,
    )
    track_uid = models.CharField(max_length=80)
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.PLAYER)
    team = models.ForeignKey(Team, on_delete=models.SET_NULL, null=True, blank=True)
    player = models.ForeignKey(Player, on_delete=models.SET_NULL, null=True, blank=True)
    predicted_shirt_number = models.SmallIntegerField(null=True, blank=True)
    identity_confidence = models.FloatField(default=0)
    video_start_ms = models.PositiveBigIntegerField(default=0)
    video_end_ms = models.PositiveBigIntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["video_start_ms"]
        constraints = [
            models.UniqueConstraint(
                fields=["analysis_run", "track_uid"],
                name="unique_analysis_track",
            )
        ]


class PossessionSegment(models.Model):
    class State(models.TextChoices):
        CONTROLLED = "controlled", "Contrôlé"
        CONTESTED = "contested", "Disputé"
        LOOSE = "loose", "Ballon libre"
        OUT = "out", "Ballon sorti"
        UNKNOWN = "unknown", "Inconnu"

    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="possessions")
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.CASCADE,
        related_name="possessions",
        null=True,
        blank=True,
    )
    period = models.ForeignKey(MatchPeriod, on_delete=models.CASCADE, related_name="possessions")
    team = models.ForeignKey(Team, on_delete=models.SET_NULL, null=True, blank=True)
    player = models.ForeignKey(Player, on_delete=models.SET_NULL, null=True, blank=True)
    owner_track = models.ForeignKey(
        Track,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="possession_segments",
    )
    state = models.CharField(max_length=16, choices=State.choices)
    video_start_ms = models.PositiveBigIntegerField()
    video_end_ms = models.PositiveBigIntegerField()
    match_start_ms = models.PositiveBigIntegerField()
    match_end_ms = models.PositiveBigIntegerField()
    confidence = models.FloatField(default=0)
    reviewed = models.BooleanField(default=False)

    class Meta:
        ordering = ["video_start_ms"]


class Event(models.Model):
    class Type(models.TextChoices):
        PASS = "pass", "Passe"
        CARRY = "carry", "Conduite"
        DRIBBLE = "dribble", "Dribble"
        HEADER = "header", "Tête"
        HIGH_PASS = "high_pass", "Passe haute"
        CROSS = "cross", "Centre"
        THROW_IN = "throw_in", "Touche"
        SHOT = "shot", "Tir"
        GOAL = "goal", "But"
        TACKLE = "tackle", "Tacle"
        BLOCK = "block", "Blocage"
        INTERCEPTION = "interception", "Interception"
        DUEL = "duel", "Duel"
        AERIAL_DUEL = "aerial_duel", "Duel aérien"
        RECOVERY = "recovery", "Récupération"
        LOSS = "loss", "Perte"
        FOUL = "foul", "Faute"
        OFFSIDE = "offside", "Hors-jeu"
        FREE_KICK = "free_kick", "Coup franc"
        CORNER = "corner", "Corner"
        PENALTY = "penalty", "Penalty"
        SAVE = "save", "Arrêt"
        OUT = "out", "Sortie du ballon"
        OTHER = "other", "Autre"

    class Outcome(models.TextChoices):
        SUCCESS = "success", "Réussi"
        FAILURE = "failure", "Raté"
        NEUTRAL = "neutral", "Neutre"
        UNKNOWN = "unknown", "Inconnu"

    class ReviewStatus(models.TextChoices):
        AUTO_ACCEPTED = "auto_accepted", "Accepté automatiquement"
        PENDING = "pending", "À valider"
        VALIDATED = "validated", "Validé"
        CORRECTED = "corrected", "Corrigé"
        REJECTED = "rejected", "Rejeté"

    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="events")
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.CASCADE,
        related_name="events",
        null=True,
        blank=True,
    )
    period = models.ForeignKey(MatchPeriod, on_delete=models.CASCADE, related_name="events")
    event_type = models.CharField(max_length=32, choices=Type.choices)
    subtype = models.CharField(max_length=64, blank=True)
    team = models.ForeignKey(Team, on_delete=models.SET_NULL, null=True, blank=True)
    player = models.ForeignKey(
        Player,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="match_events",
    )
    recipient = models.ForeignKey(
        Player,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="received_events",
    )
    actor_track = models.ForeignKey(
        Track,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="actor_events",
    )
    recipient_track = models.ForeignKey(
        Track,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recipient_events",
    )
    video_time_ms = models.PositiveBigIntegerField()
    match_time_ms = models.PositiveBigIntegerField()
    start_x = models.FloatField(null=True, blank=True)
    start_y = models.FloatField(null=True, blank=True)
    end_x = models.FloatField(null=True, blank=True)
    end_y = models.FloatField(null=True, blank=True)
    outcome = models.CharField(max_length=16, choices=Outcome.choices, default=Outcome.UNKNOWN)
    confidence = models.FloatField(default=0)
    visibility = models.CharField(max_length=24, default="unknown")
    qualifiers = models.JSONField(default=list, blank=True)
    review_status = models.CharField(
        max_length=24,
        choices=ReviewStatus.choices,
        default=ReviewStatus.PENDING,
    )
    source = models.CharField(max_length=32, default="ai")
    model_version = models.CharField(max_length=64, default="baseline-v1")
    clip = models.FileField(upload_to=event_clip_upload_to, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["video_time_ms"]
        indexes = [
            models.Index(fields=["match", "event_type"], name="event_match_type_idx"),
            models.Index(fields=["match", "review_status"], name="event_match_review_idx"),
            models.Index(fields=["match", "player"], name="event_match_player_idx"),
        ]


class TeamMatchStat(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="team_stats")
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.SET_NULL,
        related_name="team_stats",
        null=True,
        blank=True,
    )
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="match_stats")
    minutes_played = models.FloatField(default=0)
    metrics = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["match", "team"], name="unique_team_match_stat")
        ]


class PlayerMatchStat(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="player_stats")
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.SET_NULL,
        related_name="player_stats",
        null=True,
        blank=True,
    )
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="match_stats")
    minutes_played = models.FloatField(default=0)
    metrics = models.JSONField(default=dict)
    heatmap = models.JSONField(default=list, blank=True)
    touchmap = models.JSONField(default=list, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["match", "player"], name="unique_player_match_stat")
        ]
