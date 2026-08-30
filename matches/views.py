from __future__ import annotations

import csv
import json

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from .forms import EventReviewForm, MatchUploadForm, PlayerForm, RosterUploadForm
from .models import (
    AnalysisRun,
    Event,
    Match,
    MatchPeriod,
    Player,
    PlayerMatchStat,
    PossessionSegment,
    TeamMatchStat,
    Track,
)
from .services import create_match_from_upload, import_roster_csv, parse_timecode


def dashboard(request: HttpRequest) -> HttpResponse:
    matches = (
        Match.objects.select_related("home_team", "away_team")
        .annotate(event_count=Count("events"), pending_count=Count("events", filter=Q(events__review_status="pending")))
        .all()
    )
    summary = {
        "matches": matches.count(),
        "processing": matches.filter(status=Match.Status.PROCESSING).count(),
        "review": matches.filter(status=Match.Status.REVIEW).count(),
        "events": Event.objects.exclude(review_status=Event.ReviewStatus.REJECTED).count(),
    }
    return render(request, "matches/dashboard.html", {"matches": matches, "summary": summary})


def upload_match(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = MatchUploadForm(request.POST, request.FILES)
        if form.is_valid():
            match = create_match_from_upload(form.cleaned_data)
            messages.success(
                request,
                "Le match a été importé. Ajoute les effectifs puis lance l’analyse.",
            )
            return redirect(match)
    else:
        form = MatchUploadForm()
    return render(request, "matches/upload.html", {"form": form})


def match_detail(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(
        Match.objects.select_related("home_team", "away_team", "video"),
        pk=pk,
    )
    latest_run = match.analysis_runs.first()
    events = match.events.select_related(
        "period", "team", "player", "recipient", "actor_track", "recipient_track"
    )
    if latest_run is not None:
        events = events.filter(analysis_run=latest_run)
    event_type = request.GET.get("event_type", "")
    review_status = request.GET.get("review_status", "")
    team_id = request.GET.get("team", "")
    if event_type:
        events = events.filter(event_type=event_type)
    if review_status:
        events = events.filter(review_status=review_status)
    if team_id:
        events = events.filter(team_id=team_id)

    periods = list(match.periods.all())
    team_stats = {
        stat.team_id: stat
        for stat in TeamMatchStat.objects.filter(match=match).select_related("team")
    }
    player_stats = PlayerMatchStat.objects.filter(match=match).select_related("player", "player__team")
    tracks = (
        Track.objects.filter(analysis_run=latest_run)
        .select_related("team", "player")
        .order_by("team__name", "predicted_shirt_number", "track_uid")
        if latest_run
        else Track.objects.none()
    )
    context = {
        "match": match,
        "periods": periods,
        "latest_run": latest_run,
        "events": events[:500],
        "event_types": Event.Type.choices,
        "review_statuses": Event.ReviewStatus.choices,
        "team_stats": team_stats,
        "home_stats": team_stats.get(match.home_team_id),
        "away_stats": team_stats.get(match.away_team_id),
        "player_stats": player_stats,
        "tracks": tracks[:300],
        "unassigned_track_count": tracks.filter(player__isnull=True).count(),
        "home_players": match.home_team.players.filter(active=True),
        "away_players": match.away_team.players.filter(active=True),
        "player_form": PlayerForm(),
        "roster_form": RosterUploadForm(),
    }
    return render(request, "matches/detail.html", context)


@require_POST
def start_analysis(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    active = match.analysis_runs.filter(
        status__in=[AnalysisRun.Status.QUEUED, AnalysisRun.Status.PROCESSING]
    ).first()
    if active:
        messages.info(request, "Une analyse est déjà en attente ou en cours.")
        return redirect(match)

    run = AnalysisRun.objects.create(
        match=match,
        config={
            "backend": settings.ANALYSIS_BACKEND,
            "device": settings.ANALYSIS_DEVICE,
            "sample_seconds": settings.ANALYSIS_SAMPLE_SECONDS,
            "quality_max_samples": settings.ANALYSIS_QUALITY_MAX_SAMPLES,
            "tracking_fps": settings.ANALYSIS_TRACKING_FPS,
            "yolo_model_path": settings.YOLO_MODEL_PATH,
            "yolo_confidence": settings.YOLO_CONFIDENCE,
            "yolo_image_size": settings.YOLO_IMAGE_SIZE,
            "render_clips": True,
        },
    )
    match.status = Match.Status.QUEUED
    match.save(update_fields=["status", "updated_at"])
    messages.success(request, f"Analyse {str(run.pk)[:8]} mise en file.")
    return redirect(match)


@require_GET
def analysis_status(request: HttpRequest, pk) -> JsonResponse:
    run = get_object_or_404(AnalysisRun.objects.select_related("match"), pk=pk)
    return JsonResponse(
        {
            "id": str(run.pk),
            "status": run.status,
            "status_label": run.get_status_display(),
            "stage": run.current_stage,
            "stage_label": run.get_current_stage_display(),
            "progress": run.progress,
            "error": run.error_message,
            "match_status": run.match.status,
        }
    )


@require_POST
def cancel_analysis(request: HttpRequest, pk) -> HttpResponse:
    run = get_object_or_404(AnalysisRun.objects.select_related("match"), pk=pk)
    if run.status in {AnalysisRun.Status.QUEUED, AnalysisRun.Status.PROCESSING}:
        run.status = AnalysisRun.Status.CANCELLED
        run.save(update_fields=["status"])
        messages.info(request, "L’arrêt de l’analyse a été demandé.")
    return redirect(run.match)


@require_POST
@transaction.atomic
def update_periods(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    try:
        p1_start = parse_timecode(request.POST.get("p1_start", ""))
        p1_end = parse_timecode(request.POST.get("p1_end", ""))
        p2_start = parse_timecode(request.POST.get("p2_start", ""))
        p2_end = parse_timecode(request.POST.get("p2_end", ""))
        if not (p1_start < p1_end <= p2_start < p2_end):
            raise ValueError("Les périodes se chevauchent ou ne sont pas dans l’ordre.")
        duration = getattr(getattr(match, "video", None), "duration_ms", 0)
        if duration and p2_end > duration:
            raise ValueError("La fin de la deuxième période dépasse la durée de la vidéo.")
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect(match)

    values = [
        (1, "1re mi-temps", p1_start, p1_end, 0, 2_700_000),
        (2, "2e mi-temps", p2_start, p2_end, 2_700_000, 5_400_000),
    ]
    for number, label, start, end, clock_start, clock_end in values:
        MatchPeriod.objects.update_or_create(
            match=match,
            number=number,
            defaults={
                "label": label,
                "video_start_ms": start,
                "video_end_ms": end,
                "match_clock_start_ms": clock_start,
                "match_clock_end_ms": clock_end,
                "source": MatchPeriod.Source.MANUAL,
                "confidence": 1.0,
                "confirmed": True,
            },
        )
    messages.success(request, "Les limites des deux mi-temps ont été confirmées.")
    return redirect(match)


@require_POST
def add_player(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    team_id = request.POST.get("team_id")
    if str(team_id) not in {str(match.home_team_id), str(match.away_team_id)}:
        messages.error(request, "Équipe invalide.")
        return redirect(match)
    form = PlayerForm(request.POST)
    if form.is_valid():
        player = form.save(commit=False)
        player.team_id = team_id
        player.save()
        messages.success(request, f"{player} ajouté à l’effectif.")
    else:
        messages.error(request, "Impossible d’ajouter le joueur. Vérifie les informations.")
    return redirect(f"{match.get_absolute_url()}#rosters")


@require_POST
def import_roster(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    form = RosterUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Fichier CSV invalide.")
        return redirect(f"{match.get_absolute_url()}#rosters")
    team_id = form.cleaned_data["team_id"]
    if team_id not in {match.home_team_id, match.away_team_id}:
        messages.error(request, "Équipe invalide.")
        return redirect(match)
    team = match.home_team if team_id == match.home_team_id else match.away_team
    created, errors = import_roster_csv(team, form.cleaned_data["roster"])
    if created:
        messages.success(request, f"{created} joueur(s) importé(s) pour {team.name}.")
    for error in errors[:5]:
        messages.warning(request, error)
    return redirect(f"{match.get_absolute_url()}#rosters")


@require_POST
def review_event(request: HttpRequest, pk: int) -> HttpResponse:
    event = get_object_or_404(Event.objects.select_related("match"), pk=pk)
    action = request.POST.get("action")
    if action == "accept":
        event.review_status = Event.ReviewStatus.VALIDATED
        event.save(update_fields=["review_status", "updated_at"])
    elif action == "reject":
        event.review_status = Event.ReviewStatus.REJECTED
        event.save(update_fields=["review_status", "updated_at"])
    elif action == "correct":
        form = EventReviewForm(request.POST, instance=event, match=event.match)
        if form.is_valid():
            corrected = form.save(commit=False)
            corrected.review_status = Event.ReviewStatus.CORRECTED
            corrected.source = "human"
            corrected.save()
        else:
            messages.error(request, "La correction de l’événement est invalide.")
    return redirect(f"{event.match.get_absolute_url()}#events")


@require_POST
@transaction.atomic
def assign_track(request: HttpRequest, pk: int) -> HttpResponse:
    track = get_object_or_404(
        Track.objects.select_related("match", "team", "analysis_run"),
        pk=pk,
    )
    previous_player = track.player
    player_id = request.POST.get("player_id", "").strip()
    player = None
    if player_id:
        allowed_team_ids = {track.match.home_team_id, track.match.away_team_id}
        player = get_object_or_404(Player, pk=player_id, team_id__in=allowed_team_ids)
        if track.team_id and player.team_id != track.team_id:
            messages.error(request, "Ce joueur n’appartient pas à l’équipe estimée pour cette piste.")
            return redirect(f"{track.match.get_absolute_url()}#identity")
    track.player = player
    track.identity_confidence = 1.0 if player else 0.0
    track.save(update_fields=["player", "identity_confidence"])
    Event.objects.filter(actor_track=track).update(player=player)
    Event.objects.filter(recipient_track=track).update(recipient=player)
    PossessionSegment.objects.filter(owner_track=track).update(player=player)
    if previous_player and previous_player != player:
        _refresh_assigned_player_stats(track.match, previous_player)
    _refresh_assigned_player_stats(track.match, player)
    messages.success(
        request,
        f"Piste {track.track_uid} {'assignée à ' + player.name if player else 'désassignée'}.",
    )
    return redirect(f"{track.match.get_absolute_url()}#identity")


def _refresh_assigned_player_stats(match: Match, player) -> None:
    if player is None:
        return
    metrics: dict[str, float] = {}
    heatmap: list = []
    minutes = 0.0
    for track in match.tracks.filter(player=player):
        for key, value in (track.metadata.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and key not in {"pass_accuracy_pct", "possession_pct"}:
                metrics[key] = metrics.get(key, 0) + value
        heatmap.extend(track.metadata.get("points") or [])
        minutes += max(0, track.video_end_ms - track.video_start_ms) / 60_000
    passes = metrics.get("passes", 0)
    metrics["pass_accuracy_pct"] = (
        round(100 * metrics.get("passes_completed", 0) / passes, 2) if passes else 0
    )
    if not match.tracks.filter(player=player).exists():
        PlayerMatchStat.objects.filter(match=match, player=player).delete()
        return
    PlayerMatchStat.objects.update_or_create(
        match=match,
        player=player,
        defaults={
            "analysis_run": match.analysis_runs.first(),
            "minutes_played": round(minutes, 2),
            "metrics": metrics,
            "heatmap": heatmap[:5_000],
        },
    )


@require_GET
def export_events_csv(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    latest_run = match.analysis_runs.first()
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="events-{match.pk}.csv"'
    response.write("\ufeff")
    writer = csv.writer(response)
    writer.writerow(
        [
            "period",
            "match_time_ms",
            "video_time_ms",
            "type",
            "team",
            "player",
            "recipient",
            "outcome",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "confidence",
            "review_status",
            "qualifiers",
        ]
    )
    event_queryset = match.events.select_related("period", "team", "player", "recipient")
    if latest_run is not None:
        event_queryset = event_queryset.filter(analysis_run=latest_run)
    for event in event_queryset:
        writer.writerow(
            [
                event.period.number,
                event.match_time_ms,
                event.video_time_ms,
                event.event_type,
                event.team.name if event.team else "",
                event.player.name if event.player else "",
                event.recipient.name if event.recipient else "",
                event.outcome,
                event.start_x,
                event.start_y,
                event.end_x,
                event.end_y,
                round(event.confidence, 4),
                event.review_status,
                "|".join(event.qualifiers),
            ]
        )
    return response


@require_GET
def export_report_json(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match.objects.select_related("home_team", "away_team"), pk=pk)
    latest_run = match.analysis_runs.first()
    event_queryset = match.events.exclude(review_status=Event.ReviewStatus.REJECTED)
    if latest_run is not None:
        event_queryset = event_queryset.filter(analysis_run=latest_run)
    payload = {
        "schema": "sportsbase-football-tracking/0.1",
        "match": {
            "id": str(match.pk),
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
            "score": [match.home_score, match.away_score],
            "competition": match.competition,
            "venue": match.venue,
        },
        "periods": list(
            match.periods.values(
                "number",
                "label",
                "video_start_ms",
                "video_end_ms",
                "match_clock_start_ms",
                "match_clock_end_ms",
                "confidence",
                "confirmed",
            )
        ),
        "team_stats": list(match.team_stats.values("team_id", "minutes_played", "metrics")),
        "player_stats": list(match.player_stats.values("player_id", "minutes_played", "metrics")),
        "events": list(
            event_queryset.values(
                "period_id",
                "event_type",
                "team_id",
                "player_id",
                "recipient_id",
                "video_time_ms",
                "match_time_ms",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "outcome",
                "confidence",
                "review_status",
                "qualifiers",
            )
        ),
    }
    response = HttpResponse(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        content_type="application/json; charset=utf-8",
    )
    response["Content-Disposition"] = f'attachment; filename="report-{match.pk}.json"'
    return response
