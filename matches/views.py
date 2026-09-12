from __future__ import annotations

import csv
import json
import mimetypes
import os
import re

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.db.models import Count, Q
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import content_disposition_header
from django.views.decorators.http import require_GET, require_POST

from .forms import EventReviewForm, MatchUploadForm, PlayerForm, RosterUploadForm
from .models import (
    AnalysisRun,
    Event,
    Match,
    MatchPeriod,
    MatchVideo,
    Player,
    PlayerMatchStat,
    PossessionSegment,
    TeamMatchStat,
    Track,
)
from .services import create_match_from_upload, import_roster_csv, parse_timecode


def _run_mode(run: AnalysisRun | None) -> str:
    if run is None:
        return ""
    mode = str((run.config or {}).get("analysis_mode", "full"))
    return mode if mode in {"prepare", "reference", "sample", "full"} else "full"


def _passing_sample_run(match: Match) -> AnalysisRun | None:
    for run in match.analysis_runs.all()[:25]:
        if _run_mode(run) != "sample":
            continue
        diagnostics = (run.metrics or {}).get("diagnostics") or {}
        terminal = run.status in {AnalysisRun.Status.REVIEW, AnalysisRun.Status.COMPLETED}
        approved = diagnostics.get("verdict") == "pass" or diagnostics.get(
            "manual_approved", False
        )
        configuration_current = not diagnostics.get(
            "periods_changed_since_run", False
        ) and not diagnostics.get("team_mapping_changed_since_run", False)
        if terminal and approved and configuration_current:
            return run
    return None


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
    latest_mode = _run_mode(latest_run)
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
    show_match_results = (
        latest_run is not None
        and latest_mode == "full"
        and latest_run.status in {AnalysisRun.Status.REVIEW, AnalysisRun.Status.COMPLETED}
    )
    team_stats = (
        {
            stat.team_id: stat
            for stat in TeamMatchStat.objects.filter(
                match=match, analysis_run=latest_run
            ).select_related("team")
        }
        if show_match_results
        else {}
    )
    player_stats = (
        PlayerMatchStat.objects.filter(match=match, analysis_run=latest_run).select_related(
            "player", "player__team"
        )
        if show_match_results
        else PlayerMatchStat.objects.none()
    )
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
        "latest_mode": latest_mode,
        "latest_stage_label": (
            (
                "Référence main.py en direct"
                if latest_mode == "reference"
                else "Test joueurs, ballon et jeu effectif"
            )
            if latest_run
            and latest_mode in {"reference", "sample"}
            and latest_run.current_stage == AnalysisRun.Stage.TRACKING
            else (latest_run.get_current_stage_display() if latest_run else "")
        ),
        "sample_diagnostics": (
            (latest_run.metrics or {}).get("diagnostics") or {}
            if latest_mode in {"reference", "sample"}
            else {}
        ),
        "sample_windows": (
            (latest_run.metrics or {}).get("windows") or []
            if latest_mode in {"reference", "sample"}
            else []
        ),
        "sample_previews": (
            (latest_run.metrics or {}).get("previews") or []
            if latest_mode in {"reference", "sample"}
            else []
        ),
        "periods_confirmed": len(periods) == 2 and all(period.confirmed for period in periods),
        "sample_ready": _passing_sample_run(match) is not None,
        "home_team_cluster": match.home_team_cluster,
        "away_team_cluster": (
            Match.TeamCluster.B
            if match.home_team_cluster == Match.TeamCluster.A
            else Match.TeamCluster.A
        ),
        "show_match_results": show_match_results,
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


def _range_file_iterator(
    path: str,
    start: int,
    length: int,
    chunk_size: int = 1024 * 1024,
):
    with open(path, "rb") as source:
        source.seek(start)
        remaining = length
        while remaining > 0:
            chunk = source.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@require_GET
def stream_match_video(request: HttpRequest, pk) -> HttpResponse:
    video = get_object_or_404(MatchVideo, match_id=pk)
    path = video.file.path
    file_size = os.path.getsize(path)
    content_type = mimetypes.guess_type(video.original_name or path)[0] or "video/mp4"
    range_header = request.headers.get("Range", "").strip()
    range_match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)

    if range_header and range_match is None:
        response = HttpResponse(status=416)
        response["Content-Range"] = f"bytes */{file_size}"
        return response

    if range_match is not None:
        start_text, end_text = range_match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        elif end_text:
            suffix_length = min(int(end_text), file_size)
            start = file_size - suffix_length
            end = file_size - 1
        else:
            start, end = 0, file_size - 1
        if start >= file_size or start > end:
            response = HttpResponse(status=416)
            response["Content-Range"] = f"bytes */{file_size}"
            return response
        end = min(end, file_size - 1)
        length = end - start + 1
        response = StreamingHttpResponse(
            _range_file_iterator(path, start, length),
            status=206,
            content_type=content_type,
        )
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Content-Length"] = str(length)
    else:
        response = FileResponse(open(path, "rb"), content_type=content_type)
        response["Content-Length"] = str(file_size)

    response["Accept-Ranges"] = "bytes"
    filename = os.path.basename(video.original_name or path)
    response["Content-Disposition"] = content_disposition_header(False, filename)
    return response


@require_POST
def start_analysis(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    active = match.analysis_runs.filter(
        status__in=[AnalysisRun.Status.QUEUED, AnalysisRun.Status.PROCESSING]
    ).first()
    if active:
        messages.info(request, "Une analyse est déjà en attente ou en cours.")
        return redirect(match)

    mode = request.POST.get("mode", "sample")
    if mode not in {"prepare", "reference", "sample", "full"}:
        messages.error(request, "Mode d’analyse invalide.")
        return redirect(match)
    confirmed_periods = list(match.periods.filter(confirmed=True).order_by("number"))
    if mode in {"reference", "sample", "full"} and len(confirmed_periods) != 2:
        messages.warning(
            request,
            "Confirme d’abord les limites des deux mi-temps avant de lancer ce test.",
        )
        return redirect(match)
    if mode == "full" and _passing_sample_run(match) is None:
        messages.warning(
            request,
            "L’analyse complète est bloquée tant que le test rapide n’est pas validé.",
        )
        return redirect(match)

    run = AnalysisRun.objects.create(
        match=match,
        config={
            "analysis_mode": mode,
            "backend": settings.ANALYSIS_BACKEND,
            "device": settings.ANALYSIS_DEVICE,
            "yolo_profile": settings.YOLO_PROFILE,
            "sample_seconds": settings.ANALYSIS_SAMPLE_SECONDS,
            "quality_max_samples": settings.ANALYSIS_QUALITY_MAX_SAMPLES,
            "tracking_fps": settings.ANALYSIS_TRACKING_FPS,
            "min_yolo_tracking_fps": settings.ANALYSIS_MIN_YOLO_TRACKING_FPS,
            "live_window": settings.ANALYSIS_LIVE_WINDOW,
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
            "home_team_cluster": match.home_team_cluster,
            "sample_window_seconds": 5 if mode == "reference" else 60,
            "sample_windows_per_half": 4 if mode == "reference" else 1,
            "render_clips": mode == "full",
        },
    )
    match.status = Match.Status.QUEUED
    match.save(update_fields=["status", "updated_at"])
    labels = {
        "prepare": "Détection automatique des mi-temps",
        "reference": "Référence main.py de 40 secondes",
        "sample": "Test de validation de 2 minutes",
        "full": "Analyse complète",
    }
    messages.success(request, f"{labels[mode]} · {str(run.pk)[:8]} mis en file.")
    return redirect(match)


@require_POST
def swap_team_clusters(request: HttpRequest, pk) -> HttpResponse:
    match = get_object_or_404(Match, pk=pk)
    match.home_team_cluster = (
        Match.TeamCluster.B
        if match.home_team_cluster == Match.TeamCluster.A
        else Match.TeamCluster.A
    )
    match.save(update_fields=["home_team_cluster", "updated_at"])

    for run in match.analysis_runs.all()[:25]:
        if _run_mode(run) != "sample" or not run.metrics:
            continue
        metrics = dict(run.metrics)
        diagnostics = dict(metrics.get("diagnostics") or {})
        diagnostics["team_mapping_changed_since_run"] = True
        diagnostics["manual_approved"] = False
        metrics["diagnostics"] = diagnostics
        run.metrics = metrics
        run.save(update_fields=["metrics"])

    messages.success(
        request,
        "Correspondance des groupes A/B inversée. Relance la référence 40 s.",
    )
    return redirect(match)


@require_GET
def analysis_status(request: HttpRequest, pk) -> JsonResponse:
    run = get_object_or_404(AnalysisRun.objects.select_related("match"), pk=pk)
    live_progress = (run.metrics or {}).get("live_progress") or {}
    stage_label = run.get_current_stage_display()
    if _run_mode(run) in {"reference", "sample"} and run.current_stage == AnalysisRun.Stage.TRACKING:
        stage_label = (
            "Référence main.py en direct"
            if _run_mode(run) == "reference"
            else "Test joueurs, ballon et jeu effectif"
        )
    return JsonResponse(
        {
            "id": str(run.pk),
            "status": run.status,
            "status_label": run.get_status_display(),
            "stage": run.current_stage,
            "stage_label": stage_label,
            "progress": run.progress,
            "error": run.error_message,
            "match_status": run.match.status,
            "progress_detail": live_progress if run.current_stage == AnalysisRun.Stage.TRACKING else {},
            "live_preview_url": reverse("analysis-live-preview", kwargs={"pk": run.pk}),
        }
    )


@require_GET
def analysis_live_preview(request: HttpRequest, pk) -> HttpResponse:
    run = get_object_or_404(AnalysisRun, pk=pk)
    preview_path = (
        settings.MEDIA_ROOT
        / "matches"
        / str(run.match_id)
        / "live"
        / f"{run.pk}.jpg"
    )
    if not preview_path.is_file():
        return HttpResponse(status=404)
    # Read and close immediately so the worker can atomically replace the next
    # frame on Windows while this response is being sent.
    response = HttpResponse(preview_path.read_bytes(), content_type="image/jpeg")
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    return response


@require_POST
def cancel_analysis(request: HttpRequest, pk) -> HttpResponse:
    run = get_object_or_404(AnalysisRun.objects.select_related("match"), pk=pk)
    if run.status in {AnalysisRun.Status.QUEUED, AnalysisRun.Status.PROCESSING}:
        run.status = AnalysisRun.Status.CANCELLED
        run.save(update_fields=["status"])
        messages.info(request, "L’arrêt de l’analyse a été demandé.")
    return redirect(run.match)


@require_POST
def validate_sample(request: HttpRequest, pk) -> HttpResponse:
    run = get_object_or_404(AnalysisRun.objects.select_related("match"), pk=pk)
    if _run_mode(run) != "sample" or run.status not in {
        AnalysisRun.Status.REVIEW,
        AnalysisRun.Status.COMPLETED,
    }:
        messages.error(request, "Seul un test rapide terminé peut être validé.")
        return redirect(run.match)
    if request.POST.get("confirm") != "yes":
        messages.warning(
            request,
            "Confirme d’abord que tous les aperçus ont été vérifiés.",
        )
        return redirect(run.match)

    metrics = dict(run.metrics or {})
    diagnostics = dict(metrics.get("diagnostics") or {})
    if diagnostics.get("periods_changed_since_run"):
        messages.warning(
            request,
            "Les limites des mi-temps ont changé. Relance le test rapide avant de le valider.",
        )
        return redirect(run.match)
    diagnostics["manual_approved"] = True
    diagnostics["manual_approved_at"] = timezone.now().isoformat()
    metrics["diagnostics"] = diagnostics
    run.metrics = metrics
    run.save(update_fields=["metrics"])
    messages.success(
        request,
        "Test validé manuellement. L’analyse complète est maintenant déverrouillée.",
    )
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
        p1_clock_start = parse_timecode(request.POST.get("p1_clock_start", "00:00"))
        p2_clock_start = parse_timecode(request.POST.get("p2_clock_start", "45:00"))
        if not (p1_start < p1_end <= p2_start < p2_end):
            raise ValueError("Les périodes se chevauchent ou ne sont pas dans l’ordre.")
        if p2_clock_start <= p1_clock_start:
            raise ValueError("L’horloge de la deuxième mi-temps doit suivre la première.")
        duration = getattr(getattr(match, "video", None), "duration_ms", 0)
        if duration and p2_end > duration:
            raise ValueError("La fin de la deuxième période dépasse la durée de la vidéo.")
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect(match)

    values = [
        (1, "1re mi-temps", p1_start, p1_end, p1_clock_start),
        (2, "2e mi-temps", p2_start, p2_end, p2_clock_start),
    ]
    existing_video_bounds = {
        period.number: (period.video_start_ms, period.video_end_ms)
        for period in match.periods.filter(number__in=(1, 2))
    }
    video_bounds_changed = any(
        existing_video_bounds.get(number) != (start, end)
        for number, _label, start, end, _clock_start in values
    )
    for number, label, start, end, clock_start in values:
        MatchPeriod.objects.update_or_create(
            match=match,
            number=number,
            defaults={
                "label": label,
                "video_start_ms": start,
                "video_end_ms": end,
                "match_clock_start_ms": clock_start,
                "match_clock_end_ms": clock_start + (end - start),
                "source": MatchPeriod.Source.MANUAL,
                "confidence": 1.0,
                "confirmed": True,
            },
        )
    if video_bounds_changed:
        for run in match.analysis_runs.all():
            if _run_mode(run) != "sample" or not run.metrics:
                continue
            metrics = dict(run.metrics)
            diagnostics = dict(metrics.get("diagnostics") or {})
            diagnostics["periods_changed_since_run"] = True
            diagnostics["manual_approved"] = False
            metrics["diagnostics"] = diagnostics
            run.metrics = metrics
            run.save(update_fields=["metrics"])
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
