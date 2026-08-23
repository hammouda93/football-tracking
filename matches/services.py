from __future__ import annotations

import csv
import io
import re
from typing import BinaryIO

from django.db import transaction

from .models import Match, MatchVideo, Player, Team


TIMECODE_RE = re.compile(r"^(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,3}):(?P<s>\d{2})(?:[.,](?P<ms>\d{1,3}))?$")


def parse_timecode(value: str) -> int:
    """Parse MM:SS, HH:MM:SS or raw seconds into milliseconds."""
    value = (value or "").strip()
    if not value:
        raise ValueError("Le timecode est vide.")
    if re.fullmatch(r"\d+(?:[.,]\d+)?", value):
        return int(float(value.replace(",", ".")) * 1000)
    match = TIMECODE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Timecode invalide : {value}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int((match.group("ms") or "0").ljust(3, "0")[:3])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_timecode(milliseconds: int | None) -> str:
    total_ms = max(0, int(milliseconds or 0))
    total_seconds, millis = divmod(total_ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


@transaction.atomic
def create_match_from_upload(cleaned_data: dict) -> Match:
    home_team = Team.objects.create(
        name=cleaned_data["home_team_name"].strip(),
        short_name=cleaned_data["home_team_name"].strip()[:3].upper(),
        primary_color=cleaned_data["home_team_color"],
    )
    away_team = Team.objects.create(
        name=cleaned_data["away_team_name"].strip(),
        short_name=cleaned_data["away_team_name"].strip()[:3].upper(),
        primary_color=cleaned_data["away_team_color"],
    )
    match = Match.objects.create(
        home_team=home_team,
        away_team=away_team,
        competition=cleaned_data.get("competition", ""),
        venue=cleaned_data.get("venue", ""),
        status=Match.Status.UPLOADED,
    )
    video = cleaned_data["video"]
    MatchVideo.objects.create(
        match=match,
        file=video,
        original_name=video.name,
        size_bytes=video.size,
    )
    return match


def import_roster_csv(team: Team, uploaded_file: BinaryIO) -> tuple[int, list[str]]:
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = raw
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "name" not in {name.strip() for name in reader.fieldnames}:
        return 0, ["La colonne obligatoire 'name' est absente."]

    created = 0
    errors: list[str] = []
    valid_positions = {choice for choice, _ in Player.Position.choices}
    for row_number, row in enumerate(reader, start=2):
        name = (row.get("name") or "").strip()
        if not name:
            errors.append(f"Ligne {row_number} : nom vide.")
            continue
        shirt_raw = (row.get("shirt_number") or "").strip()
        try:
            shirt_number = int(shirt_raw) if shirt_raw else None
        except ValueError:
            errors.append(f"Ligne {row_number} : numéro invalide '{shirt_raw}'.")
            continue
        position = (row.get("position") or Player.Position.OTHER).strip().upper()
        if position not in valid_positions:
            position = Player.Position.OTHER
        Player.objects.create(
            team=team,
            name=name,
            shirt_number=shirt_number,
            position=position,
        )
        created += 1
    return created, errors
