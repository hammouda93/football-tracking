from __future__ import annotations

from pathlib import Path

from django import forms
from django.core.validators import RegexValidator

from .models import Event, Player


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


class MatchUploadForm(forms.Form):
    home_team_name = forms.CharField(label="Équipe à domicile", max_length=160)
    home_team_color = forms.CharField(
        label="Couleur principale domicile",
        max_length=7,
        initial="#12B76A",
        validators=[RegexValidator(r"^#[0-9A-Fa-f]{6}$", "Couleur hexadécimale invalide.")],
        widget=forms.TextInput(attrs={"type": "color"}),
    )
    away_team_name = forms.CharField(label="Équipe à l’extérieur", max_length=160)
    away_team_color = forms.CharField(
        label="Couleur principale extérieur",
        max_length=7,
        initial="#F04438",
        validators=[RegexValidator(r"^#[0-9A-Fa-f]{6}$", "Couleur hexadécimale invalide.")],
        widget=forms.TextInput(attrs={"type": "color"}),
    )
    competition = forms.CharField(label="Compétition", max_length=160, required=False)
    venue = forms.CharField(label="Stade", max_length=160, required=False)
    video = forms.FileField(
        label="Vidéo complète du match",
        widget=forms.ClearableFileInput(attrs={"accept": "video/*"}),
    )

    def clean_video(self):
        video = self.cleaned_data["video"]
        extension = Path(video.name).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise forms.ValidationError(
                "Format non pris en charge. Utilise MP4, MOV, MKV, AVI ou M4V."
            )
        return video


class PlayerForm(forms.ModelForm):
    class Meta:
        model = Player
        fields = ["name", "shirt_number", "position"]


class RosterUploadForm(forms.Form):
    team_id = forms.IntegerField(widget=forms.HiddenInput())
    roster = forms.FileField(
        label="Effectif CSV",
        help_text="Colonnes : name, shirt_number, position",
        widget=forms.ClearableFileInput(attrs={"accept": ".csv,text/csv"}),
    )


class EventReviewForm(forms.ModelForm):
    class Meta:
        model = Event
        fields = [
            "event_type",
            "team",
            "player",
            "recipient",
            "outcome",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]

    def __init__(self, *args, match=None, **kwargs):
        super().__init__(*args, **kwargs)
        if match is not None:
            team_ids = [match.home_team_id, match.away_team_id]
            self.fields["team"].queryset = self.fields["team"].queryset.filter(id__in=team_ids)
            players = Player.objects.filter(team_id__in=team_ids).order_by(
                "team__name", "shirt_number", "name"
            )
            self.fields["player"].queryset = players
            self.fields["recipient"].queryset = players
