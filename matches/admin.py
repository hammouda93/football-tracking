from django.contrib import admin

from .models import (
    AnalysisArtifact,
    AnalysisRun,
    Event,
    Match,
    MatchPeriod,
    MatchVideo,
    Player,
    PlayerMatchStat,
    PossessionSegment,
    Team,
    TeamMatchStat,
    Track,
)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("name", "short_name", "primary_color")
    search_fields = ("name", "short_name")


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("name", "team", "shirt_number", "position", "active")
    list_filter = ("team", "position", "active")
    search_fields = ("name",)


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ("home_team", "away_team", "competition", "status", "created_at")
    list_filter = ("status", "competition")


admin.site.register(MatchVideo)
admin.site.register(MatchPeriod)
admin.site.register(AnalysisRun)
admin.site.register(AnalysisArtifact)
admin.site.register(Track)
admin.site.register(PossessionSegment)
admin.site.register(Event)
admin.site.register(TeamMatchStat)
admin.site.register(PlayerMatchStat)
