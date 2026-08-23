from django.urls import path

from . import views


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("matches/upload/", views.upload_match, name="match-upload"),
    path("matches/<uuid:pk>/", views.match_detail, name="match-detail"),
    path("matches/<uuid:pk>/analyse/", views.start_analysis, name="match-start-analysis"),
    path("matches/<uuid:pk>/periods/", views.update_periods, name="match-update-periods"),
    path("matches/<uuid:pk>/players/add/", views.add_player, name="match-add-player"),
    path("matches/<uuid:pk>/roster/import/", views.import_roster, name="match-import-roster"),
    path("matches/<uuid:pk>/export/events.csv", views.export_events_csv, name="match-export-events"),
    path("matches/<uuid:pk>/export/report.json", views.export_report_json, name="match-export-report"),
    path("analysis/<uuid:pk>/status/", views.analysis_status, name="analysis-status"),
    path("analysis/<uuid:pk>/cancel/", views.cancel_analysis, name="analysis-cancel"),
    path("events/<int:pk>/review/", views.review_event, name="event-review"),
    path("tracks/<int:pk>/assign/", views.assign_track, name="track-assign"),
]
