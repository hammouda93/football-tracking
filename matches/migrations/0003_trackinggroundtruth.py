from django.db import migrations, models
import django.db.models.deletion
import matches.models


class Migration(migrations.Migration):
    dependencies = [("matches", "0002_match_home_team_cluster")]

    operations = [
        migrations.CreateModel(
            name="TrackingGroundTruth",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to=matches.models.ground_truth_upload_to)),
                ("original_name", models.CharField(max_length=255)),
                ("row_count", models.PositiveIntegerField(default=0)),
                ("frame_count", models.PositiveIntegerField(default=0)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "match",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="tracking_ground_truth",
                        to="matches.match",
                    ),
                ),
            ],
        )
    ]
