from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("matches", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="match",
            name="home_team_cluster",
            field=models.CharField(
                choices=[("A", "Groupe A"), ("B", "Groupe B")],
                default="B",
                max_length=1,
            ),
        ),
    ]
