import json

from django.core.management.base import BaseCommand

from pricing.services import run_monitoring


class Command(BaseCommand):
    help = "Score served forecasts against actuals and test for data drift. Schedule monthly (cron/Celery beat)."

    def handle(self, *a, **o):
        run = run_monitoring()
        self.stdout.write(json.dumps(run.result, indent=2))
        if run.retrain_recommended:
            self.stdout.write(self.style.WARNING(
                "Retrain recommended. Run train_model --activate-if-better with refreshed data."))
