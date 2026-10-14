from django.core.management.base import BaseCommand
from django.conf import settings

from forecasting.train import run, _parse_filters
from pricing.services import register_from_training


class Command(BaseCommand):
    help = "Train a model, register it, and optionally activate it if it beats the current one."

    def add_arguments(self, p):
        p.add_argument("--data", required=True)
        p.add_argument("--date-col")
        p.add_argument("--price-col")
        p.add_argument("--filter", action="append")
        p.add_argument("--activate", action="store_true", help="Activate unconditionally")
        p.add_argument("--activate-if-better", action="store_true",
                       help="Activate only if backtest MAPE beats the active model (champion/challenger)")

    def handle(self, *a, **o):
        from pricing.models import ModelVersion
        result = run(o["data"], settings.MODELS_DIR, date_col=o["date_col"], price_col=o["price_col"],
                     filters=_parse_filters(o["filter"]))
        champion = ModelVersion.objects.filter(is_active=True).first()
        new_mape = result["meta"]["reference"]["backtest_mape"]
        activate = o["activate"] or (o["activate_if_better"] and (champion is None or new_mape < champion.backtest_mape))
        mv = register_from_training(result, activate=activate)
        self.stdout.write(self.style.SUCCESS(
            f"Registered {mv} | backtest MAPE {new_mape:.2f}%"
            + (f" vs champion {champion.backtest_mape:.2f}%" if champion else "")
            + (" -> ACTIVATED" if activate else " -> not activated")))
