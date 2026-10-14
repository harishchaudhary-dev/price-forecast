from django.contrib import admin, messages

from . import services
from .models import ActualPrice, ForecastLog, ModelVersion, MonitoringRun


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "description", "train_end", "backtest_mape", "is_active", "created_at")
    list_filter = ("is_active", "model_name")
    readonly_fields = ("metrics",)
    actions = ["activate"]

    @admin.action(description="Activate selected model (one only)")
    def activate(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model.", messages.ERROR)
            return
        mv = queryset.first()
        mv.activate()
        services.notify_fastapi_reload(mv.version)
        self.message_user(request, f"Activated {mv.version}")


@admin.register(ForecastLog)
class ForecastLogAdmin(admin.ModelAdmin):
    list_display = ("target_month", "predicted_price", "lower_95", "upper_95", "model_version", "created_at")
    list_filter = ("model_version",)


@admin.register(ActualPrice)
class ActualPriceAdmin(admin.ModelAdmin):
    list_display = ("month", "price", "source")


@admin.register(MonitoringRun)
class MonitoringRunAdmin(admin.ModelAdmin):
    list_display = ("created_at", "model_version", "retrain_recommended")
