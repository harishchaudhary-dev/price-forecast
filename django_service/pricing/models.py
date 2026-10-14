from django.db import models, transaction


class ModelVersion(models.Model):
    """Registry of trained model bundles. Exactly one is active at a time."""
    version = models.CharField(max_length=32, unique=True)
    model_name = models.CharField(max_length=64)
    description = models.CharField(max_length=128)
    bundle_path = models.CharField(max_length=512)
    train_start = models.CharField(max_length=7)
    train_end = models.CharField(max_length=7)
    backtest_mape = models.FloatField()
    metrics = models.JSONField(default=dict)
    is_active = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["is_active"], condition=models.Q(is_active=True),
                                               name="single_active_model")]

    def __str__(self):
        return f"{self.version} {self.description}{' [ACTIVE]' if self.is_active else ''}"

    @transaction.atomic
    def activate(self):
        ModelVersion.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        self.is_active = True
        self.save(update_fields=["is_active"])


class ForecastLog(models.Model):
    """Every forecast served is stored, so it can be scored once actuals arrive."""
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name="forecasts")
    target_month = models.DateField()
    predicted_price = models.FloatField()
    lower_95 = models.FloatField()
    upper_95 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("model_version", "target_month")
        ordering = ["target_month"]


class ActualPrice(models.Model):
    """Observed monthly average prices, ingested as they become available."""
    month = models.DateField(unique=True)
    price = models.FloatField()
    source = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["month"]


class MonitoringRun(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name="monitoring_runs")
    result = models.JSONField()
    retrain_recommended = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
