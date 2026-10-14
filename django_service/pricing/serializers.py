from rest_framework import serializers

from forecasting.predict import MAX_HORIZON

from .models import ActualPrice, ModelVersion, MonitoringRun


class HorizonSerializer(serializers.Serializer):
    horizon = serializers.IntegerField(min_value=1, max_value=MAX_HORIZON, default=12)


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = ["version", "model_name", "description", "train_start", "train_end",
                  "backtest_mape", "is_active", "created_at"]


class ActualPriceSerializer(serializers.ModelSerializer):
    month = serializers.DateField(input_formats=["%Y-%m", "%Y-%m-%d"])

    class Meta:
        model = ActualPrice
        fields = ["month", "price", "source"]
        extra_kwargs = {"month": {"validators": []}}  # upsert handled in the view

    def validate_month(self, value):
        return value.replace(day=1)

    def validate_price(self, value):
        if value <= 0:
            raise serializers.ValidationError("Price must be positive.")
        return value


class MonitoringRunSerializer(serializers.ModelSerializer):
    model_version = serializers.CharField(source="model_version.version")

    class Meta:
        model = MonitoringRun
        fields = ["model_version", "result", "retrain_recommended", "created_at"]
