from django.db import connection
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from . import services
from .models import ActualPrice, ModelVersion, MonitoringRun
from .serializers import (ActualPriceSerializer, HorizonSerializer, ModelVersionSerializer,
                          MonitoringRunSerializer)


@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
    connection.ensure_connection()
    active = ModelVersion.objects.filter(is_active=True).values_list("version", flat=True).first()
    return Response({"status": "ok", "active_model": active, "backend": services.settings.FORECAST_BACKEND})


class ForecastView(APIView):
    """GET /api/forecast/?horizon=12  - public read."""
    permission_classes = [AllowAny]

    def get(self, request):
        s = HorizonSerializer(data=request.query_params)
        s.is_valid(raise_exception=True)
        try:
            return Response(services.get_forecast(s.validated_data["horizon"]))
        except services.NoActiveModel as e:
            return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except OSError as e:  # FastAPI backend unreachable
            return Response({"detail": f"Forecast backend unavailable: {e}"},
                            status=status.HTTP_502_BAD_GATEWAY)


class ModelVersionListView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        return Response(ModelVersionSerializer(ModelVersion.objects.all(), many=True).data)


class ActivateModelView(APIView):
    """POST /api/models/<version>/activate/ - promote or roll back."""
    permission_classes = [IsAdminUser]

    def post(self, request, version):
        try:
            mv = ModelVersion.objects.get(version=version)
        except ModelVersion.DoesNotExist:
            return Response({"detail": "Unknown version"}, status=404)
        mv.activate()
        services.notify_fastapi_reload(mv.version)
        return Response(ModelVersionSerializer(mv).data)


class ActualPriceView(APIView):
    """GET list / POST upsert of observed prices: [{"month": "2026-01", "price": 1820.5}]"""

    def get_permissions(self):
        return [AllowAny()] if self.request.method == "GET" else [IsAdminUser()]

    def get(self, request):
        return Response(ActualPriceSerializer(ActualPrice.objects.all(), many=True).data)

    def post(self, request):
        payload = request.data if isinstance(request.data, list) else [request.data]
        s = ActualPriceSerializer(data=payload, many=True)
        s.is_valid(raise_exception=True)
        for row in s.validated_data:
            ActualPrice.objects.update_or_create(month=row["month"], defaults={
                "price": row["price"], "source": row.get("source", "")})
        return Response({"upserted": len(s.validated_data)}, status=201)


class MonitoringView(APIView):
    """GET latest run; POST triggers a new run."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        run = MonitoringRun.objects.first()
        return Response(MonitoringRunSerializer(run).data if run else {"detail": "No runs yet"})

    def post(self, request):
        try:
            run = services.run_monitoring()
        except services.NoActiveModel as e:
            return Response({"detail": str(e)}, status=503)
        return Response(MonitoringRunSerializer(run).data, status=201)
