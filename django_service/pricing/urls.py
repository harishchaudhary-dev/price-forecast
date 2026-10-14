from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health),
    path("forecast/", views.ForecastView.as_view()),
    path("models/", views.ModelVersionListView.as_view()),
    path("models/<str:version>/activate/", views.ActivateModelView.as_view()),
    path("actuals/", views.ActualPriceView.as_view()),
    path("monitoring/", views.MonitoringView.as_view()),
]
