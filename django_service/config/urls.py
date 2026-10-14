from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path("", RedirectView.as_view(url="/api/health/", permanent=False), name="home"),
    path("admin/", admin.site.urls),
    path("api/", include("pricing.urls")),
    path("api/auth/token/", obtain_auth_token),
]
