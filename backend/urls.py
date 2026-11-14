from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def health_check(request):
    """
    Ultra-lightweight public health check endpoint for external schedulers
    (cron-job.org, UptimeRobot, GitHub Actions) and Render keep-alive pings.
    Does not invoke DB or ML models, returns pure JSON regardless of Accept header.
    """
    return JsonResponse({
        "status": "healthy",
        "message": "Backend is running",
        "server": "online"
    }, status=200)

urlpatterns = [
    path('', health_check, name='root_health_check'),
    path('health/', health_check, name='health_check'),
    path('api/accounts/', include('accounts.urls')),
    path('api/prediction/', include('prediction.urls')),
    path('admin/', admin.site.urls),
]