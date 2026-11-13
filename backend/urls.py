from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def root_health_check(request):
    return JsonResponse({
        "status": "online",
        "message": "Student Pass/Fail Prediction API is active",
        "server": "online",
        "endpoints": {
            "test": "/api/accounts/test/",
            "auth": "/api/accounts/",
            "prediction": "/api/prediction/"
        }
    }, status=200)

urlpatterns = [
    path('', root_health_check, name='root_health_check'),
    path('admin/', admin.site.urls),
    path('api/accounts/', include('accounts.urls')),
    path('api/prediction/', include('prediction.urls')),
]