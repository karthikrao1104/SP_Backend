from django.urls import path
from . import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

urlpatterns = [
    # Registration
    path('register/', views.register_user, name='register'),

    # Login
    path('login/', TokenObtainPairView.as_view(), name='login'),

    # Refresh access token
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Verify access token
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),

    path('dashboard/', views.dashboard, name='dashboard'),
]