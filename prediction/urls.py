from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_student, name='predict'),
    path('history/', views.prediction_history, name='history'),
    path('upload/', views.bulk_predict_csv, name='bulk_predict_csv'),
]