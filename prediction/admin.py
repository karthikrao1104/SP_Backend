from django.contrib import admin
from .models import StudentPrediction


@admin.register(StudentPrediction)
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'user', 'attendance', 'study_hours',
        'internal_marks', 'assignment_marks', 'prediction', 'created_at'
    )
    list_filter = ('prediction',)
    search_fields = ('user__username',)
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)
