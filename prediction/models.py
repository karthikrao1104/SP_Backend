from django.db import models
from django.contrib.auth.models import User


class StudentPrediction(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="predictions"
    )

    attendance = models.FloatField()
    study_hours = models.FloatField()
    internal_marks = models.FloatField()
    assignment_marks = models.FloatField()

    prediction = models.CharField(max_length=10)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.prediction}"