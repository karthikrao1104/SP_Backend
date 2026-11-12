from rest_framework import serializers


class StudentPredictionSerializer(serializers.Serializer):

    attendance = serializers.FloatField(min_value=0, max_value=100)

    study_hours = serializers.FloatField(min_value=0, max_value=12)

    internal_marks = serializers.FloatField(min_value=0, max_value=100)

    assignment_marks = serializers.FloatField(min_value=0, max_value=100)