
import joblib
import pandas as pd

from pathlib import Path

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .serializers import StudentPredictionSerializer
from .models import StudentPrediction
from django.db import transaction

# Get model path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml_model" / "student_model.pkl"


# Load ML model
model = joblib.load(MODEL_PATH)


# Student Pass/Fail Prediction API
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_student(request):

    # Validate incoming student data
    serializer = StudentPredictionSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )

    # Get validated student data
    student_data = serializer.validated_data

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [student_data],
        columns=[
            'attendance',
            'study_hours',
            'internal_marks',
            'assignment_marks'
        ]
    )

    # Make prediction using ML model
    prediction = model.predict(input_data)[0]

    # Convert prediction into PASS or FAIL
    result = "PASS" if prediction == 1 else "FAIL"

    # Save prediction in database
    saved_prediction = StudentPrediction.objects.create(
        user=request.user,
        attendance=student_data['attendance'],
        study_hours=student_data['study_hours'],
        internal_marks=student_data['internal_marks'],
        assignment_marks=student_data['assignment_marks'],
        prediction=result
    )

    # Return prediction response
    return Response(
        {
            "id": saved_prediction.id,
            "username": request.user.username,
            "prediction": int(prediction),
            "result": result,
            "student_data": student_data,
            "created_at": saved_prediction.created_at
        },
        status=status.HTTP_200_OK
    )



# Student Prediction History API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def prediction_history(request):

    # Get predictions belonging to the logged-in user
    history = StudentPrediction.objects.filter(
        user=request.user
    ).order_by('-created_at')

    # Convert database records into JSON
    history_data = []

    for item in history:
        history_data.append({
            "id": item.id,
            "attendance": item.attendance,
            "study_hours": item.study_hours,
            "internal_marks": item.internal_marks,
            "assignment_marks": item.assignment_marks,
            "prediction": item.prediction,
            "created_at": item.created_at
        })

    return Response(
        {
            "username": request.user.username,
            "total_predictions": history.count(),
            "history": history_data
        },
        status=status.HTTP_200_OK
    )



# CSV Bulk Prediction API
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bulk_predict_csv(request):

    # Get uploaded CSV file
    csv_file = request.FILES.get('file')

    if not csv_file:
        return Response(
            {"error": "Please upload a CSV file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check file extension
    if not csv_file.name.lower().endswith('.csv'):
        return Response(
            {"error": "Only CSV files are allowed."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return Response(
            {"error": "Unable to read CSV file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Required columns
    required_columns = [
        'attendance',
        'study_hours',
        'internal_marks',
        'assignment_marks'
    ]

    # Check required columns
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        return Response(
            {
                "error": "Missing required columns.",
                "missing_columns": missing_columns
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check empty CSV
    if df.empty:
        return Response(
            {"error": "CSV file contains no student records."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Maximum 100 students
    if len(df) > 100:
        return Response(
            {"error": "Maximum 100 student records are allowed."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Keep only required columns
    df = df[required_columns]

    # Validate student data
    validated_rows = []
    errors = []

    for index, row in enumerate(df.to_dict(orient='records'), start=1):

        serializer = StudentPredictionSerializer(data=row)

        if serializer.is_valid():
            validated_rows.append(serializer.validated_data)
        else:
            errors.append({
                "row": index,
                "errors": serializer.errors
            })

    # Stop if any row is invalid
    if errors:
        return Response(
            {
                "error": "Some student records contain invalid data.",
                "details": errors
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # Prepare input for ML model
    input_data = pd.DataFrame(
        validated_rows,
        columns=required_columns
    )

    # Predict all students
    predictions = model.predict(input_data)

    # Save all predictions in database
    results = []

    with transaction.atomic():

        for index, (student_data, prediction) in enumerate(
            zip(validated_rows, predictions), start=1
        ):

            result = "PASS" if prediction == 1 else "FAIL"

            saved_record = StudentPrediction.objects.create(
                user=request.user,
                attendance=student_data['attendance'],
                study_hours=student_data['study_hours'],
                internal_marks=student_data['internal_marks'],
                assignment_marks=student_data['assignment_marks'],
                prediction=result
            )

            results.append({
                "id": saved_record.id,
                "student_number": index,
                "attendance": student_data['attendance'],
                "study_hours": student_data['study_hours'],
                "internal_marks": student_data['internal_marks'],
                "assignment_marks": student_data['assignment_marks'],
                "prediction": result,
                "created_at": saved_record.created_at
            })

    # Calculate statistics
    pass_count = sum(
        1 for item in results if item['prediction'] == "PASS"
    )

    fail_count = sum(
        1 for item in results if item['prediction'] == "FAIL"
    )

    return Response(
        {
            "message": "CSV predictions completed successfully.",
            "username": request.user.username,
            "total_students": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "results": results
        },
        status=status.HTTP_200_OK
    )
