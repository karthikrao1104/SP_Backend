
import logging

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML model – loaded lazily so a missing .pkl does not crash the entire
# Django process at startup. The first prediction request will load it.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml_model" / "student_model.pkl"

_model = None


def get_model():
    """Return the cached ML model, loading it on first call."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ML model not found at {MODEL_PATH}. "
                "Ensure student_model.pkl is committed to the repository."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("ML model loaded successfully from %s", MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Student Pass/Fail Prediction API
# ---------------------------------------------------------------------------

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_student(request):
    """Predict pass/fail for a single student and save to history."""

    # Validate incoming student data
    serializer = StudentPredictionSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )

    student_data = serializer.validated_data

    # Load model (lazy)
    try:
        model = get_model()
    except FileNotFoundError as exc:
        logger.error("ML model not found: %s", exc)
        return Response(
            {"error": "ML model is not available. Please contact support."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as exc:
        logger.error("Failed to load ML model: %s", exc)
        return Response(
            {"error": "ML model failed to load. Please contact support."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # Convert input into DataFrame (preserves feature names, suppresses warning)
    input_data = pd.DataFrame(
        [student_data],
        columns=[
            'attendance',
            'study_hours',
            'internal_marks',
            'assignment_marks',
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "PASS" if prediction == 1 else "FAIL"

    # Save prediction to database
    saved_prediction = StudentPrediction.objects.create(
        user=request.user,
        attendance=student_data['attendance'],
        study_hours=student_data['study_hours'],
        internal_marks=student_data['internal_marks'],
        assignment_marks=student_data['assignment_marks'],
        prediction=result
    )

    return Response(
        {
            "id": saved_prediction.id,
            "username": request.user.username,
            "prediction": int(prediction),
            "result": result,
            "student_data": student_data,
            "created_at": saved_prediction.created_at,
        },
        status=status.HTTP_201_CREATED
    )


# ---------------------------------------------------------------------------
# Student Prediction History API
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def prediction_history(request):
    """Return the prediction history for the currently authenticated user."""

    history = StudentPrediction.objects.filter(
        user=request.user
    ).order_by('-created_at')

    history_data = [
        {
            "id": item.id,
            "attendance": item.attendance,
            "study_hours": item.study_hours,
            "internal_marks": item.internal_marks,
            "assignment_marks": item.assignment_marks,
            "prediction": item.prediction,
            "created_at": item.created_at,
        }
        for item in history
    ]

    return Response(
        {
            "username": request.user.username,
            "total_predictions": history.count(),
            "history": history_data,
        },
        status=status.HTTP_200_OK
    )


# ---------------------------------------------------------------------------
# CSV Bulk Prediction API
# ---------------------------------------------------------------------------

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bulk_predict_csv(request):
    """Accept a CSV file, predict pass/fail for each row, and save results."""

    csv_file = request.FILES.get('file')

    if not csv_file:
        return Response(
            {"error": "Please upload a CSV file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if not csv_file.name.lower().endswith('.csv'):
        return Response(
            {"error": "Only CSV files are allowed."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return Response(
            {"error": "Unable to read CSV file. Ensure it is a valid CSV."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        'attendance',
        'study_hours',
        'internal_marks',
        'assignment_marks',
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return Response(
            {
                "error": "Missing required columns.",
                "missing_columns": missing_columns,
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    if df.empty:
        return Response(
            {"error": "CSV file contains no student records."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if len(df) > 100:
        return Response(
            {"error": "Maximum 100 student records are allowed per upload."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Keep only the required columns
    df = df[required_columns]

    # Validate every row through the serializer
    validated_rows = []
    errors = []

    for index, row in enumerate(df.to_dict(orient='records'), start=1):
        serializer = StudentPredictionSerializer(data=row)
        if serializer.is_valid():
            validated_rows.append(serializer.validated_data)
        else:
            errors.append({"row": index, "errors": serializer.errors})

    if errors:
        return Response(
            {
                "error": "Some student records contain invalid data.",
                "details": errors,
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # Load model (lazy)
    try:
        model = get_model()
    except FileNotFoundError as exc:
        logger.error("ML model not found: %s", exc)
        return Response(
            {"error": "ML model is not available. Please contact support."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as exc:
        logger.error("Failed to load ML model: %s", exc)
        return Response(
            {"error": "ML model failed to load. Please contact support."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # Prepare DataFrame for batch prediction
    input_data = pd.DataFrame(validated_rows, columns=required_columns)
    predictions = model.predict(input_data)

    # Save all records atomically
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
                "created_at": saved_record.created_at,
            })

    pass_count = sum(1 for item in results if item['prediction'] == "PASS")
    fail_count = sum(1 for item in results if item['prediction'] == "FAIL")

    return Response(
        {
            "message": "CSV predictions completed successfully.",
            "username": request.user.username,
            "total_students": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "results": results,
        },
        status=status.HTTP_201_CREATED
    )
