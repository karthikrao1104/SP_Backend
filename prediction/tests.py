"""
Comprehensive tests for the Student Prediction Backend.

Covers:
  - User registration
  - Login / JWT token issuance
  - Token refresh
  - Protected endpoint access
  - Single student prediction
  - Prediction history (user-scoped)
  - CSV bulk upload and prediction
  - Input validation (invalid ranges, missing fields)
  - Unauthorized access rejection
  - ML model loading
"""

import io
import csv
import joblib

from pathlib import Path
from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from rest_framework.test import APIClient
from rest_framework import status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client_with_token(user):
    """Return an authenticated APIClient for *user*."""
    client = APIClient()
    # Obtain tokens via the login endpoint
    response = client.post(
        '/api/accounts/login/',
        {'username': user.username, 'password': 'testpass123'},
        format='json',
    )
    access = response.data['access']
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')
    return client


def make_csv_file(rows, header=None, filename='students.csv'):
    """Build a SimpleUploadedFile CSV from a list of dicts."""
    if header is None:
        header = ['attendance', 'study_hours', 'internal_marks', 'assignment_marks']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    return SimpleUploadedFile(
        filename,
        buf.getvalue().encode(),
        content_type='text/csv',
    )


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class RegistrationTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.url = '/api/accounts/register/'

    def test_successful_registration(self):
        response = self.client.post(
            self.url,
            {'username': 'alice', 'email': 'alice@example.com', 'password': 'pass1234'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('message', response.data)
        self.assertTrue(User.objects.filter(username='alice').exists())

    def test_duplicate_username_rejected(self):
        User.objects.create_user('bob', 'bob@example.com', 'pass1234')
        response = self.client.post(
            self.url,
            {'username': 'bob', 'email': 'bob2@example.com', 'password': 'pass1234'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_password_rejected(self):
        response = self.client.post(
            self.url,
            {'username': 'carol', 'email': 'carol@example.com'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_short_password_rejected(self):
        response = self.client.post(
            self.url,
            {'username': 'dave', 'email': 'dave@example.com', 'password': '123'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# ---------------------------------------------------------------------------
# JWT authentication tests
# ---------------------------------------------------------------------------

class AuthenticationTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user('eve', 'eve@example.com', 'testpass123')
        self.login_url = '/api/accounts/login/'
        self.refresh_url = '/api/accounts/token/refresh/'

    def test_login_returns_tokens(self):
        response = self.client.post(
            self.login_url,
            {'username': 'eve', 'password': 'testpass123'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)

    def test_wrong_password_rejected(self):
        response = self.client.post(
            self.login_url,
            {'username': 'eve', 'password': 'wrongpass'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_refresh_token_works(self):
        login = self.client.post(
            self.login_url,
            {'username': 'eve', 'password': 'testpass123'},
            format='json',
        )
        refresh_token = login.data['refresh']
        response = self.client.post(
            self.refresh_url,
            {'refresh': refresh_token},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)

    def test_protected_endpoint_requires_auth(self):
        response = self.client.get('/api/accounts/dashboard/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_authenticated_dashboard_access(self):
        client = make_client_with_token(self.user)
        response = client.get('/api/accounts/dashboard/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], 'eve')


# ---------------------------------------------------------------------------
# Single prediction tests
# ---------------------------------------------------------------------------

class PredictionTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user('frank', 'frank@example.com', 'testpass123')
        self.client = make_client_with_token(self.user)
        self.url = '/api/prediction/predict/'

    def test_valid_prediction_pass(self):
        data = {
            'attendance': 90,
            'study_hours': 7,
            'internal_marks': 85,
            'assignment_marks': 90,
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('result', response.data)
        self.assertIn(response.data['result'], ['PASS', 'FAIL'])

    def test_valid_prediction_fail(self):
        data = {
            'attendance': 20,
            'study_hours': 1,
            'internal_marks': 15,
            'assignment_marks': 10,
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('result', response.data)

    def test_prediction_saves_to_history(self):
        data = {
            'attendance': 75,
            'study_hours': 5,
            'internal_marks': 65,
            'assignment_marks': 70,
        }
        self.client.post(self.url, data, format='json')
        from prediction.models import StudentPrediction
        self.assertEqual(StudentPrediction.objects.filter(user=self.user).count(), 1)

    def test_attendance_above_100_rejected(self):
        data = {
            'attendance': 110,
            'study_hours': 5,
            'internal_marks': 65,
            'assignment_marks': 70,
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_negative_values_rejected(self):
        data = {
            'attendance': -5,
            'study_hours': 5,
            'internal_marks': 65,
            'assignment_marks': 70,
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_study_hours_above_12_rejected(self):
        data = {
            'attendance': 80,
            'study_hours': 15,
            'internal_marks': 65,
            'assignment_marks': 70,
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_field_rejected(self):
        data = {
            'attendance': 80,
            'study_hours': 5,
            'internal_marks': 65,
            # assignment_marks missing
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_unauthenticated_prediction_rejected(self):
        anon = APIClient()
        data = {
            'attendance': 80,
            'study_hours': 5,
            'internal_marks': 65,
            'assignment_marks': 70,
        }
        response = anon.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------------
# Prediction history tests
# ---------------------------------------------------------------------------

class PredictionHistoryTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user('grace', 'grace@example.com', 'testpass123')
        self.other_user = User.objects.create_user('henry', 'henry@example.com', 'testpass123')
        self.client = make_client_with_token(self.user)
        self.predict_url = '/api/prediction/predict/'
        self.history_url = '/api/prediction/history/'

    def test_empty_history(self):
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_predictions'], 0)
        self.assertEqual(response.data['history'], [])

    def test_history_contains_own_predictions(self):
        data = {
            'attendance': 90,
            'study_hours': 7,
            'internal_marks': 85,
            'assignment_marks': 90,
        }
        self.client.post(self.predict_url, data, format='json')
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_predictions'], 1)

    def test_history_is_user_scoped(self):
        """Other user's predictions must not appear in history."""
        other_client = make_client_with_token(self.other_user)
        data = {
            'attendance': 90,
            'study_hours': 7,
            'internal_marks': 85,
            'assignment_marks': 90,
        }
        other_client.post(self.predict_url, data, format='json')
        response = self.client.get(self.history_url)
        self.assertEqual(response.data['total_predictions'], 0)

    def test_history_unauthenticated_rejected(self):
        anon = APIClient()
        response = anon.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------------
# CSV bulk upload tests
# ---------------------------------------------------------------------------

class CSVUploadTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user('iris', 'iris@example.com', 'testpass123')
        self.client = make_client_with_token(self.user)
        self.url = '/api/prediction/upload/'

    def test_valid_csv_upload(self):
        rows = [
            {'attendance': 90, 'study_hours': 7, 'internal_marks': 85, 'assignment_marks': 90},
            {'attendance': 20, 'study_hours': 1, 'internal_marks': 15, 'assignment_marks': 10},
        ]
        csv_file = make_csv_file(rows)
        response = self.client.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['total_students'], 2)
        self.assertIn('pass_count', response.data)
        self.assertIn('fail_count', response.data)

    def test_missing_file_rejected(self):
        response = self.client.post(self.url, {}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_non_csv_file_rejected(self):
        txt_file = SimpleUploadedFile('data.txt', b'some text content', content_type='text/plain')
        response = self.client.post(
            self.url,
            {'file': txt_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_column_rejected(self):
        # CSV without study_hours
        rows = [{'attendance': 90, 'internal_marks': 85, 'assignment_marks': 90}]
        csv_file = make_csv_file(
            rows, header=['attendance', 'internal_marks', 'assignment_marks']
        )
        response = self.client.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('missing_columns', response.data)

    def test_invalid_data_row_rejected(self):
        rows = [
            {'attendance': 999, 'study_hours': 7, 'internal_marks': 85, 'assignment_marks': 90},
        ]
        csv_file = make_csv_file(rows)
        response = self.client.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_empty_csv_rejected(self):
        csv_file = make_csv_file([])
        response = self.client.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_csv_saves_all_rows_to_db(self):
        rows = [
            {'attendance': 90, 'study_hours': 7, 'internal_marks': 85, 'assignment_marks': 90},
            {'attendance': 20, 'study_hours': 1, 'internal_marks': 15, 'assignment_marks': 10},
            {'attendance': 60, 'study_hours': 4, 'internal_marks': 55, 'assignment_marks': 60},
        ]
        csv_file = make_csv_file(rows)
        self.client.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        from prediction.models import StudentPrediction
        self.assertEqual(StudentPrediction.objects.filter(user=self.user).count(), 3)

    def test_unauthenticated_upload_rejected(self):
        anon = APIClient()
        rows = [{'attendance': 90, 'study_hours': 7, 'internal_marks': 85, 'assignment_marks': 90}]
        csv_file = make_csv_file(rows)
        response = anon.post(
            self.url,
            {'file': csv_file},
            format='multipart',
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------------
# ML model loading test
# ---------------------------------------------------------------------------

class MLModelTests(TestCase):

    def test_model_file_exists(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / 'ml_model' / 'student_model.pkl'
        self.assertTrue(model_path.exists(), f"Model file missing: {model_path}")

    def test_model_loads_without_error(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / 'ml_model' / 'student_model.pkl'
        import joblib
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

    def test_model_predicts_pass(self):
        import joblib, pandas as pd
        base_dir = Path(__file__).resolve().parent.parent
        model = joblib.load(base_dir / 'ml_model' / 'student_model.pkl')
        df = pd.DataFrame(
            [{'attendance': 90, 'study_hours': 7, 'internal_marks': 85, 'assignment_marks': 90}]
        )
        prediction = model.predict(df)[0]
        self.assertEqual(prediction, 1)  # PASS

    def test_model_predicts_fail(self):
        import joblib, pandas as pd
        base_dir = Path(__file__).resolve().parent.parent
        model = joblib.load(base_dir / 'ml_model' / 'student_model.pkl')
        df = pd.DataFrame(
            [{'attendance': 20, 'study_hours': 1, 'internal_marks': 15, 'assignment_marks': 10}]
        )
        prediction = model.predict(df)[0]
        self.assertEqual(prediction, 0)  # FAIL
