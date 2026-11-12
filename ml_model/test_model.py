import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model = joblib.load(BASE_DIR / "student_model.pkl")

# New student details
student = [[90, 6, 85, 90]]

# Make prediction
prediction = model.predict(student)

if prediction[0] == 1:
    print("Student Result: PASS")
else:
    print("Student Result: FAIL")