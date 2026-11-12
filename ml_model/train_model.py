import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(BASE_DIR / "student_data.csv")

print("Dataset loaded successfully")
print(data.head())

# 2. Separate features and target
X = data.drop("result", axis=1)
y = data["result"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Predict test data
y_pred = model.predict(X_test)

# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Feature importance
print("\nFeature Importance:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(feature, ":", round(importance * 100, 2), "%")

# 9. Save trained model
joblib.dump(model, BASE_DIR / "student_model.pkl")

print("\nModel saved successfully!")

# 10. Load saved PKL file
loaded_model = joblib.load(BASE_DIR / "student_model.pkl")

print("\n========== PKL FILE OUTPUT ==========")

print("\nModel:")
print(loaded_model)

print("\nModel Parameters:")
print(loaded_model.get_params())

print("\nNumber of Trees:")
print(loaded_model.n_estimators)

print("\nNumber of Features:")
print(loaded_model.n_features_in_)

print("\nFeature Names:")
print(loaded_model.feature_names_in_)

print("\nFeature Importance:")
for feature, importance in zip(
    loaded_model.feature_names_in_,
    loaded_model.feature_importances_
):
    print(feature, ":", round(importance * 100, 2), "%")

print("\nClasses:")
print(loaded_model.classes_)