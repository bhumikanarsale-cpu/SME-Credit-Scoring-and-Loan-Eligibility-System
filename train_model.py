import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/sme_credit_data.csv")

# Features
features = [
    "business_age",
    "annual_revenue",
    "monthly_cash_flow",
    "existing_loan",
    "repayment_history",
    "gst_consistency",
    "profit_margin",
    "transaction_count",
    "payment_defaults",
    "loan_amount",
    "debt_to_income"
]

X = df[features]
y = df["loan_default"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/credit_risk_model.pkl")

print("\nModel saved successfully!")
print("Location: models/credit_risk_model.pkl")