import pandas as pd
import numpy as np

# For reproducible results
np.random.seed(42)

# Number of SME records
n = 5000

# Generate SME financial data
data = {
    "business_age": np.random.randint(1, 21, n),
    "annual_revenue": np.random.randint(200000, 5000000, n),
    "monthly_cash_flow": np.random.randint(10000, 400000, n),
    "existing_loan": np.random.randint(0, 1500000, n),
    "repayment_history": np.random.randint(50, 101, n),
    "gst_consistency": np.random.randint(50, 101, n),
    "profit_margin": np.round(np.random.uniform(5, 35, n), 2),
    "transaction_count": np.random.randint(20, 1000, n),
    "payment_defaults": np.random.randint(0, 10, n),
    "loan_amount": np.random.randint(100000, 2000000, n)
}

df = pd.DataFrame(data)

# Calculate debt-to-income ratio
df["debt_to_income"] = (
    df["existing_loan"] / df["annual_revenue"]
).round(2)

# Create a risk score for generating realistic target values
risk_score = (
    (100 - df["repayment_history"]) * 0.30
    + (100 - df["gst_consistency"]) * 0.15
    + df["debt_to_income"].clip(0, 2) * 20
    + df["payment_defaults"] * 3
    + (10 - df["profit_margin"].clip(0, 10)) * 0.5
)

# Add some randomness
risk_score += np.random.normal(0, 5, n)

# Create target variable
# 1 = Default
# 0 = No Default
df["loan_default"] = (risk_score > 35).astype(int)

# Save dataset
df.to_csv("data/sme_credit_data.csv", index=False)

print("Dataset generated successfully!")
print(f"Total records: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nDefault distribution:")
print(df["loan_default"].value_counts())