import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/credit_risk_model.pkl")

st.set_page_config(
    page_title="SME Credit Scoring System",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("💼 SME Credit Scoring & Loan Eligibility System")

st.write(
    "A machine learning-based system for evaluating SME credit risk "
    "and estimating loan eligibility."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Business Information")

business_age = st.sidebar.number_input(
    "Business Age (Years)",
    min_value=1,
    max_value=50,
    value=5
)

annual_revenue = st.sidebar.number_input(
    "Annual Revenue (₹)",
    min_value=100000,
    max_value=10000000,
    value=2000000,
    step=100000
)

monthly_cash_flow = st.sidebar.number_input(
    "Monthly Cash Flow (₹)",
    min_value=5000,
    max_value=1000000,
    value=150000,
    step=10000
)

existing_loan = st.sidebar.number_input(
    "Existing Loan (₹)",
    min_value=0,
    max_value=5000000,
    value=500000,
    step=50000
)

repayment_history = st.sidebar.slider(
    "Repayment History (%)",
    min_value=0,
    max_value=100,
    value=85
)

gst_consistency = st.sidebar.slider(
    "GST Filing Consistency (%)",
    min_value=0,
    max_value=100,
    value=90
)

profit_margin = st.sidebar.slider(
    "Profit Margin (%)",
    min_value=0.0,
    max_value=50.0,
    value=20.0
)

transaction_count = st.sidebar.number_input(
    "Monthly Transactions",
    min_value=1,
    max_value=5000,
    value=300
)

payment_defaults = st.sidebar.number_input(
    "Payment Defaults",
    min_value=0,
    max_value=20,
    value=1
)

loan_amount = st.sidebar.number_input(
    "Requested Loan Amount (₹)",
    min_value=50000,
    max_value=5000000,
    value=500000,
    step=50000
)

# Calculate debt-to-income ratio
debt_to_income = round(
    existing_loan / annual_revenue,
    2
)

st.sidebar.info(
    f"Debt-to-Income Ratio: {debt_to_income}"
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.sidebar.button("🔍 Assess Credit Risk"):

    input_data = pd.DataFrame({
        "business_age": [business_age],
        "annual_revenue": [annual_revenue],
        "monthly_cash_flow": [monthly_cash_flow],
        "existing_loan": [existing_loan],
        "repayment_history": [repayment_history],
        "gst_consistency": [gst_consistency],
        "profit_margin": [profit_margin],
        "transaction_count": [transaction_count],
        "payment_defaults": [payment_defaults],
        "loan_amount": [loan_amount],
        "debt_to_income": [debt_to_income]
    })

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Credit score
    credit_score = int(850 - (probability * 550))

    # Risk category
    if credit_score >= 700:
        risk = "Low Risk"
    elif credit_score >= 550:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # -----------------------------
    # Loan Eligibility
    # -----------------------------

    if risk == "Low Risk":

        eligibility = "Eligible"

        suggested_loan = min(
            loan_amount,
            int(monthly_cash_flow * 12 * 2)
        )

    elif risk == "Medium Risk":

        eligibility = "Conditionally Eligible"

        suggested_loan = min(
            loan_amount,
            int(monthly_cash_flow * 12)
        )

    else:

        eligibility = "Not Eligible"

        suggested_loan = 0

    # -----------------------------
    # Loan Optimizer
    # -----------------------------

    if suggested_loan > 0:

        # Maximum affordable EMI = 30% of monthly cash flow
        max_affordable_emi = monthly_cash_flow * 0.30

        # Annual interest rate used for demonstration
        annual_interest_rate = 12

        monthly_interest_rate = (
            annual_interest_rate / 12 / 100
        )

        # Decide suitable tenure
        recommended_tenure = 36

        for tenure in [12, 24, 36]:

            months = tenure

            emi = (
                suggested_loan
                * monthly_interest_rate
                * (1 + monthly_interest_rate) ** months
            ) / (
                (1 + monthly_interest_rate) ** months - 1
            )

            if emi <= max_affordable_emi:
                recommended_tenure = tenure
                break

        # Final EMI calculation
        months = recommended_tenure

        estimated_emi = (
            suggested_loan
            * monthly_interest_rate
            * (1 + monthly_interest_rate) ** months
        ) / (
            (1 + monthly_interest_rate) ** months - 1
        )

    else:

        recommended_tenure = 0
        estimated_emi = 0
        max_affordable_emi = monthly_cash_flow * 0.30

    # -----------------------------
    # Results
    # -----------------------------

    st.subheader("📊 Credit Assessment")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Credit Score",
        credit_score
    )

    col2.metric(
        "Risk Category",
        risk
    )

    col3.metric(
        "Loan Eligibility",
        eligibility
    )

    col4.metric(
        "Suggested Loan",
        f"₹{suggested_loan:,.0f}"
    )

    st.divider()

    # -----------------------------
    # Risk Probability
    # -----------------------------

    st.subheader("📈 Risk Analysis")

    risk_percentage = probability * 100

    st.progress(
        min(int(risk_percentage), 100)
    )

    st.write(
        f"Estimated default risk: **{risk_percentage:.2f}%**"
    )

    st.divider()

    # -----------------------------
    # Loan Recommendation
    # -----------------------------

    st.subheader("💰 Loan Recommendation")

    if suggested_loan > 0:

        loan_col1, loan_col2, loan_col3 = st.columns(3)

        loan_col1.metric(
            "Suggested Loan Amount",
            f"₹{suggested_loan:,.0f}"
        )

        loan_col2.metric(
            "Recommended Tenure",
            f"{recommended_tenure} Months"
        )

        loan_col3.metric(
            "Estimated Monthly EMI",
            f"₹{estimated_emi:,.0f}"
        )

        st.info(
            f"Maximum estimated affordable EMI based on cash flow: "
            f"₹{max_affordable_emi:,.0f}"
        )

    else:

        st.warning(
            "Loan recommendation is not available because "
            "the application does not meet the current eligibility criteria."
        )

    st.divider()

    # -----------------------------
    # Key Factors
    # -----------------------------

    st.subheader("🔎 Key Assessment Factors")

    factors = []

    if repayment_history >= 80:
        factors.append("✅ Strong repayment history")
    else:
        factors.append("⚠️ Weak repayment history")

    if gst_consistency >= 80:
        factors.append("✅ Good GST filing consistency")
    else:
        factors.append("⚠️ Inconsistent GST filing")

    if profit_margin >= 15:
        factors.append("✅ Healthy profit margin")
    else:
        factors.append("⚠️ Low profit margin")

    if debt_to_income <= 0.4:
        factors.append("✅ Manageable existing debt")
    else:
        factors.append("⚠️ High existing debt")

    if payment_defaults <= 2:
        factors.append("✅ Low number of payment defaults")
    else:
        factors.append("⚠️ High number of payment defaults")

    for factor in factors:
        st.write(factor)

else:

    st.info(
        "👈 Enter the SME's financial information from the sidebar "
        "and click **Assess Credit Risk**."
    )

st.divider()

st.caption(
    "SME Credit Scoring System | Machine Learning Based Decision Support"
)