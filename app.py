import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")


model = load_model()


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("💳 FinSight AI")
st.subheader("AI-Powered Loan Risk Classification Assistant")

st.info(
    "This is an educational machine learning demo. "
    "It estimates loan risk using historical loan data "
    "and does not make actual banking or lending decisions."
)


# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

st.sidebar.header("Applicant Information")

annual_income = st.sidebar.number_input(
    "Annual Income",
    min_value=0.0,
    value=60000.0,
    step=5000.0
)

emp_length = st.sidebar.number_input(
    "Employment Length (years)",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=1.0
)

state = st.sidebar.selectbox(
    "State",
    [
        "CA", "NY", "TX", "FL", "IL",
        "NJ", "PA", "OH", "GA", "NC",
        "MI", "VA", "WA", "AZ", "MA",
        "Other"
    ]
)

homeownership = st.sidebar.selectbox(
    "Homeownership",
    [
        "RENT",
        "MORTGAGE",
        "OWN",
        "OTHER"
    ]
)

verified_income = st.sidebar.selectbox(
    "Income Verification",
    [
        "Not Verified",
        "Source Verified",
        "Verified"
    ]
)

loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "medical",
        "car",
        "small_business",
        "other"
    ]
)

application_type = st.sidebar.selectbox(
    "Application Type",
    [
        "individual",
        "joint"
    ]
)


# --------------------------------------------------
# Loan details
# --------------------------------------------------

st.sidebar.header("Loan Details")

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

term = st.sidebar.selectbox(
    "Loan Term",
    [
        36,
        60
    ]
)

interest_rate = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=12.0,
    step=0.1
)

installment = st.sidebar.number_input(
    "Monthly Installment",
    min_value=0.0,
    value=350.0,
    step=25.0
)

debt_to_income = st.sidebar.number_input(
    "Debt-to-Income Ratio (%)",
    min_value=0.0,
    max_value=200.0,
    value=15.0,
    step=1.0
)


# --------------------------------------------------
# Credit information
# --------------------------------------------------

st.sidebar.header("Credit Information")

delinq_2y = st.sidebar.number_input(
    "Delinquencies in Last 2 Years",
    min_value=0,
    value=0,
    step=1
)

inquiries_last_12m = st.sidebar.number_input(
    "Credit Inquiries in Last 12 Months",
    min_value=0,
    value=1,
    step=1
)

total_credit_lines = st.sidebar.number_input(
    "Total Credit Lines",
    min_value=0,
    value=10,
    step=1
)

open_credit_lines = st.sidebar.number_input(
    "Open Credit Lines",
    min_value=0,
    value=5,
    step=1
)

total_credit_limit = st.sidebar.number_input(
    "Total Credit Limit",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)

total_credit_utilized = st.sidebar.number_input(
    "Total Credit Utilized",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

num_historical_failed_to_pay = st.sidebar.number_input(
    "Historical Failed Payments",
    min_value=0,
    value=0,
    step=1
)

public_record_bankrupt = st.sidebar.number_input(
    "Public Record Bankruptcies",
    min_value=0,
    value=0,
    step=1
)


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = {
    "emp_title": "Unknown",
    "emp_length": emp_length,
    "state": state,
    "homeownership": homeownership,
    "annual_income": annual_income,
    "verified_income": verified_income,
    "debt_to_income": debt_to_income,
    "annual_income_joint": None,
    "verification_income_joint": None,
    "debt_to_income_joint": None,
    "delinq_2y": delinq_2y,
    "months_since_last_delinq": None,
    "earliest_credit_line": 2010,
    "inquiries_last_12m": inquiries_last_12m,
    "total_credit_lines": total_credit_lines,
    "open_credit_lines": open_credit_lines,
    "total_credit_limit": total_credit_limit,
    "total_credit_utilized": total_credit_utilized,
    "num_collections_last_12m": 0,
    "num_historical_failed_to_pay": num_historical_failed_to_pay,
    "months_since_90d_late": None,
    "current_accounts_delinq": 0,
    "total_collection_amount_ever": 0,
    "current_installment_accounts": 2,
    "accounts_opened_24m": 2,
    "months_since_last_credit_inquiry": 3,
    "num_satisfactory_accounts": 8,
    "num_accounts_120d_past_due": 0,
    "num_accounts_30d_past_due": 0,
    "num_active_debit_accounts": 2,
    "total_debit_limit": 15000,
    "num_total_cc_accounts": 5,
    "num_open_cc_accounts": 3,
    "num_cc_carrying_balance": 2,
    "num_mort_accounts": 1,
    "account_never_delinq_percent": 100.0,
    "tax_liens": 0,
    "public_record_bankrupt": public_record_bankrupt,
    "loan_purpose": loan_purpose,
    "application_type": application_type,
    "loan_amount": loan_amount,
    "term": term,
    "interest_rate": interest_rate,
    "installment": installment,
    "grade": "C",
    "sub_grade": "C1",
    "issue_month": "Jan-2018",
    "initial_listing_status": "whole",
    "disbursement_method": "Cash"
}

input_df = pd.DataFrame([input_data])


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------

st.header("Applicant Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Annual Income", f"${annual_income:,.0f}")

with col2:
    st.metric("Requested Loan", f"${loan_amount:,.0f}")

with col3:
    st.metric("Debt-to-Income", f"{debt_to_income:.1f}%")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Assess Loan Risk", use_container_width=True):

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        risk_percentage = probability * 100

        st.divider()
        st.header("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if probability >= 0.30:
                st.error("⚠️ Potentially Risky Loan")
            else:
                st.success("✅ Lower Risk Loan")

        with result_col2:
            st.metric(
                "Estimated Risk Probability",
                f"{risk_percentage:.2f}%"
            )

        st.progress(float(probability))

        st.warning(
            "This prediction is based on historical patterns. "
            "It should not be treated as a real loan approval or rejection."
        )

        st.subheader("Risk Interpretation")

        if probability >= 0.5:
            st.write(
                "The model identifies this application as potentially risky. "
                "Additional verification of income, credit history, and repayment "
                "capacity may be required."
            )
        else:
            st.write(
                "The model identifies this application as relatively lower risk "
                "based on the provided information."
            )

        with st.expander("View Applicant Data"):
            st.dataframe(input_df)

    except Exception as error:
        st.error(f"Prediction error: {error}")
        st.info(
            "Please check that loan_model.pkl was generated successfully "
            "and that all feature names match the training dataset."
        )