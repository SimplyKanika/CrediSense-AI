import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="CrediSense AI",
    page_icon="💳",
    layout="wide"
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("credit_risk_model.pkl")


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #123B73;
}

.subtitle {
    font-size: 18px;
    color: #555555;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">💳 CrediSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Credit Risk & Loan Assessment System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("About CrediSense AI")

st.sidebar.write(
    """
    CrediSense AI is a machine learning based
    credit risk assessment prototype.

    It analyzes applicant financial information
    and predicts whether a loan application is
    likely to be approved.
    """
)

st.sidebar.info(
    "This is an educational prototype and should "
    "not be used as a real financial decision system."
)


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

DEFAULT_VALUES = {
    "age": 25,
    "annual_income": 600000,
    "credit_score": 720,
    "loan_amount": 300000,
    "loan_term": 12,
    "existing_debt": 50000,
    "employment": "Salaried",
    "previous_defaults": 0,
    "dependents": 2
}


def reset_inputs():
    """Reset all applicant input fields to their default values."""
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value

def validate_inputs(
    age,
    annual_income,
    credit_score,
    loan_amount,
    existing_debt,
    previous_defaults,
    dependents
):
    errors = []
    warnings = []

    # Hard validation errors
    if annual_income <= 0:
        errors.append("Annual income must be greater than ₹0.")

    if loan_amount <= 0:
        errors.append("Loan amount must be greater than ₹0.")

    if existing_debt < 0:
        errors.append("Existing debt cannot be negative.")

    if age < 18:
        errors.append("Applicant must be at least 18 years old.")

    # Financial ratios
    dti = None
    lti = None

    if annual_income > 0:
        dti = (existing_debt / annual_income) * 100
        lti = (loan_amount / annual_income) * 100

    # Model training range warnings
    if age < 21 or age > 60:
        warnings.append(
            "Age is outside the range used to train the model (21–60 years)."
        )

    if annual_income < 200000 or annual_income > 1500000:
        warnings.append(
            "Annual income is outside the range used to train the model "
            "(₹2,00,000–₹15,00,000)."
        )

    if loan_amount < 50000 or loan_amount > 1000000:
        warnings.append(
            "Loan amount is outside the range used to train the model "
            "(₹50,000–₹10,00,000)."
        )

    if existing_debt > 500000:
        warnings.append(
            "Existing debt is outside the range used to train the model "
            "(maximum ₹5,00,000)."
        )

    if previous_defaults > 3:
        warnings.append(
            "Previous defaults are outside the range used to train the model "
            "(0–3)."
        )

    if dependents > 4:
        warnings.append(
            "Number of dependents are outside the range used to train the model "
            "(0–4)."
        )

    # Financial ratio warnings
    if dti is not None and dti > 50:
        warnings.append(
            f"Debt-to-income ratio is high ({dti:.1f}%). "
            "Review the applicant's existing debt."
        )

    if lti is not None and lti > 200:
        warnings.append(
            f"Loan-to-income ratio is high ({lti:.1f}%). "
            "The requested loan is large relative to annual income."
        )

    return errors, warnings, dti, lti

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=DEFAULT_VALUES["age"],
        key="age"
    )

    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=50000,
        max_value=5000000,
        value=DEFAULT_VALUES["annual_income"],
        step=50000,
        key="annual_income"
    )

    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=DEFAULT_VALUES["credit_score"],
        key="credit_score"
    )


with col2:

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=10000,
        max_value=3000000,
        value=DEFAULT_VALUES["loan_amount"],
        step=10000,
        key="loan_amount"
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60],
        index=[12, 24, 36, 48, 60].index(
            DEFAULT_VALUES["loan_term"]
        ),
        key="loan_term"
    )

    existing_debt = st.number_input(
        "Existing Debt (₹)",
        min_value=0,
        max_value=2000000,
        step=10000,
        value=DEFAULT_VALUES["existing_debt"],
        key="existing_debt"
    )


with col3:

    employment = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Self Employed",
            "Unemployed"
        ],
        index=[
            "Salaried",
            "Self Employed",
            "Unemployed"
        ].index(DEFAULT_VALUES["employment"]),
        key="employment"
    )

    previous_defaults = st.number_input(
        "Previous Defaults",
        min_value=0,
        max_value=5,
        value=DEFAULT_VALUES["previous_defaults"],
        key="previous_defaults"
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=DEFAULT_VALUES["dependents"],
        key="dependents"
    )


# ---------------------------------------------------
# ACTION BUTTONS
# ---------------------------------------------------

st.divider()

button_col1, button_col2 = st.columns(2)

with button_col1:

    predict_button = st.button(
        "🔍 ASSESS CREDIT RISK",
        use_container_width=True
    )

with button_col2:

    st.button(
        "🔄 RESET INPUTS",
        use_container_width=True,
        on_click=reset_inputs
    )

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:
    errors, warnings, dti, lti = validate_inputs(
            age,
            annual_income,
            credit_score,
            loan_amount,
            existing_debt,
            previous_defaults,
            dependents
        )

        # Stop prediction if there are invalid inputs
    if errors:
            st.error("❌ Please correct the following input errors:")
            for error in errors:
                st.write(f"- {error}")
            st.stop()

        # Display warnings but allow prediction to continue
    if warnings:
            st.warning("⚠️ Input Validation Warnings")
            for warning in warnings:
                st.write(f"- {warning}")    
    
    
    employment_mapping = {
        "Salaried": 0,
        "Self Employed": 1,
        "Unemployed": 2
    }
    
    employment_encoded = employment_mapping[employment]    


    
    input_data = pd.DataFrame({
        "age": [age],
        "annual_income": [annual_income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "existing_debt": [existing_debt],
        "employment_type": [employment_encoded],
        "previous_defaults": [previous_defaults],
        "dependents": [dependents]
    })


    # -----------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    approval_probability = probabilities[1] * 100

    rejection_probability = probabilities[0] * 100


    # -----------------------------------------------
    # DETERMINE RISK
    # -----------------------------------------------

    if approval_probability >= 75:

        risk_level = "LOW"

    elif approval_probability >= 50:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # -----------------------------------------------
    # RESULT
    # -----------------------------------------------

    st.header("📊 Assessment Result")

    st.subheader("📋 Financial Indicators")

    indicator_col1, indicator_col2 = st.columns(2)

    with indicator_col1:
        st.metric("Debt-to-Income Ratio", f"{dti:.1f}%")

    with indicator_col2:
        st.metric("Loan-to-Income Ratio", f"{lti:.1f}%")

    


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Approval Probability",
            f"{approval_probability:.1f}%"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with result_col3:

        if prediction == 1:

            st.metric(
                "Recommendation",
                "APPROVE"
            )

        else:

            st.metric(
                "Recommendation",
                "REVIEW"
            )


    # -----------------------------------------------
    # RESULT MESSAGE
    # -----------------------------------------------

    if prediction == 1:

        st.success(
            "🟢 The model recommends APPROVAL "
            "based on the applicant's financial profile."
        )

    else:

        st.error(
            "🔴 The application requires additional "
            "review due to elevated predicted risk."
        )


    # -----------------------------------------------
    # PROBABILITY CHART
    # -----------------------------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Outcome": [
            "Approved",
            "Not Approved"
        ],
        "Probability": [
            approval_probability,
            rejection_probability
        ]
    })

    st.bar_chart(
        probability_df.set_index("Outcome")
    )


    # -----------------------------------------------
    # RISK FACTORS
    # -----------------------------------------------

    st.subheader("🔎 Key Risk Factors")


    factors = []


    if credit_score >= 700:

        factors.append(
            "✅ Strong credit score reduces credit risk."
        )

    elif credit_score < 600:

        factors.append(
            "⚠️ Low credit score increases credit risk."
        )

    else:

        factors.append(
            "ℹ️ Credit score is in the moderate range."
        )


    if annual_income >= loan_amount * 2:

        factors.append(
            "✅ Income is relatively strong compared "
            "with the requested loan."
        )

    else:

        factors.append(
            "⚠️ Loan amount is relatively high compared "
            "with annual income."
        )


    if existing_debt < annual_income * 0.2:

        factors.append(
            "✅ Existing debt is relatively low."
        )

    else:

        factors.append(
            "⚠️ Existing debt represents a significant "
            "financial burden."
        )


    if previous_defaults == 0:

        factors.append(
            "✅ No previous payment defaults reported."
        )

    else:

        factors.append(
            "⚠️ Previous payment defaults increase risk."
        )


    if employment == "Salaried":

        factors.append(
            "✅ Salaried employment indicates relatively "
            "stable income."
        )

    elif employment == "Self Employed":

        factors.append(
            "ℹ️ Self-employment may require additional "
            "income verification."
        )

    else:

        factors.append(
            "⚠️ Unemployment significantly increases "
            "credit risk."
        )


    for factor in factors:

        st.write(factor)


    # -----------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------

    st.warning(
        "⚠️ This prediction is generated by a machine "
        "learning prototype using synthetic training data. "
        "It should not be used for actual lending decisions."
    )
    
    st.divider()

st.header("📈 Model Information")

st.write(
    """
    The CrediSense AI model uses a Random Forest
    classification algorithm.

    The model considers multiple financial and
    demographic features to estimate loan approval.
    """
)

st.info(
    "Model trained using an 80/20 train-test split."
)