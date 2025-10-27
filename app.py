import streamlit as st
import pandas as pd
import joblib
import os

#Loading the saved model
# Use a function with st.cache_resource to load the model only once.
@st.cache_resource
def load_model(model_path):
    """Loads the saved model pipeline from the specified path."""
    model = joblib.load(model_path)
    return model

# Define the path to the model
MODEL_PATH = os.path.join("models", "churn_prediction_model.joblib")
model = load_model(MODEL_PATH)

# --- App Title and Description ---
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")
st.title('Customer Churn Prediction App')
st.markdown("This app predicts whether a telecom customer is likely to churn based on their account details. The model was trained on the Telco Customer Churn dataset.")

# --- Create Input Fields for User in a structured layout ---
st.header("Enter Customer Details")

# Using columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ('Male', 'Female'))
    partner = st.selectbox("Has a Partner?", ('Yes', 'No'))
    dependents = st.selectbox("Has Dependents?", ('Yes', 'No'))
    phone_service = st.selectbox("Has Phone Service?", ('Yes', 'No'))
    paperless_billing = st.selectbox("Uses Paperless Billing?", ('Yes', 'No'))
    
with col2:
    multiple_lines = st.selectbox("Multiple Lines", ('No phone service', 'No', 'Yes'))
    internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox("Online Security", ('No internet service', 'No', 'Yes'))
    online_backup = st.selectbox("Online Backup", ('No internet service', 'No', 'Yes'))
    payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

with col3:
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=24)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=70.0)
    # Since TotalCharges is tenure * MonthlyCharges, we can calculate it
    # We add a small number to avoid division by zero if tenure is 0.
    total_charges = tenure * monthly_charges


# --- Prediction Logic and Button ---
if st.button("Predict Churn", type="primary"):
    # Create a dictionary from the user inputs that matches the model's training columns
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # Defaulting other features as they were not included for simplicity
        # In a real app, you'd want inputs for all features.
        'SeniorCitizen': 0, 
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month'
    }

    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make a prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the result
    st.subheader("Prediction Result")
    churn_probability = prediction_proba[0][1]

    if prediction[0] == 1:
        st.error(f"Prediction: **Churn** (Probability: {churn_probability:.2%})", icon="🚨")
        st.warning("This customer is at a high risk of leaving. Consider offering a retention incentive.")
    else:
        st.success(f"Prediction: **Stay** (Probability of Staying: {1-churn_probability:.2%})", icon="✅")
        st.info("This customer is likely to stay.")