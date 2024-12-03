import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
feedback_model = joblib.load('/home/amin/my_projects/Machine_learning-Projects/Hair_saloon/src/LogisticRegression_Feedback_Score_model.pkl')
promo_model = joblib.load('/home/amin/my_projects/Machine_learning-Projects/Hair_saloon/src/LogisticRegression_Promotional_Response_model.pkl')

# Page Configuration
st.set_page_config(
    page_title="Customer Insights Predictor",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header Section
st.title("🌟 Customer Insights Predictor 🌟")
st.markdown(
    """
    Welcome to the Customer Insights Predictor! Use this app to:
    - Predict customer feedback scores based on their details.
    - Assess the likelihood of customers responding to promotional offers.
    """
)

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a functionality:",
    ["Feedback Score Prediction", "Promotional Response Prediction"]
)

# Common Form Inputs
def input_form():
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1, help="Enter the customer's age.")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
    visit_frequency = st.number_input("Visit Frequency (per month)", min_value=0, max_value=50, value=1, step=1)
    service_type = st.selectbox(
        "Service Type",
        ["Haircut", "Treatment", "Hair Coloring", "Styling"],
        help="Select the type of service the customer used."
    )
    avg_spend = st.number_input(
        "Average Spend Per Visit ($)", min_value=0.0, step=0.01, value=50.0, help="Enter the average spend per visit."
    )
    total_spend = st.number_input(
        "Total Spend ($)", min_value=0.0, step=0.01, value=500.0, help="Enter the total spend of the customer."
    )
    loyalty_program = st.radio(
        "Is the customer in the loyalty program?", ["Yes", "No"], help="Choose if the customer is part of the loyalty program."
    )
    return {
        "Age": age,
        "Gender": gender,
        "Visit_Frequency": visit_frequency,
        "Service_Type": service_type,
        "Average_Spend_Per_Visit": avg_spend,
        "Total_Spend": total_spend,
        "Loyalty_Program": loyalty_program
    }

# Feature Engineering Function
def feature_engineering(data):
    data["Age_Group"] = np.select(
        [
            (data['Age'] >= 1) & (data['Age'] <= 18),
            (data['Age'] >= 19) & (data['Age'] <= 30),
            (data['Age'] >= 31) & (data['Age'] <= 45),
            (data['Age'] >= 46) & (data['Age'] <= 60),
            (data['Age'] >= 61)
        ],
        ['Teenager', 'Young Adult', 'Adult', 'Middle Age', 'Senior'],
        default='Unknown'
    )
    data["Customer_Value"] = np.select(
        [
            (data['Total_Spend'] >= 1) & (data['Total_Spend'] <= 200),
            (data['Total_Spend'] >= 201) & (data['Total_Spend'] <= 600),
            (data['Total_Spend'] >= 601)
        ],
        ['Low', 'Medium', 'High'],
        default='Unknown'
    )
    data["Visit_Frequency_Group"] = np.select(
        [
            (data['Visit_Frequency'] >= 1) & (data['Visit_Frequency'] <= 4),
            (data['Visit_Frequency'] >= 5) & (data['Visit_Frequency'] <= 8),
            (data['Visit_Frequency'] >= 9)
        ],
        ['Rare Visitors', 'Occasional Visitors', 'Frequent Visitors'],
        default='Unknown'
    )
    return data

# App Modes
if app_mode == "Feedback Score Prediction":
    st.header("🔮 Feedback Score Prediction")
    with st.form(key="feedback_form"):
        inputs = input_form()
        Promotional_Response =  st.radio(
        "Does the customer respond to promotions?", ["Yes", "No"], help="Choose if the customer responds to promotional offers."
    )

        inputs["Promotional_Response"] = Promotional_Response
        submit_button = st.form_submit_button(label="Predict Feedback Score")
    
    if submit_button:
        input_data = pd.DataFrame([inputs])
        input_data = feature_engineering(input_data)
        feedback_score = feedback_model.predict(input_data)[0]
        st.subheader("Predicted Feedback Score")
        st.metric(label="Feedback Score", value=f"{feedback_score:.2f}")
        st.success("Prediction complete! 🎉")

elif app_mode == "Promotional Response Prediction":
    st.header("📊 Promotional Response Prediction")
    with st.form(key="promo_form"):
        inputs = input_form()
        feedback_score = st.selectbox(
            "Feedback Score",
            [1, 2, 3, 4, 5],
            help="Select the customer's feedback score."
        )
        inputs["Feedback_Score"] = feedback_score
        submit_button = st.form_submit_button(label="Predict Likelihood")
    
    if submit_button:
        input_data = pd.DataFrame([inputs])
        input_data = feature_engineering(input_data)
        prediction = promo_model.predict(input_data)[0]
        likelihood = "Likely" if prediction == 1 else "Unlikely"
        st.subheader("Promotional Response Likelihood")
        st.write(f"The customer is **{likelihood}** to respond to promotions.")
        st.balloons()
