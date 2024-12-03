import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('LogisticRegression_Feedback_Score_model.pkl')

# Set up the page configuration
st.set_page_config(
    page_title="Feedback Score Predictor",
    page_icon="💇‍♀️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Page Header
st.title("✨ Customer Feedback Score Predictor ✨")
st.write("Predict the feedback score based on customer details using a trained machine learning model.")

# Input form
with st.form(key="customer_form"):
    st.subheader("Customer Details")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1, help="Enter the customer's age.")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
    visit_frequency = st.number_input("Visit Frequency (per month)", min_value=0, max_value=50, value=1, step=1)
    service_type = st.selectbox(
        "Service Type",
        ["Haircut", "Treatment", "Hair Coloring", "Styling"],
        help="Select the type of service the customer used.",
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
    promotional_response = st.radio(
        "Does the customer respond to promotions?", ["Yes", "No"], help="Choose if the customer responds to promotional offers."
    )

    # Submit button
    submit_button = st.form_submit_button(label="Predict Feedback Score")

    
        

# Predict and display results
if submit_button:
    # Convert inputs to DataFrame
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Visit_Frequency": [visit_frequency],
            "Service_Type": [service_type],
            "Average_Spend_Per_Visit": [avg_spend],
            "Total_Spend": [total_spend],
            "Loyalty_Program": [loyalty_program ],
            "Promotional_Response": [promotional_response],


        }
    )
     # Add derived columns
    # 1. Age Group
    age_conditions = [
        (input_data['Age'] >= 1) & (input_data['Age'] <= 18),
        (input_data['Age'] >= 19) & (input_data['Age'] <= 30),
        (input_data['Age'] >= 31) & (input_data['Age'] <= 45),
        (input_data['Age'] >= 46) & (input_data['Age'] <= 60),
        (input_data['Age'] >= 61),
    ]
    age_values = ['Teenager', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
    input_data['Age_Group'] = np.select(age_conditions, age_values, default='Unknown')

    # 2. Customer Value
    spend_conditions = [
        (input_data['Total_Spend'] >= 1) & (input_data['Total_Spend'] <= 200),
        (input_data['Total_Spend'] >= 201) & (input_data['Total_Spend'] <= 600),
        (input_data['Total_Spend'] >= 601),
    ]
    spend_values = ['Low', 'Medium', 'High']
    input_data['Customer_Value'] = np.select(spend_conditions, spend_values, default='Unknown')

    # 3. Visit Frequency Group
    visit_conditions = [
        (input_data['Visit_Frequency'] >= 1) & (input_data['Visit_Frequency'] <= 4),
        (input_data['Visit_Frequency'] >= 5) & (input_data['Visit_Frequency'] <= 8),
        (input_data['Visit_Frequency'] >= 9) & (input_data['Visit_Frequency'] <= 12),
    ]
    visit_values = ['Rare Visitors', 'Occasional Visitors', 'Frequent Visitors']
    input_data['Visit_Frequency_Group'] = np.select(visit_conditions, visit_values, default='Unknown')


    # Predict the feedback score
    feedback_score = model.predict(input_data)[0]

    # Display the prediction
    st.subheader("Predicted Feedback Score")
    st.metric(label="Feedback Score", value=f"{feedback_score:.2f}", delta=None)
    st.success("Prediction complete! 🎉")


