import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model

def load_model():
    return joblib.load("LogisticRegression_Promotional_Response_model.pkl")

# Main App
def main():
    st.title("Promotional Response Prediction")
    
    # Input Form
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
        feedback_score = st.selectbox(
            "FeedBack Score",
            [1,2,3,4,5],
            help="Select the customer's feedback score.",
        )

        
        submit_button = st.form_submit_button(label="Predict Likelihood")

    # When Submit Button is Clicked
    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender],
                "Visit_Frequency": [visit_frequency],
                "Service_Type": [service_type],
                "Average_Spend_Per_Visit": [avg_spend],
                "Total_Spend": [total_spend],
                "Loyalty_Program": [loyalty_program],
                "Feedback_Score": [feedback_score],
            }
        )

        # Feature Engineering
        input_data["Age_Group"] = np.select(
            [
                (input_data['Age'] >= 1) & (input_data['Age'] <= 18),
                (input_data['Age'] >= 19) & (input_data['Age'] <= 30),
                (input_data['Age'] >= 31) & (input_data['Age'] <= 45),
                (input_data['Age'] >= 46) & (input_data['Age'] <= 60),
                (input_data['Age'] >= 61),
            ],
            ['Teenager', 'Young Adult', 'Adult', 'Middle Age', 'Senior'],
            default='Unknown'
        )

        input_data["Customer_Value"] = np.select(
            [
                (input_data['Total_Spend'] >= 1) & (input_data['Total_Spend'] <= 200),
                (input_data['Total_Spend'] >= 201) & (input_data['Total_Spend'] <= 600),
                (input_data['Total_Spend'] >= 601),
            ],
            ['Low', 'Medium', 'High'],
            default='Unknown'
        )

        input_data["Visit_Frequency_Group"] = np.select(
            [
                (input_data['Visit_Frequency'] >= 1) & (input_data['Visit_Frequency'] <= 4),
                (input_data['Visit_Frequency'] >= 5) & (input_data['Visit_Frequency'] <= 8),
                (input_data['Visit_Frequency'] >= 9),
            ],
            ['Rare Visitors', 'Occasional Visitors', 'Frequent Visitors'],
            default='Unknown'
        )

        # Load Model and Predict
        model = load_model()
        prediction = model.predict(input_data)[0]
        likelihood = "Likely" if prediction == 1 else "Unlikely"

        st.write(f"The customer is **{likelihood}** to respond to promotions.")

if __name__ == "__main__":
    main()
