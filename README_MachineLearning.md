# Promotional Response Modeling with Machine Learning

This project demonstrates a machine learning pipeline for predicting customer responses to promotions. It includes data preprocessing, feature selection, model evaluation, and visualization of learning curves and feature importances. The final model is saved for deployment.

## Features

- **Data Preprocessing:** Handles both numerical and categorical data using pipelines.
- **Model Training:** Supports multiple classifiers including:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Feature Selection:** Utilizes `SelectKBest` for identifying the most relevant features.
- **Evaluation Metrics:**
  - Cross-validation balanced accuracy
  - Learning and error curves
  - Classification report
- **Visualization:** Provides insights into model learning and feature importance.
- **Model Saving:** Saves the trained model using `joblib`.

## Requirements

The project uses the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `joblib`

Install dependencies using:

pip install pandas numpy scikit-learn seaborn matplotlib joblib


## Getting Started

### Data Requirements

Ensure the input DataFrame contains the following columns:
- **Target Column:**
  - `Promotional_Response`: Binary target variable (1 for positive response, 0 for negative).
- **Numerical Features:**
  - `Age`
  - `Visit_Frequency`
  - `Average_Spend_Per_Visit`
  - `Total_Spend`
- **Categorical Features:**
  - `Gender`
  - `Service_Type`
  - `Loyalty_Program`
  - `Age_Group`
  - `Visit_Frequency_Group`
  - `Customer_Value`

### Usage

1. Prepare your data in a Pandas DataFrame with the required columns.
2. Run the script to train models and visualize results.
3. The best model is saved as `BestModelName_Promotional_Response_model.pkl`.



### Outputs

- **Learning Curves:** Evaluates the training and validation performance across different dataset sizes.
- **Error Curves:** Visualizes training and validation errors.
- **Feature Importance Plot:** Highlights the top 10 features driving model predictions.
- **Classification Report:** Summarizes model performance on the test set.
- **Saved Model:** Trained pipeline stored as a `.pkl` file.


