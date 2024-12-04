
# Hair Salon Data Analysis

## Overview

This repository contains a comprehensive analysis of customer data from a hair salon business. The analysis aims to uncover valuable insights about customer behavior, spending patterns, and preferences, which can help improve customer satisfaction, optimize marketing efforts, and enhance overall business performance.

## Dataset Description

The dataset consists of several features related to customer demographics, visit behavior, and spending. Below is a summary of the key columns in the dataset:

- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male/Female).
- **Visit_Frequency**: Number of visits within a specific period.
- **Service_Type**: Type of service used (e.g., Haircut, Treatment, Hair Coloring).
- **Average_Spend_Per_Visit**: Average amount spent per visit.
- **Total_Spend**: Total amount spent by the customer over a defined period.
- **Loyalty_Program**: Indicates if the customer is enrolled in a loyalty program (Yes/No).
- **Promotional_Response**: Response to promotional campaigns (Yes/No).
- **Feedback_Score**: Customer satisfaction score (1–5).

## Objectives

The analysis addresses the following business questions:

1. What are the primary customer demographics (e.g., age, gender)?
2. Which services are most popular among customers?
3. What is the impact of loyalty programs on customer spending and visit frequency?
4. How responsive are customers to promotional campaigns?
5. What factors influence customer satisfaction scores?

## Features

- **Exploratory Data Analysis (EDA)**:
  - Summary statistics and visualizations of customer demographics and behaviors.
  - Distribution analysis of age, spending, and visit frequency.
  - Service popularity analysis.

- **Loyalty Program Effectiveness**:
  - Comparisons between enrolled and non-enrolled customers regarding total spending and visit frequency.

- **Promotion Response Analysis**:
  - Assessment of customer segments that respond to promotions.

- **Customer Satisfaction Insights**:
  - Correlations between feedback scores and other features.

## Key Libraries

The project leverages the following Python libraries:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For creating visualizations.
- **ydata-profiling**: For automated data profiling and summary.

## Results

- **Customer Insights**: Identification of high-value customers and frequent visitors.
- **Service Preferences**: Most customers prefer specific services like haircuts or treatments.
- **Marketing Recommendations**: Insights into customer responsiveness to promotions.
- **Satisfaction Drivers**: Factors influencing higher feedback scores.

## Usage

To explore the analysis:

1. Clone this repository:
   ```bash
   git clone https://github.com/aminmoghadasi75/Hair_Saloon.git
   cd hair-salon-data-analysis
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook hair_salon.ipynb
   ```

## Future Work

- Incorporating predictive models to forecast customer spending or visit frequency.
- Enhancing analysis with additional datasets (e.g., competitor data, external market trends).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.


