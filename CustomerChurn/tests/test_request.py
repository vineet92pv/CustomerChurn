import requests

url = "http://127.0.0.1:5002/predict"

# Sample input data with all expected features
data = {
    "Customer_ID": 123456,
    "Credit_Score": 700,
    "Age": 35,
    "Tenure": 5,
    "Balance": 12000.50,
    "Num_Products": 2,
    "Has_Credit_Card": 1,
    "Is_Active": 0,
    "Salary": 75000.00,
    "Country_Germany": 0,
    "Country_Spain": 1,
    "Gender_Male": 1,
    "Customer_Tenure_Years": 4.5,
    "Activity_Frequency": 0.75,
    "Avg_Balance_per_Product": 6000.25,
    "Credit_Card_Utilization": 0.3,
    "Salary_Normalized": 0.85,
    "Spending_Power_Index": 1.2,
    "High_Value_Customer": 0
}

# Send request
response = requests.post(url, json=data)

# Print response
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
