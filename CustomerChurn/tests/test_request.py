import requests

url = "http://127.0.0.1:5000/predict"

# Sample input data including all expected features
data = {
    "Credit_Limit": 10000,
    "Total_Trans_Amt": 3000,
    "Total_Revolving_Bal": 500,
    "Avg_Transaction_Amount": 200,
    "Activity_Rate": 0.03,
    "Avg_Open_To_Buy": 5000,
    "Avg_Utilization_Ratio": 0.25,
    "Card_Category_Gold": 0,
    "Card_Category_Platinum": 1
}

# Send request
response = requests.post(url, json=data)

# Print response
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
