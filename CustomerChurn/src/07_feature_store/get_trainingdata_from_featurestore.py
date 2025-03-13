from feast import FeatureStore
import pandas as pd
import os

# Initialize the feature store
# store = FeatureStore(repo_path="feast_project/feature_repo")  # Update with actual path
store = FeatureStore(repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../feast_churn/feature_repo")))

# Define the feature references (feature view name and feature names)
feature_refs = [
    "customer_features:Credit_Score",
    "customer_features:Age",
    "customer_features:Tenure",
    "customer_features:Balance",
    "customer_features:Num_Products",
    "customer_features:Has_Credit_Card",
    "customer_features:Is_Active",
    "customer_features:Salary",
    "customer_features:Churn",
    "customer_features:Country_Germany",
    "customer_features:Country_Spain",
    "customer_features:Gender_Male",
    "customer_features:Customer_Tenure_Years",
    "customer_features:Activity_Frequency",
    "customer_features:Avg_Balance_per_Product",
    "customer_features:Credit_Card_Utilization",
    "customer_features:Salary_Normalized",
    "customer_features:Spending_Power_Index",
    "customer_features:High_Value_Customer",
]

# Load entity dataframe with customer_id and event_timestamp
# df_entities = pd.DataFrame({
#     "Customer_ID": [1001, 1002, 1003]  # Replace with actual customer IDs
#     ,"event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),  # Replace with actual timestamps
# })

df_entities = pd.DataFrame({
    "Customer_ID": [15771669]  
    ,"event_timestamp": pd.to_datetime(["2025-03-11 18:51:33.250981+00:00"]),  # Replace with actual timestamps
})


# Fetch historical features
df_training = store.get_historical_features(
    entity_df=df_entities,
    features=feature_refs
).to_df()

# print("----- Feature schema -----\n")
# print(df_training.info())

print()
print("----- Example features -----\n")
print(df_training.head())
