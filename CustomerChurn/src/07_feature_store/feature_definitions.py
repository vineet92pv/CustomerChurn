from datetime import timedelta
import pandas as pd
import os
import warnings

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    FeatureStore,
    ValueType
)
from feast.types import Float32, Int64, Bool

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Define the project
project = Project(name="feast_churn", description="A project for Bank Customer Churn")

# Define the data source
customer_data_source = FileSource(
    path="data/bank_churn.parquet",  # Ensure this path is correct
    event_timestamp_column="event_timestamp",
)

# Define the entity with explicit value_type to fix warning
customer = Entity(name="Customer_ID", value_type=ValueType.INT64, description="Customer unique identifier")

# Define a dummy entity to prevent internal warnings
dummy_entity = Entity(name="__dummy", value_type=ValueType.STRING, description="Dummy entity")

# Define the feature view
customer_feature_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=None,
    schema=[
        Field(name="Credit_Score", dtype=Float32),
        Field(name="Age", dtype=Float32),
        Field(name="Tenure", dtype=Float32),
        Field(name="Balance", dtype=Float32),
        Field(name="Num_Products", dtype=Float32),
        Field(name="Has_Credit_Card", dtype=Float32),
        Field(name="Is_Active", dtype=Float32),
        Field(name="Salary", dtype=Float32),
        Field(name="Churn", dtype=Int64),
        Field(name="Country_Germany", dtype=Bool),
        Field(name="Country_Spain", dtype=Bool),
        Field(name="Gender_Male", dtype=Bool),
        Field(name="Customer_Tenure_Years", dtype=Float32),
        Field(name="Activity_Frequency", dtype=Float32),
        Field(name="Avg_Balance_per_Product", dtype=Float32),
        Field(name="Credit_Card_Utilization", dtype=Float32),
        Field(name="Salary_Normalized", dtype=Float32),
        Field(name="Spending_Power_Index", dtype=Float32),
        Field(name="High_Value_Customer", dtype=Int64),
    ],
    online=True,
    source=customer_data_source,
)

# Apply the changes to the Feast feature store
if __name__ == "__main__":
    feast_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../feast_churn/feature_repo"))
    store = FeatureStore(repo_path=feast_project_dir)
    
    # Apply both entities and feature view
    store.apply([customer, dummy_entity, customer_feature_view])

    print("✅ Feature store updated successfully!")
