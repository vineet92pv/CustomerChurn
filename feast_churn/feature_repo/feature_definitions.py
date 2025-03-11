from datetime import timedelta

import pandas as pd
import os

from feast import (
    Entity,
    Feature,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64,Bool

from feast import FeatureStore

# 01 - Define a project for the feature repo
project = Project(name="feast_churn", description="A project for Bank Customer Churn")

# 02 - Define the data source
customer_data_source = FileSource(
    path="data/bank_churn.parquet",  # Update with actual path
    event_timestamp_column="event_timestamp",

)

# 03 - Define the entity
customer = Entity(name="Customer_ID", value_type=ValueType.FLOAT, description="Customer unique identifier")

# 04 - Define the feature view
customer_feature_view = FeatureView(
    name="customer_features",
    entities=[customer],  # Ensure this references the Entity object
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

if __name__ == "__main__":
    feast_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../feast_churn/feature_repo"))
    store = FeatureStore(repo_path=feast_project_dir)  # Ensure the correct path to your Feast repo
    store.apply([customer, customer_feature_view])


































































































