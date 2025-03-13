import pandas as pd
fs_df=pd.read_parquet("/home/vineeth/customer_churn_airflow/feast_churn/feature_repo/data/bank_churn.parquet")
print(fs_df)
print(fs_df.info) 


