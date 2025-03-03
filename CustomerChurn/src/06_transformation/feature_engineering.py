import pandas as pd
import numpy as np
import os
import sys
import logging
import sqlite3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataTransformation:
    def __init__(self, input_file: str, output_dir: str, db_path: str, logger):
        self.input_file = input_file
        self.output_dir = output_dir
        self.db_path = db_path
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        try:
            df = pd.read_csv(self.input_file)
            self.logger.info("Cleaned data loaded successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading cleaned data: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def feature_engineering(df):
        logging.info("Performing feature engineering")

        # Customer tenure estimation (if 'Tenure' is in months)
        df["Customer_Tenure_Years"] = df["Tenure"] / 12  

        # Activity frequency: Is_Active * Tenure (Gives weight to long-term active members)
        df["Activity_Frequency"] = df["Is_Active"] * df["Tenure"]

        # Average balance per product (Avoid division by zero)
        df["Avg_Balance_per_Product"] = df["Balance"] / df["Num_Products"]
        df.loc[df["Avg_Balance_per_Product"].isin([np.inf, -np.inf]), "Avg_Balance_per_Product"] = 0

        # Credit card impact: If a customer has a credit card, consider their balance level
        df["Credit_Card_Utilization"] = df["Has_Credit_Card"] * df["Balance"]

        # Normalizing Salary (Min-Max scaling)
        df["Salary_Normalized"] = (df["Salary"] - df["Salary"].min()) / (df["Salary"].max() - df["Salary"].min())

        # Spending Power Index (Balance to Salary ratio)
        df["Spending_Power_Index"] = df["Balance"] / df["Salary"]
        df.loc[df["Spending_Power_Index"].isin([np.inf, -np.inf]), "Spending_Power_Index"] = 0

        # Interaction Features
        df["High_Value_Customer"] = ((df["Balance"] > df["Balance"].median()) & (df["Salary"] > df["Salary"].median())).astype(int)

        logging.info("Feature engineering completed successfully.")
        return df    
    
    def save_transformed_data(self, df):
        output_file = os.path.join(self.output_dir, "bank_churn_transformed.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Transformed data saved at {output_file}")
    
    def store_in_database(self, df):
        # Ensure the database directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql("bank_churn_transformed", conn, if_exists="replace", index=False)
            conn.close()
            self.logger.info("Transformed data stored in database successfully.")
        except Exception as e:
            self.logger.error(f"Error storing data in database: {str(e)}", exc_info=True)
    
    def run_transformation(self):
        df = self.load_data()
        if df is not None:
            df = self.feature_engineering(df)
            self.save_transformed_data(df)
            self.store_in_database(df)

def run_data_transformation():
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_file = os.path.join(BASE_DIR, "data/processed/bank_churn_processed.csv")  # Load cleaned data
    output_dir = os.path.join(BASE_DIR, "data/transformed")
    db_path = os.path.join(BASE_DIR, "data/database/churn_data.db")
    transformation = DataTransformation(input_file, output_dir, db_path, logger)
    transformation.run_transformation()

if __name__ == "__main__":
    run_data_transformation()
