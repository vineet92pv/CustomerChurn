import pandas as pd
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
        df["Customer_Tenure"] = df["Months_on_book"]
        df["Transaction_Frequency"] = df["Total_Trans_Ct"] / df["Months_on_book"]
        df["Avg_Transaction_Amount"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"]
        df["Credit_Utilization_Ratio"] = df["Total_Revolving_Bal"] / df["Credit_Limit"]
        df["Activity_Rate"] = df["Total_Trans_Ct"] / (df["Months_on_book"] * 30)  # Approx daily activity
        df["Spending_Change_Rate"] = df["Total_Amt_Chng_Q4_Q1"] * df["Total_Trans_Amt"]
        return df
    
    def save_transformed_data(self, df):
        output_file = os.path.join(self.output_dir, "transformed_data.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Transformed data saved at {output_file}")
    
    def store_in_database(self, df):
        # Ensure the database directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql("transformed_data", conn, if_exists="replace", index=False)
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
    input_file = os.path.join(BASE_DIR, "data/processed/cleaned_data.csv")  # Load cleaned data
    output_dir = os.path.join(BASE_DIR, "data/transformed")
    db_path = os.path.join("data/database/churn_data.db")
    transformation = DataTransformation(input_file, output_dir, db_path, logger)
    transformation.run_transformation()

if __name__ == "__main__":
    run_data_transformation()
