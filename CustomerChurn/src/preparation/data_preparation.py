import pandas as pd
import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataPreparation:
    def __init__(self, input_file: str, output_dir: str, logger):
        self.input_file = input_file
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define columns to transform
        self.numeric_cols = ["Credit_Limit", "Total_Trans_Amt", "Total_Revolving_Bal"]
        self.categorical_cols = ["Gender", "Income_Category", "Education_Level", "Marital_Status", "Card_Category"]
    
    def load_data(self):
        try:
            df = pd.read_csv(self.input_file)
            self.logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def handle_missing_values(df):
        logging.info("Handling missing values.")
        df.ffill(inplace=True)  # Forward fill as a simple strategy
        return df
    
    def encode_categorical(self, df):
        logging.info("Encoding categorical variables.")
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)  # One-hot encoding for selected columns
        return df
    
    def standardize_numerical(self, df):
        logging.info("Standardizing numerical features.")
        df[self.numeric_cols] = (df[self.numeric_cols] - df[self.numeric_cols].mean()) / df[self.numeric_cols].std()
        return df
    
    @staticmethod
    def remove_duplicates(df):
        logging.info("Removing duplicate records.")
        df = df.drop_duplicates()
        return df
    
    def transform(self, df):
        self.logger.info("Applying data transformations.")
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        df = self.standardize_numerical(df)
        df = self.remove_duplicates(df)
        return df
    
    def save_cleaned_data(self, df):
        output_file = os.path.join(self.output_dir, "cleaned_data.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Cleaned data saved at {output_file}")
    
    def run_preparation(self):
        df = self.load_data()
        if df is not None:
            df = self.transform(df)
            self.save_cleaned_data(df)

def run_data_preparation():
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_file = os.path.join(BASE_DIR, "data/raw/bank_churn_raw.csv")  # Adjust path if needed
    output_dir = os.path.join(BASE_DIR, "data/processed")
    preparation = DataPreparation(input_file, output_dir, logger)
    preparation.run_preparation()

if __name__ == "__main__":
    run_data_preparation()
