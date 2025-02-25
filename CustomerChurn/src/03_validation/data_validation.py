import pandas as pd
import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataValidation:
    def __init__(self, input_file: str, output_dir: str, logger):
        self.input_file = input_file
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        try:
            df = pd.read_csv(self.input_file)
            self.logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def check_missing_values(df):
        return df.isnull().sum()
    
    @staticmethod
    def check_duplicates(df):
        return df.duplicated().sum()
    
    @staticmethod
    def validate_data_types(df):
        return df.dtypes
    
    def generate_report(self, df):
        report = {
            "missing_values": DataValidation.check_missing_values(df).to_dict(),
            "duplicate_count": {"count": {"count": {"count": DataValidation.check_duplicates(df)}}},
            "data_types": DataValidation.validate_data_types(df).apply(str).to_dict()
        }
        report_path = os.path.join(self.output_dir, "data_validation_report.json")
        pd.DataFrame([report]).to_json(report_path, indent=4)
        self.logger.info(f"Validation report saved at {report_path}")
    
    def run_validations(self):
        df = self.load_data()
        if df is not None:
            self.generate_report(df)

def run_data_validation():
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_file = os.path.join(BASE_DIR, "data/raw/bank_churn_raw.csv")  # Adjust path if needed
    output_dir = os.path.join(BASE_DIR, "data/validation")
    validation = DataValidation(input_file, output_dir, logger)
    validation.run_validations()

if __name__ == "__main__":
    run_data_validation()
