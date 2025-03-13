import pandas as pd
import numpy as np
import os
import sys
import logging
import json
import ydata_profiling as pandas_profiling

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataValidation:
    def __init__(self, input_file: str, output_dir: str, logger):
        """Initialize validation with input file and output directory."""
        self.input_file = input_file
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Loads data from the selected file."""
        try:
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"File not found: {self.input_file}")

            df = pd.read_csv(self.input_file)
            self.logger.info(f"✅ Data loaded successfully from {self.input_file}")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}", exc_info=True)
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

    def generate_json_report(self, df, file_name):
        """Generates and saves a JSON validation report for each file."""
        report = {
            "missing_values": DataValidation.check_missing_values(df).to_dict(),
            "duplicate_count": {"count": int(DataValidation.check_duplicates(df))},  # Convert to int
            "data_types": {col: str(dtype) for col, dtype in DataValidation.validate_data_types(df).items()}  # Convert dtype to string
        }
    
        # Convert all NumPy data types to native Python types
        report = json.loads(json.dumps(report, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o))
    
        report_path = os.path.join(self.output_dir, f"{file_name}_validation_report.json")
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        self.logger.info(f"✅ Validation report saved at {report_path}")

    def generate_html_report(self, df, file_name):
        """Generates and saves an interactive HTML validation report using Pandas Profiling."""
        try:
            profile = df.profile_report(title=f"Data Validation Report - {file_name}")
            report_path = os.path.join(self.output_dir, f"{file_name}_validation_report.html")
            profile.to_file(report_path)

            self.logger.info(f"✅ HTML Validation report saved at {report_path}")
        except Exception as e:
            self.logger.error(f"❌ Error generating HTML report: {str(e)}", exc_info=True)

    def run_validations(self):
        """Runs all data validation steps and generates reports."""
        df = self.load_data()
        if df is not None:
            file_name = os.path.basename(self.input_file).replace(".csv", "")
            self.generate_json_report(df, file_name)
            self.generate_html_report(df, file_name)

def run_data_validation():
    """Main function to execute data validation on available files."""
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(BASE_DIR, "data/raw")

    # File names
    file_candidates = {
        "api": "bank_churn_api_raw.csv",
        "loc": "bank_churn_loc_raw.csv"
    }

    files_found = False  # Flag to check if any file is processed

    # Process each file if it exists
    for key, file_name in file_candidates.items():
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            files_found = True  # At least one file is found
            logger.info(f"📂 Running data validation for {key.upper()} Data: {file_path}")
            output_dir = os.path.join(BASE_DIR, "data/validation")
            validation = DataValidation(file_path, output_dir, logger)
            validation.run_validations()
        else:
            logger.warning(f"⚠️ {key.upper()} data file not found: {file_path}")

    if not files_found:
        logger.warning("⚠️ No data files found for validation. Exiting.")

if __name__ == "__main__":
    run_data_validation()
