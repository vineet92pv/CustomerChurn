import pandas as pd
import os
import logging
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataIngestion:
    def __init__(self, input_file: str, output_dir: str, logger):
        self.input_file = input_file
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
    
    def ingest_data(self):
        try:
            # Read the dataset
            df = pd.read_csv(self.input_file)
            
            # Generate timestamped filename
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"bank_churn_raw.csv")
            
            # Save the ingested file
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Data ingestion successful. File saved at {output_file}")
        except Exception as e:
            self.logger.error(f"Error during data ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger = setup_logging()
    input_file = "BankChurners.csv"  # Adjust path if needed
    output_dir = "data/raw"
    ingestion = DataIngestion(input_file, output_dir, logger)
    ingestion.ingest_data()
