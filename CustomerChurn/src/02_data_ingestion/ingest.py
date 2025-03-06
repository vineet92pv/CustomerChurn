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
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d")
            
            # Define partitioned directory structure
            partitioned_dir = os.path.join(self.output_dir, f"source=local/type=churn/timestamp={timestamp}")
            os.makedirs(partitioned_dir, exist_ok=True)
            
            # Define file paths
            partitioned_file = os.path.join(partitioned_dir, "bank_churn_loc_raw.csv")
            backup_file = os.path.join(self.output_dir, f"bank_churn_loc_raw.csv")
            
            # Save the ingested file in both locations
            df.to_csv(partitioned_file, index=False)
            df.to_csv(backup_file, index=False)
            
            self.logger.info(f"Data ingestion successful. Files saved at:\n  - {partitioned_file}\n  - {backup_file}")
        except Exception as e:
            self.logger.error(f"Error during data ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    BASE_DIR_IN = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    input_file = os.path.join(BASE_DIR_IN, "bank_churn_loc.csv") 
    output_dir = os.path.join(BASE_DIR, "data/raw")
    
    ingestion = DataIngestion(input_file, output_dir, logger)
    ingestion.ingest_data()
