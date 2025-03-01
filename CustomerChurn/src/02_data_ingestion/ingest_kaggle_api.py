import os
import logging
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataIngestion:
    def __init__(self, dataset_name: str, output_dir: str, output_filename: str, logger):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)

    def ingest_data(self):
        try:
            # Construct Kaggle download command
            command = f"kaggle datasets download -d {self.dataset_name} -p {self.output_dir} --unzip"
            
            # Execute the command
            os.system(command)

            # Find the downloaded CSV file
            csv_files = glob.glob(os.path.join(self.output_dir, "*.csv"))
            if csv_files:
                downloaded_file = csv_files[0]  # Assuming only one CSV is downloaded
                
                # Rename it to required filename
                new_file_path = os.path.join(self.output_dir, self.output_filename)
                os.rename(downloaded_file, new_file_path)
                self.logger.info(f"✅ Data ingestion successful. File saved as {new_file_path}")
            else:
                self.logger.error("❌ No CSV file found after download.")

        except Exception as e:
            self.logger.error(f"❌ Error during data ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger = setup_logging()
    
    # Dynamically set the base directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Define dataset name, output directory, and desired filename
    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    output_dir = os.path.join(BASE_DIR, "data/raw")
    output_filename = "bank_churn_api_raw.csv"
    
    # Run ingestion
    ingestion = DataIngestion(dataset_name, output_dir, output_filename, logger)
    ingestion.ingest_data()
