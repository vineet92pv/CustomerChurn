import os
import logging
import glob
import sys
import shutil  # For moving files safely
import subprocess  # Better command execution
from datetime import datetime  # For timestamping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataIngestion:
    def __init__(self, dataset_name: str, download_dir: str, output_dir: str, logger):
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        self.output_dir = output_dir
        self.logger = logger

        # Ensure directories exist
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def ingest_data(self):
        try:
            # Construct Kaggle API command
            command = f"kaggle datasets download -d {self.dataset_name} -p {self.download_dir} --unzip"

            # Execute the command and check for errors
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"❌ Kaggle download failed: {result.stderr}")
                return

            # Find the downloaded CSV files
            csv_files = glob.glob(os.path.join(self.download_dir, "*.csv"))
            if not csv_files:
                self.logger.error("❌ No CSV file found after download.")
                return

            # Pick the correct CSV file (if multiple exist, choose the largest)
            downloaded_file = max(csv_files, key=os.path.getsize)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d")

            # Define partitioned directory structure
            partitioned_dir = os.path.join(self.output_dir, f"source=kaggle_api/type=churn/timestamp={timestamp}")
            os.makedirs(partitioned_dir, exist_ok=True)

            # Define file paths
            partitioned_file = os.path.join(partitioned_dir, "bank_churn_api_raw.csv")
            backup_file = os.path.join(self.output_dir, f"bank_churn_api_raw.csv")

            # Move the file to both locations
            shutil.move(downloaded_file, partitioned_file)
            shutil.copy(partitioned_file, backup_file)  # Backup copy

            self.logger.info(f"✅ Data ingestion successful. Files saved at:\n  - {partitioned_file}\n  - {backup_file}")

        except Exception as e:
            self.logger.error(f"❌ Error during data ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger = setup_logging()

    # Dynamically set the base directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Define dataset name, download directory, and output directory
    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    download_dir = os.path.join(BASE_DIR, "data/api_downloaded")  
    output_dir = os.path.join(BASE_DIR, "data/raw")

    # ✅ Fixed Argument Order
    ingestion = DataIngestion(dataset_name, download_dir, output_dir, logger)
    ingestion.ingest_data()
