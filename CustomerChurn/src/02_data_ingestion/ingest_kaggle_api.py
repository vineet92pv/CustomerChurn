import os
import logging
import glob
import sys
import shutil  # For moving files safely
import subprocess  # Better command execution

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataIngestion:
    def __init__(self, dataset_name: str, download_dir: str, output_dir: str, output_filename: str, logger):
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        self.output_dir = output_dir
        self.output_filename = output_filename
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

            # Rename and move to output directory
            new_file_path = os.path.join(self.output_dir, self.output_filename)
            shutil.move(downloaded_file, new_file_path)

            self.logger.info(f"✅ Data ingestion successful. File saved as {new_file_path}")

        except Exception as e:
            self.logger.error(f"❌ Error during data ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger = setup_logging()

    # Dynamically set the base directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Define dataset name, download directory, output directory, and desired filename
    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    download_dir = os.path.join(BASE_DIR, "data/api_downloaded")  # ✅ Added back
    output_dir = os.path.join(BASE_DIR, "data/raw")
    output_filename = "bank_churn_api_raw.csv"

    # ✅ Fixed Argument Order
    ingestion = DataIngestion(dataset_name, download_dir, output_dir, output_filename, logger)
    ingestion.ingest_data()
