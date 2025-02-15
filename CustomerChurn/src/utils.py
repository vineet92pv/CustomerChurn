import logging
import os

def setup_logging():
    logger = logging.getLogger("data_ingestion")
    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)
    log_filename = "logs/ingestion.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger