import logging
import os
import sys

def setup_logging(script_name=None):
    """Sets up logging with filenames based on the calling script."""
    
    # Get the calling script's name if not provided
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Define log file path with script name
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{script_name}.log"

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
