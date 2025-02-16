import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define paths for WSL
WSL_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
#WSL_PROJECT_DIR = "/home/vineeth/airflow/CustomerChurn"
PYTHON_ENV = f"{WSL_PROJECT_DIR}/airflow_env/bin/python"
LOG_DIR = f"{WSL_PROJECT_DIR}/logs"

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=f"{LOG_DIR}/airflow_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define DAG arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 2, 13),
    "retries": 1
}

dag = DAG(
    "churn_pipeline",
    default_args=default_args,
    description="End-to-End Churn Prediction Pipeline",
    schedule="@daily",
    catchup=False
)

# Function to execute and log Python scripts
def run_script(script_path: str, task_name: str):
    logger.info(f"Starting {task_name}...")
    result = os.system(f"cd {WSL_PROJECT_DIR} && {PYTHON_ENV} {script_path}")
    if result == 0:
        logger.info(f"{task_name} completed successfully!")
    else:
        logger.error(f"{task_name} failed with exit code {result}")

# Define PythonOperator tasks with logging
ingestion_task = PythonOperator(
    task_id="ingest_data",
    python_callable=run_script,
    op_kwargs={"script_path": "src/ingestion/ingest.py", "task_name": "Data Ingestion"},
    dag=dag
)

validation_task = PythonOperator(
    task_id="validate_data",
    python_callable=run_script,
    op_kwargs={"script_path": "src/validation/data_validation.py", "task_name": "Data Validation"},
    dag=dag
)

preparation_task = PythonOperator(
    task_id="prepare_data",
    python_callable=run_script,
    op_kwargs={"script_path": "src/preparation/data_preparation.py", "task_name": "Data Preparation"},
    dag=dag
)

transformation_task = PythonOperator(
    task_id="transform_data",
    python_callable=run_script,
    op_kwargs={"script_path": "src/feature_engineering/feature_engineering.py", "task_name": "Data Transformation"},
    dag=dag
)

model_training_task = PythonOperator(
    task_id="train_model",
    python_callable=run_script,
    op_kwargs={"script_path": "src/model_training/model_training.py", "task_name": "Model Training"},
    dag=dag
)

# Set task dependencies
ingestion_task >> validation_task >> preparation_task >> transformation_task >> model_training_task