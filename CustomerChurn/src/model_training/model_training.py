import pandas as pd
import sqlite3
import os
import pickle
import logging
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class ModelTraining:
    def __init__(self, db_path: str, model_dir: str, logger):
        self.db_path = db_path
        self.model_dir = model_dir
        self.logger = logger
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM transformed_data", conn)
            conn.close()
            self.logger.info("Transformed data loaded successfully for model training.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from database: {str(e)}", exc_info=True)
            return None

    def preprocess_data(self, df, target_column="Attrition_Flag"):
        try:
            # Convert target column to binary (e.g., 'Existing Customer' → 0, 'Attrited Customer' → 1)
            df[target_column] = df[target_column].map({"Existing Customer": 0, "Attrited Customer": 1})

            # Select features and target
            X = df.drop(columns=[target_column, "CLIENTNUM"])  # Remove ID column
            y = df[target_column]

            # Standardize numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            feature_names_path = os.path.join(self.model_dir, "feature_names.json")
            with open(feature_names_path, "w") as f:
                json.dump(list(X.columns), f)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.logger.info("Data preprocessed and split into train and test sets.")
            return X_train, X_test, y_train, y_test, scaler
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}", exc_info=True)
            return None, None, None, None, None

    def train_model(self, X_train, y_train, model_type="logistic"):
        try:
            if model_type == "logistic":
                model = LogisticRegression()
            elif model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.logger.error(f"Invalid model type: {model_type}")
                return None

            model.fit(X_train, y_train)
            self.logger.info(f"{model_type} model trained successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}", exc_info=True)
            return None

    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            self.logger.info(f"Model Evaluation: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
            return None

    def save_model(self, model, scaler, model_type):
        try:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            self.logger.info(f"{model_type} model saved at {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}", exc_info=True)

    def run_training_pipeline(self):
        df = self.load_data()
        if df is not None:
            X_train, X_test, y_train, y_test, scaler = self.preprocess_data(df)

            if X_train is not None:
                logistic_model = self.train_model(X_train, y_train, "logistic")
                rf_model = self.train_model(X_train, y_train, "random_forest")

                if logistic_model:
                    self.evaluate_model(logistic_model, X_test, y_test)
                    self.save_model(logistic_model, scaler, "logistic")

                if rf_model:
                    self.evaluate_model(rf_model, X_test, y_test)
                    self.save_model(rf_model, scaler, "random_forest")

def run_model_training():
    logger = setup_logging()
    db_path = "data/database/churn_data.db"
    model_dir = "data/models"
    trainer = ModelTraining(db_path, model_dir, logger)
    trainer.run_training_pipeline()

if __name__ == "__main__":
    run_model_training()
