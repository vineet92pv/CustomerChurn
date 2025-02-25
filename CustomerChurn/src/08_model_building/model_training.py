import pandas as pd
import sqlite3
import os
import pickle
import logging
import sys
import json
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class ModelTraining:
    def __init__(self, db_path: str, model_dir: str, logger):
        """Initialize model training with database path and logging."""
        self.db_path = db_path
        self.model_dir = model_dir
        self.logger = logger
        os.makedirs(self.model_dir, exist_ok=True)

        #Set MLflow tracking URI to store models inside `data/models/`
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Customer Churn Prediction")

    def load_data(self):
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM transformed_data", conn)
            conn.close()
            self.logger.info("✅ Transformed data loaded successfully for model training.")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error loading data from database: {str(e)}", exc_info=True)
            return None

    def preprocess_data(self, df, target_column="Attrition_Flag"):
        """Preprocess data: feature selection, scaling, and train-test split."""
        try:
            df[target_column] = df[target_column].map({"Existing Customer": 0, "Attrited Customer": 1})

            # Select features and target
            X = df.drop(columns=[target_column, "CLIENTNUM"])
            y = df[target_column]

            # Standardize numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Save feature names for future inference
            feature_names_path = os.path.join(self.model_dir, "feature_names.json")
            with open(feature_names_path, "w") as f:
                json.dump(list(X.columns), f)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            self.logger.info("✅ Data preprocessed and split into train and test sets.")
            return X_train, X_test, y_train, y_test, scaler
        except Exception as e:
            self.logger.error(f"❌ Error in data preprocessing: {str(e)}", exc_info=True)
            return None, None, None, None, None

    def train_and_log_model(self, X_train, y_train, X_test, y_test, model_type="logistic"):
        """Train model, evaluate it, and log everything to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_type}_model"):
                if model_type == "logistic":
                    # Hyperparameter tuning for Logistic Regression
                    log_reg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
                    log_reg_grid = GridSearchCV(LogisticRegression(max_iter=200, random_state=42),
                                                log_reg_params, cv=5, scoring='accuracy')
                    log_reg_grid.fit(X_train, y_train)  

                    # Get the best model
                    model = log_reg_grid.best_estimator_    

                    # Predict probabilities
                    y_proba = model.predict_proba(X_test)[:, 1] 

                    # Log best hyperparameters
                    mlflow.log_params(log_reg_grid.best_params_)    

                elif model_type == "random_forest":
                    # Hyperparameter tuning for Random Forest
                    rf_params = {
                        'n_estimators': [50, 100, 150],  # Number of trees
                        'max_depth': [3, 5, 7],  # Maximum depth of trees
                        'min_samples_split': [2, 5, 10]  # Minimum samples required to split a node
                    }
                    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
                    rf_grid.fit(X_train, y_train)   

                    # Get the best model
                    model = rf_grid.best_estimator_ 

                    # Predict probabilities
                    y_proba = model.predict_proba(X_test)[:, 1] 

                    # Log best hyperparameters
                    mlflow.log_params(rf_grid.best_params_) 

                else:
                    self.logger.error(f"❌ Invalid model type: {model_type}")
                    return None 

                # Predictions
                y_pred = model.predict(X_test)  

                # Compute metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred)
                }   

                # Log metrics & parameters
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("num_features", X_train.shape[1])  

                # Create input example & infer signature
                input_example = X_train[:5]  # First 5 rows as sample input
                signature = infer_signature(X_train, y_pred)    

                # Log model with signature & input example
                mlflow.sklearn.log_model(model, f"{model_type}_model",
                                         signature=signature,
                                         input_example=input_example)   

                self.logger.info(f"✅ {model_type} model trained and logged successfully to MLflow.")   

                return model
        except Exception as e:
            self.logger.error(f"❌ Error in model training: {str(e)}", exc_info=True)
            return None

    def save_model(self, model, scaler, model_type):
        """Save trained model and scaler as pickle files."""
        try:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            self.logger.info(f"✅ {model_type} model saved at {model_path}")
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {str(e)}", exc_info=True)

    def run_training_pipeline(self):
        """Run the entire training pipeline with MLflow logging."""
        df = self.load_data()
        if df is not None:
            X_train, X_test, y_train, y_test, scaler = self.preprocess_data(df)

            if X_train is not None:
                logistic_model = self.train_and_log_model(X_train, y_train, X_test, y_test, "logistic")
                rf_model = self.train_and_log_model(X_train, y_train, X_test, y_test, "random_forest")

                # Save models locally
                if logistic_model:
                    self.save_model(logistic_model, scaler, "logistic")
                if rf_model:
                    self.save_model(rf_model, scaler, "random_forest")

def run_model_training():
    """Initialize and run the training pipeline."""
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    db_path = os.path.join(BASE_DIR, "data/database/churn_data.db")
    model_dir = os.path.join(BASE_DIR, "data/models")
    trainer = ModelTraining(db_path, model_dir, logger)
    trainer.run_training_pipeline()

if __name__ == "__main__":
    run_model_training()
