import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import setup_logging

class DataPreparation:
    def __init__(self, input_files: list, output_dir: str, logger):
        self.input_files = input_files  # Accept multiple files
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Standardized column mapping
        self.column_mapping = {
            "CustomerId": "Customer_ID",
            "Customer_": "Customer_ID",
            "Surname": "Last_Name",
            "CreditScore": "Credit_Score",
            "Credit_Scc": "Credit_Score",
            "credit_sco": "Credit_Score",
            "Geography": "Country",
            "Gender": "Gender",
            "Age": "Age",
            "age": "Age",
            "Tenure": "Tenure",
            "tenure": "Tenure",
            "Balance": "Balance",
            "balance": "Balance",
            "NumOfProducts": "Num_Products",
            "products_n": "Num_Products",
            "products_number": "Num_Products",
            "HasCrCard": "Has_Credit_Card",
            "credit_car": "Has_Credit_Card",
            "credit_card": "Has_Credit_Card",
            "IsActiveMember": "Is_Active",
            "active_me": "Is_Active",
            "active_member": "Is_Active",
            "EstimatedSalary": "Salary",
            "estimated_cl": "Salary",
            "estimated_salary": "Salary",
            "Exited": "Churn",
            "churn": "Churn"
        }

    def load_data(self, file_path):
        """Load a single file"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"✅ Data loaded from {file_path}")
            self.logger.info(f"Columns before renaming: {df.columns.tolist()}")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error loading {file_path}: {str(e)}", exc_info=True)
            return None

    def standardize_columns(self, df):
        """Rename columns to a common format and remove unwanted columns."""
        df.rename(columns={k: v for k, v in self.column_mapping.items() if k in df.columns}, inplace=True)

        # Ensure 'Customer_ID' is correctly standardized
        if "customer_id" in df.columns:
            df.rename(columns={"customer_id": "Customer_ID"}, inplace=True)

        # Remove unwanted columns
        for col in ["id", "Last_Name"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                self.logger.info(f"🔹 Removed column: {col}")

        self.logger.info(f"✅ Standardized columns: {df.columns.tolist()}")
        return df

    def remove_duplicate_columns(self, df):
        """Ensure only one correct version of each column remains."""
        seen = set()
        drop_cols = []
        
        for col in df.columns:
            normalized_col = col.lower().strip()
            if normalized_col in seen:
                drop_cols.append(col)
            else:
                seen.add(normalized_col)
        
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            self.logger.info(f"🛑 Dropped duplicate columns: {drop_cols}")
        
        return df

    def detect_column_types(self, df):
        """Automatically detect numeric and categorical columns."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        # Exclude `Churn` from encoding or transformations
        if "Churn" in numeric_cols:
            numeric_cols.remove("Churn")
        if "Churn" in categorical_cols:
            categorical_cols.remove("Churn")

        self.logger.info(f"🔹 Numeric columns: {numeric_cols}")
        self.logger.info(f"🔹 Categorical columns: {categorical_cols}")
        return numeric_cols, categorical_cols

    @staticmethod
    def handle_missing_values(df):
        """Handle missing values by forward filling."""
        logging.info("🔹 Handling missing values.")
        df.ffill(inplace=True)
        return df

    def encode_categorical(self, df, categorical_cols):
        """Convert categorical columns to numerical values using one-hot encoding."""
        self.logger.info("🔹 Encoding categorical variables.")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df

    def standardize_numerical(self, df, numeric_cols):
        """Standardize numerical columns using Z-score normalization."""
        self.logger.info("🔹 Standardizing numerical features.")
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        return df

    def remove_duplicates(self, df):
        """Remove duplicate customers based on Customer_ID."""
        logging.info("🔹 Removing duplicate customers based on Customer_ID.")
        if "Customer_ID" in df.columns:
            df = df.drop_duplicates(subset=["Customer_ID"], keep="first")
        else:
            logging.warning(f"⚠️ 'Customer_ID' column not found in dataset. Available columns: {df.columns.tolist()}")
        return df

    def save_cleaned_data(self, df, filename):
        """Save the cleaned dataset."""
        output_file = os.path.join(self.output_dir, filename)
        df.to_csv(output_file, index=False)
        self.logger.info(f"✅ Cleaned data saved at {output_file}")

    def run_preparation(self):
        """Run preprocessing on all datasets and merge if needed."""
        processed_dataframes = []
        
        for file_path in self.input_files:
            df = self.load_data(file_path)
            if df is not None:
                df = self.standardize_columns(df)
                df = self.remove_duplicate_columns(df)
                df = self.handle_missing_values(df)
                numeric_cols, categorical_cols = self.detect_column_types(df)
                df = self.encode_categorical(df, categorical_cols)
                df = self.standardize_numerical(df, numeric_cols)
                df = self.remove_duplicates(df)
                processed_dataframes.append(df)

                # Save each processed file
                filename = f"processed_{os.path.basename(file_path)}"
                self.save_cleaned_data(df, filename)

        # Merge all processed files and save final dataset
        if len(processed_dataframes) > 1:
            self.logger.info("🔄 Merging multiple datasets")
            final_df = pd.concat(processed_dataframes, ignore_index=True)
        else:
            final_df = processed_dataframes[0]

        # Remove duplicate columns after merging
        final_df = self.remove_duplicate_columns(final_df)
        final_df = self.remove_duplicates(final_df)

        # Save final merged file
        self.save_cleaned_data(final_df, "bank_churn_processed.csv")

def run_data_preparation():
    """Run the full pipeline"""
    logger = setup_logging()
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    raw_data_dir = os.path.join(BASE_DIR, "data/raw")
    input_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith(".csv")]

    if not input_files:
        logger.error("❌ No raw data files found.")
        return

    output_dir = os.path.join(BASE_DIR, "data/processed")
    preparation = DataPreparation(input_files, output_dir, logger)
    preparation.run_preparation()

if __name__ == "__main__":
    run_data_preparation()
