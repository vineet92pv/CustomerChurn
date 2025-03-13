import pandas as pd
import sqlite3
import os   

def retieve_transformed_data():
        """Load data from SQLite database."""
        try:
            BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            db_path = os.path.join(BASE_DIR, "data/database/churn_data.db")
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM bank_churn_transformed", conn)
            conn.close()
            print(df)
        except Exception as e:
            print(f"❌ Error loading data from database: {str(e)}", exc_info=True)
            return None
        
if __name__ == "__main__":
    retieve_transformed_data()