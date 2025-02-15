import pickle
import os
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify

# Load trained model and scaler
MODEL_DIR = "data/models"
logistic_model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")

if not os.path.exists(logistic_model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found. Train the model first.")

with open(logistic_model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(feature_names_path, "r") as f:
    expected_features = json.load(f)    

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure input features match training features
        df = df.reindex(columns=expected_features, fill_value=0)
        df_scaled = scaler.transform(df)

        # Predict churn probability
        churn_probability = model.predict_proba(df_scaled)[:, 1]

        return jsonify({"churn_probability": float(churn_probability[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
