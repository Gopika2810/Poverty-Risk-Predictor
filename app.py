# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load model, scaler, and label encoder
model = joblib.load("best_poverty_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        X = np.array([list(data.values())])
        X_scaled = scaler.transform(X)
        pred_class = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        pred_probs = model.predict_proba(X_scaled)[0].tolist()
        return jsonify({"predicted_class": pred_label, "probabilities": pred_probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
