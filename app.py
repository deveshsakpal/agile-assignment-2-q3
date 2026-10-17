import os
import numpy as np
import mlflow
import mlflow.pyfunc

from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

app = Flask(__name__)

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
)

MODEL_URI = os.getenv(
    "MODEL_URI", "models:/IrisRandomForest/1"
)

def load_prediction_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_prediction_model()

PREDICTIONS = Counter(
    "iris_predictions_total",
    "Total number of successful Iris predictions"
)

REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method"]
)

PREDICTION_TIME = Histogram(
    "prediction_duration_seconds",
    "Time taken to make Iris predictions"
)

@app.route("/")
def home():
    return jsonify({
        "message": "Iris ML Model API is running",
        "model": MODEL_URI,
        "prediction_endpoint": "/predict",
        "metrics_endpoint": "/metrics"
    })

@app.route("/predict", methods=["POST"])
def predict():
    REQUESTS.labels(
        endpoint="/predict", method="POST"
    ).inc()

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing features"
        }), 400

    features = data.get("features")

    if (
        not isinstance(features, list)
        or len(features) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in features
        )
    ):
        return jsonify({
            "error": "features must contain exactly four numbers"
        }), 400

    with PREDICTION_TIME.time():
        prediction = model.predict(np.array([features], dtype=np.float64))

    PREDICTIONS.inc()

    return jsonify({
        "prediction": int(prediction[0]),
        "class_name": [
            "setosa", "versicolor", "virginica"
        ][int(prediction[0])]
    })

@app.route("/metrics")
def metrics():
    REQUESTS.labels(
        endpoint="/metrics", method="GET"
    ).inc()

    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
