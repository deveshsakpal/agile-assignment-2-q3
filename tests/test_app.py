import unittest
from unittest.mock import patch

import numpy as np
import mlflow
import mlflow.pyfunc


class FakeModel:
    def predict(self, features):
        return np.array([0])


# Patch MLflow BEFORE importing the application.
with patch.object(mlflow, "set_tracking_uri"), \
     patch.object(mlflow.pyfunc, "load_model", return_value=FakeModel()):
    from app import app


class IrisAPITests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction_endpoint"], "/predict")
        self.assertEqual(data["metrics_endpoint"], "/metrics")

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], 0)
        self.assertEqual(response.get_json()["class_name"], "setosa")

    def test_wrong_number_of_features(self):
        response = self.client.post(
            "/predict",
            json={"features": [5.1, 3.5]}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_feature(self):
        response = self.client.post(
            "/predict",
            json={"features": [5.1, "bad", 1.4, 0.2]}
        )
        self.assertEqual(response.status_code, 400)

    def test_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"iris_predictions_total", response.data)


if __name__ == "__main__":
    unittest.main()
