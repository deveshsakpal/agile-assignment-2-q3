import unittest
import numpy as np
import mlflow

# Mock the model so tests do not require a running MLflow server.
class FakeModel:
    def predict(self, features):
        return np.array([0])

mlflow.set_tracking_uri = lambda *args, **kwargs: None
mlflow.pyfunc.load_model = lambda *args, **kwargs: FakeModel()

from app import app


class IrisAPITests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction_endpoint"], "/predict"
        )

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["class_name"], "setosa")
        self.assertEqual(response.get_json()["prediction"], 0)

    def test_wrong_number_of_features(self):
        response = self.client.post(
            "/predict", json={"features": [5.1, 3.5]}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_feature(self):
        response = self.client.post(
            "/predict",
            json={"features": [5.1, "bad", 1.4, 0.2]}
        )
        self.assertEqual(response.status_code, 400)

    def test_metrics(self):
        self.client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]}
        )
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"iris_predictions_total", response.data)


if __name__ == "__main__":
    unittest.main()
