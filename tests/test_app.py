import unittest
from unittest.mock import patch
import numpy as np

class FakeModel:
    def predict(self, features):
        return np.array([0])

# Mock MLflow before app.py is imported.
with patch("mlflow.set_tracking_uri"), \
     patch("mlflow.pyfunc.load_model", return_value=FakeModel()):
    from app import app

class IrisAPITests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction_endpoint"], "/predict")

    def test_valid_prediction(self):
        r = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["class_name"], "setosa")

    def test_wrong_feature_count(self):
        r = self.client.post("/predict", json={"features": [5.1, 3.5]})
        self.assertEqual(r.status_code, 400)

    def test_non_numeric_feature(self):
        r = self.client.post("/predict", json={"features": [5.1, "bad", 1.4, 0.2]})
        self.assertEqual(r.status_code, 400)

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"iris_predictions_total", r.data)

if __name__ == "__main__":
    unittest.main()
