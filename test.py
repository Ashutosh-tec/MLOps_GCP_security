import unittest
import joblib
import numpy as np
import pandas as pd
import os

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        # Load model using joblib
        model_path = 'decision_tree_model.joblib'
        self.assertTrue(os.path.exists(model_path), f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)

        # Define test input
        self.sample_data = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]],
            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        )

        self.expected_class = 'setosa'  # Adjust if your model uses encoded labels like 0, 1, 2

    def test_prediction_accuracy(self):
        prediction = self.model.predict(self.sample_data)[0]
        self.assertEqual(
            prediction,
            self.expected_class,
            f"Expected class {self.expected_class}, got {prediction}"
        )

    def test_data_validation(self):
        df = self.sample_data

        self.assertEqual(df.shape[1], 4, "Input must have 4 features")
        self.assertFalse(df.isnull().values.any(), "Input contains null values")
        self.assertTrue(
            all(np.issubdtype(dtype, np.number) for dtype in df.dtypes),
            "All feature columns must be numeric"
        )

if __name__ == '__main__':
    unittest.main()

