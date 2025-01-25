import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TestHousingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup method for loading data once for all test cases
        cls.data = pd.read_csv('data/housing.csv')
        cls.X = cls.data.drop(['price', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], axis=1)
        cls.y = cls.data['price'].copy()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_data_loading(self):
        """Test if the data loads and has the expected columns."""
        # Check if the dataset has the correct number of rows and columns
        self.assertGreater(len(self.data), 0, "Dataset is empty")
        self.assertIn('price', self.data.columns, "'price' column is missing from the dataset")

    def test_feature_columns(self):
        """Test if the feature columns are correctly dropped from X."""
        expected_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']  # Replace with actual feature names
        dropped_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

        # Ensure that dropped columns are no longer in X
        for col in dropped_columns:
            self.assertNotIn(col, self.X.columns, f"Column '{col}' should be dropped")

        # Ensure that remaining columns are correct (replace with your actual columns)
        for col in expected_columns:
            self.assertIn(col, self.X.columns, f"Column '{col}' is missing from X")

    def test_model_training(self):
        """Test if the model trains without errors."""
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(self.X_train, self.y_train)

        # Check if the model has been trained (it should have the `feature_importances_` attribute)
        self.assertTrue(hasattr(model, 'feature_importances_'), "Model was not trained successfully")

    def test_model_prediction(self):
        """Test if predictions can be made."""
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        # Check if predictions are made and their length matches the number of test samples
        self.assertEqual(len(y_pred), len(self.y_test), "Prediction length does not match test set length")

    def test_model_evaluation(self):
        """Test if the model evaluation metrics are computed correctly."""
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Test if evaluation metrics are within expected ranges
        self.assertGreater(mae, 0, "Mean Absolute Error should be positive")
        self.assertGreater(mse, 0, "Mean Squared Error should be positive")
        self.assertGreater(r2, 0, "R^2 Score should be positive")

if __name__ == "__main__":
    unittest.main()
