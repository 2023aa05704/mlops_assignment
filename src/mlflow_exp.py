import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn
import boto3

# Load dataset
data = pd.read_csv('data/housing.csv')

# Split columns for X and y
X = data.drop(
    ['price', 'mainroad', 'guestroom', 'basement', 'hotwaterheating',
     'airconditioning', 'prefarea', 'furnishingstatus'], axis=1
)
y = data.price.copy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models with hyperparameters
models = [
    ('RandomForest1', RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
    ('RandomForest2', RandomForestRegressor, {'n_estimators': 200, 'random_state': 42}),
    ('RandomForest3', RandomForestRegressor, {'n_estimators': 300, 'random_state': 42}),
    ('GradientBoosting1', GradientBoostingRegressor, {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}),
    ('GradientBoosting2', GradientBoostingRegressor, {'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 42}),
    ('GradientBoosting3', GradientBoostingRegressor, {'n_estimators': 300, 'learning_rate': 0.1, 'random_state': 42}),
    ('XGBoost1', XGBRegressor, {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}),
    ('XGBoost2', XGBRegressor, {'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 42}),
    ('XGBoost3', XGBRegressor, {'n_estimators': 300, 'learning_rate': 0.1, 'random_state': 42}),
]

# Loop through each model configuration and track with MLflow
for model_name, model_class, params in models:
    with mlflow.start_run():
        # Log scikit-learn version
        mlflow.log_param("sklearn_version", sklearn.__version__)

        # Initialize and train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Make predictions and evaluate metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = mse ** 0.5

        # Log model parameters and metrics
        mlflow.log_param("model", model_name)
        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)

        # Log the model as an artifact
        model_filename = f"{model_name}_model.joblib"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)

        print(f"Experiment with {model_name}:")
        print(f"  Hyperparameters: {params}")
        print(f"  MAE: {mae}")
        print(f"  MSE: {mse}")
        print(f"  R^2: {r2}")
        print(f"  RMSE: {rmse}")
        print('-' * 40)
