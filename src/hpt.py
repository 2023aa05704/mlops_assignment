import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn
import boto3
import os
import configparser

# Check if AWS credentials are set via environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# If the environment variables aren't set, fall back to reading from a config file
if not aws_access_key_id or not aws_secret_access_key:
    CONFIG_FILE_PATH = ".aws/aws_config.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    aws_access_key_id = config.get("aws", "AWS_ACCESS_KEY_ID")
    aws_secret_access_key = config.get("aws", "AWS_SECRET_ACCESS_KEY")

os.environ['MLFLOW_ARTIFACT_URI'] = 's3://2023aa05704-mlops-assignment/mlflow'  # S3 for artifacts
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
os.environ["AWS_DEFAULT_REGION"] = 'ap-south-1'

# Load dataset
if not os.path.exists('data/housing_pre.csv'):
    raise FileNotFoundError("The dataset file 'data/housing_pre.csv' was not found.")
data = pd.read_csv('data/housing_pre.csv')

# Split columns for X and y
X = data.drop(
    ['price'], axis=1
)
y = data.price.copy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1
)

# Track with MLflow
with mlflow.start_run():
    # Log scikit-learn version
    mlflow.log_param("sklearn_version", sklearn.__version__)

    # Perform GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best estimator and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions and evaluate metrics
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mse ** 0.5

    # Log hyperparameters, metrics, and model
    mlflow.log_param("model", "RandomForestRegressor")
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSE", rmse)

    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")

    # Print results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Metrics:")
    print(f"  MAE: {mae}")
    print(f"  MSE: {mse}")
    print(f"  R^2: {r2}")
    print(f"  RMSE: {rmse}")
    print('-' * 40)
