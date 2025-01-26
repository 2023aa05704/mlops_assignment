from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

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

    # Fetch values from the "aws" section
    aws_access_key_id = config.get("aws", "AWS_ACCESS_KEY_ID")
    aws_secret_access_key = config.get("aws", "AWS_SECRET_ACCESS_KEY")

os.environ['MLFLOW_ARTIFACT_URI'] = 's3://2023aa05704-mlops-assignment/mlflow'  # S3 for artifacts
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
os.environ["AWS_DEFAULT_REGION"] = 'ap-south-1'

# Initialize Flask app
app = Flask(__name__)

# S3 URI of the model artifact
MODEL_URI = "s3://2023aa05704-mlops-assignment/mlflow/0/2ddc842baa034d49be2b35e266a08abe/artifacts/best_random_forest_model"

# Load the model from the S3 URI
model = mlflow.sklearn.load_model(MODEL_URI)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions.
    Accepts JSON input with the feature values and returns predictions.
    """
    try:
        # Parse JSON input
        input_data = request.get_json()
        data = pd.DataFrame(input_data)

        # Make predictions
        predictions = model.predict(data)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222)
