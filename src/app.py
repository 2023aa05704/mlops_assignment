from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from logging.handlers import RotatingFileHandler
import os

# Initialize Flask app
app = Flask(__name__)

# Configure logging
log_file = "logs/app.log"
log_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=3)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
log_handler.setFormatter(formatter)
log_handler.setLevel(logging.DEBUG)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.DEBUG)

# Load the saved model
try:
    model_path = 'model/rf-default.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as model_file:
        model = joblib.load(model_file)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    raise

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received prediction request.")

        # Get the input data from the POST request
        data = request.get_json()
        if not data:
            raise ValueError("No input data provided.")

        app.logger.debug(f"Input data: {data}")

        # Convert the input JSON to a pandas DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the model
        prediction = model.predict(input_df)
        app.logger.info("Prediction successful.")

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# Start the Flask application
if __name__ == '__main__':
    app.logger.info("Starting Flask application.")
    app.run(debug=True, host='0.0.0.0', port=5000)
