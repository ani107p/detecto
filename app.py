import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load the trained model, scaler, and one-hot encoded column names
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
one_hot_columns = pickle.load(open("one_hot_columns.pkl", "rb"))

# Define home route
@app.route("/")
def home():
    return render_template("index.html")

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form values
        features = [float(x) for x in request.form.values()]
        
        # Load the one-hot encoded column names
        one_hot_columns = pickle.load(open("one_hot_columns.pkl", "rb"))
        
        # Ensure all features used during training are present
        all_features = ['months_as_customer', 'policy_deductable', 'umbrella_limit', 'capital-gains', 'capital-loss',
                        'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
                        'injury_claim', 'property_claim', 'vehicle_claim']
        
        # Create a DataFrame with all features
        input_df = pd.DataFrame([features], columns=all_features)
        
        # Fill missing categorical values with zeros
        for col in one_hot_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale numerical features
        scaled_features = scaler.transform(input_df.drop(columns=one_hot_columns))
        
        # Combine scaled numerical and encoded categorical features
        processed_features = np.hstack((scaled_features, input_df[one_hot_columns].values))

        # Predict using the loaded model
        prediction = model.predict(processed_features)

        # Render the result in the template
        return render_template("index.html", prediction_text="The prediction is: {}".format(prediction[0]))

    except Exception as e:
        # Handle exceptions
        return render_template("index.html", prediction_text="Error: {}".format(e))


if __name__ == "__main__":
    app.run(debug=True)
