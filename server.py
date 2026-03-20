from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return "HousePricePro Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get values from frontend
        area = float(data["area"])
        bedrooms = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
        year = int(data["year"])
        location = data["location"]

        # Encode location
        loc_downtown = 1 if location == "Downtown" else 0
        loc_suburban = 1 if location == "Suburban" else 0
        loc_rural = 1 if location == "Rural" else 0
        loc_urban = 1 if location == "Urban" else 0

        # Create input array (MUST MATCH TRAINING)
        sample = np.array([[area, bedrooms, bathrooms, year,
                            loc_downtown, loc_suburban, loc_rural, loc_urban]])

        # Scale input
        sample = scaler.transform(sample)

        # Predict
        prediction = model.predict(sample)[0]

        return jsonify({
            "price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run server
if __name__ == "__main__":
    app.run(debug=True)