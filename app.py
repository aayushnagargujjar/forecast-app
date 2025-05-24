from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

# Firebase setup
cred = credentials.Certificate("firebase-key.json")
initialize_app(cred)
db = firestore.client()

from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from prophet import Prophet
import pandas as pd
import traceback

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")  # Make sure this file is in the same folder
initialize_app(cred)
db = firestore.client()

@app.route('/predict', methods=['POST'])
def run_forecast():
    try:
        # Get UID from request
        uid = request.json.get('uid')
        if not uid:
            return jsonify({"error": "UID not provided"}), 400

        # Fetch user document from Firestore
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": f"User '{uid}' not found"}), 404

        # Get CO2 data
        co2_data = user_doc.to_dict().get("co2_data", [])
        if not co2_data or len(co2_data) < 3:
            return jsonify({"error": "Not enough CO₂ data (minimum 3 data points required)"}), 400

        # Prepare DataFrame for Prophet
        df = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})
        df['ds'] = pd.to_datetime(df['ds'])

        # Forecast using Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=3)
        forecast = model.predict(future)[['ds', 'yhat']].tail(3)

        # Format forecast output
        result = [{
            "date": str(row['ds'].date()),
            "co2_pred": round(row['yhat'], 2)
        } for _, row in forecast.iterrows()]

        # Update Firestore document
        db.collection("users").document(uid).update({"user_forecast": result})

        return jsonify({
            "message": f"Forecast updated for {uid}",
            "forecast": result
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
