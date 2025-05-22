from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

# Firebase setup
cred = credentials.Certificate("firebase-key.json")
initialize_app(cred)
db = firestore.client()

@app.route('/predict', methods=['POST'])
def run_forecast():
    try:
        uid = request.json.get('uid')
        if not uid:
            return jsonify({"error": "UID not provided"}), 400

        # Fetch user data
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": f"User '{uid}' not found"}), 404

        co2_data = user_doc.to_dict().get("co2_data", [])
        if not co2_data:
            return jsonify({"error": "No CO₂ data found"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})
        df['ds'] = pd.to_datetime(df['ds'])

        # Forecast with Prophet
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=3)
        forecast = model.predict(future)[['ds', 'yhat']].tail(3)

        # Format result
        result = [{
            "date": str(row['ds'].date()),
            "co2_pred": round(row['yhat'], 2)
        } for _, row in forecast.iterrows()]

        # Save back to Firestore
        db.collection("users").document(uid).update({"user_forecast": result})

        return jsonify({"message": f"Forecast updated for {uid}", "forecast": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
