from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from prophet import Prophet
import pandas as pd
import traceback
import os # Import os for environment variables (better for security)

app = Flask(__name__)

# --- Firebase Initialization ---
# It's often safer to load the Firebase key path from an environment variable
# or a configuration file rather than hardcoding.
# For local development, you'd keep 'firebase-key.json'.
# For deployment (e.g., Render.com), you'd set this as an environment variable.
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase-key.json")

try:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    # Consider more robust error handling for production (e.g., exit if critical)

@app.route('/predict', methods=['POST'])
def run_forecast():
    try:
        # Get UID from request (Flask handles JSON parsing automatically here)
        uid = request.json.get('uid')
        if not uid:
            return jsonify({"error": "UID not provided in request body"}), 400

        # Fetch user document from Firestore
        user_doc_ref = db.collection("users").document(uid)
        user_doc = user_doc_ref.get()
        if not user_doc.exists:
            return jsonify({"error": f"User '{uid}' not found in Firestore"}), 404

        user_data = user_doc.to_dict()

        # --- Handle CO2 Data Forecast ---
        co2_data = user_data.get("co2_data", [])
        if not co2_data or len(co2_data) < 3:
            return jsonify({"error": "Not enough CO₂ data (minimum 3 data points required)"}), 400

        df_co2 = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})
        df_co2['ds'] = pd.to_datetime(df_co2['ds'])

        model_co2 = Prophet()
        model_co2.fit(df_co2)
        future_co2 = model_co2.make_future_dataframe(periods=3)
        forecast_co2 = model_co2.predict(future_co2)[['ds', 'yhat']].tail(3)

        co2_forecast_results = [{
            "date": str(row['ds'].date()),
            "co2_pred": round(row['yhat'], 2)
        } for _, row in forecast_co2.iterrows()]

        # --- OPTIONAL: Handle Water Data Forecast (if needed) ---
        # Your current Flask code *does not* forecast water data.
        # If your Android app expects 'water_pred' in 'user_forecast',
        # you *must* implement water forecasting here.
        water_data = user_data.get("water_data", [])
        water_forecast_results = [] # Initialize empty, will be populated if data exists

        if water_data and len(water_data) >= 3:
            df_water = pd.DataFrame(water_data).rename(columns={"date": "ds", "value": "y"})
            df_water['ds'] = pd.to_datetime(df_water['ds'])

            model_water = Prophet()
            model_water.fit(df_water)
            future_water = model_water.make_future_dataframe(periods=3)
            forecast_water = model_water.predict(future_water)[['ds', 'yhat']].tail(3)

            water_forecast_results = [{
                "date": str(row['ds'].date()),
                "water_pred": round(row['yhat'], 2)
            } for _, row in forecast_water.iterrows()]

        # --- Combine Forecasts and Update Firestore ---
        # The structure here needs to align with what your Android app expects.
        # If your Android app expects a single list for 'user_forecast'
        # containing both CO2 and water predictions, you need to merge them.
        # A simple approach for separate models is to combine based on date.
        # This example assumes you want *both* co2_pred and water_pred in each item.
        # If water_forecast_results is empty, water_pred will be omitted or 0.
        combined_forecast = []
        for i in range(len(co2_forecast_results)):
            item = co2_forecast_results[i].copy()
            if i < len(water_forecast_results) and water_forecast_results[i]['date'] == item['date']:
                item['water_pred'] = water_forecast_results[i]['water_pred']
            else:
                item['water_pred'] = 0.0 # Or None, or omit if water data not available
            combined_forecast.append(item)


        # Update Firestore document with the combined forecast
        # We use a dictionary to update, aligning with Firestore's merge behavior.
        # Note: If 'user_forecast' already exists, it will be overwritten.
        user_doc_ref.update({"user_forecast": combined_forecast})

        return jsonify({
            "message": f"Forecast updated for {uid}",
            "forecast": combined_forecast # Send the combined forecast back to the client
        }), 200

    except Exception as e:
        # Log the full traceback for better debugging on the server side
        traceback.print_exc()
        # Return a generic 500 error for security, but log details
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    # In production, use a production-ready WSGI server like Gunicorn or uWSGI.
    # For Render.com, they handle this, but for local testing, this is fine.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000))) # Use PORT env var for Render.com
