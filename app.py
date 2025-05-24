from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from prophet import Prophet
import pandas as pd
import traceback
import os
import json

app = Flask(__name__)

firebase_config_json = os.environ.get("FIREBASE_KEY_JSON")

try:
    if firebase_config_json:
        # Load credentials from environment variable for deployment
        cred = credentials.Certificate(json.loads(firebase_config_json))
    else:
        # Fallback to local file for development
        cred = credentials.Certificate("firebase-key.json")

    initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {e}")

@app.route('/predict', methods=['POST'])
def run_forecast():
    try:
        # Attempt to get UID from the request JSON body
        uid = request.json.get('uid')
        print(f"Received UID in request: {uid}") # DEBUG PRINT

        if not uid:
            print("Error: UID not provided in request body.") # DEBUG PRINT
            return jsonify({"error": "UID not provided in request body"}), 400

        user_doc_ref = db.collection("users").document(uid)
        user_doc = user_doc_ref.get()

        print(f"Checking Firestore for user '{uid}'. Document exists: {user_doc.exists}") # DEBUG PRINT

        if not user_doc.exists:
            print(f"Error: User '{uid}' not found in Firestore.") # DEBUG PRINT
            return jsonify({"error": f"User '{uid}' not found in Firestore"}), 404

        user_data = user_doc.to_dict()
        co2_data = user_data.get("co2_data", [])

        if not co2_data or len(co2_data) < 3:
            print(f"Warning: Not enough CO₂ data for user '{uid}'. Length: {len(co2_data)}") # DEBUG PRINT
            return jsonify({"error": "Not enough CO₂ data (minimum 3 data points required)"}), 403

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

        water_data = user_data.get("water_data", [])
        water_forecast_results = []

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
        else:
            print(f"Warning: Not enough water data for user '{uid}' or no water data. Length: {len(water_data)}") # DEBUG PRINT

        combined_forecast = []
        for co2_item in co2_forecast_results:
            combined_item = co2_item.copy()
            # Try to find a matching water prediction for the current date
            for water_item in water_forecast_results:
                if water_item['date'] == co2_item['date']:
                    combined_item['water_pred'] = water_item['water_pred']
                    break # Found a match, move to the next co2_item
            else: # This else block executes if the inner loop completes without a 'break'
                combined_item['water_pred'] = 0.0 # Default if no water data or no matching date
            combined_forecast.append(combined_item)

        user_doc_ref.update({"user_forecast": combined_forecast})
        print(f"Forecast successfully generated and updated for user '{uid}'.") # DEBUG PRINT

        return jsonify({
            "message": f"Forecast updated for {uid}",
            "forecast": combined_forecast
        }), 200

    except Exception as e:
        traceback.print_exc() # This prints the full traceback to your console/logs
        print(f"An unexpected error occurred: {e}") # DEBUG PRINT
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    # The host should be 0.0.0.0 to make the Flask app accessible from outside the container/localhost
    # The port should be taken from the environment variable 'PORT' (common in cloud environments)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
