from prophet import Prophet
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def forecast(df):
    # Ensure 'ds' column is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    model = Prophet()
    model.fit(df)
    # Forecast next 7 periods (days, if your data is daily)
    future = model.make_future_dataframe(periods=7)
    forecast_result = model.predict(future)
    return forecast_result[['ds', 'yhat']].tail(7) # Return only the forecasted dates and yhat

@app.route('/predict', methods=['POST']) # Endpoint for prediction
def run_forecast():
    try:
        # Get UID from the JSON body of the POST request
        req_data = request.get_json()
        if not req_data or 'uid' not in req_data:
            return jsonify({"error": "UID not provided in request body"}), 400

        uid = req_data['uid']
        print(f"Received forecast request for user: {uid}") # Debugging print

        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": f"User '{uid}' not found"}), 404

        user_data = user_doc.to_dict()

        co2_data = user_data.get("co2_data", [])
        # water_data = user_data.get("water_data", []) # Removed for CO2 only

        if not co2_data: # Only check for CO2 data now
            return jsonify({"error": "No historical CO2 data for this user to forecast"}), 400

        # Convert CO2 data to Pandas DataFrame for Prophet
        co2_df = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})

        # Perform CO2 forecast
        co2_pred = forecast(co2_df)
        # water_pred = forecast(water_df) # Removed for CO2 only

        # Combine results into a list of dictionaries
        result = []
        for i in range(7): # Assuming 7 days forecast from Prophet
            result.append({
                "date": str(co2_pred.iloc[i]["ds"].date()), # Convert timestamp to date string
                "co2_pred": round(co2_pred.iloc[i]["yhat"], 2),
                # "water_pred": round(water_pred.iloc[i]["yhat"], 2) # Removed for CO2 only
            })

        # Update Firestore with the new forecast (only for CO2 part)
        # Note: You might want to update this to a specific field for CO2 forecast
        # if you later want to add water forecast back separately.
        # For now, it updates 'user_forecast' but it will only contain 'co2_pred'.
        db.collection("users").document(uid).update({"user_forecast": result})

        print(f"Forecast updated successfully for user: {uid} (CO2 only)") # Debugging print
        return jsonify({"message": f"Forecast updated for user {uid} (CO2 only)"}), 200 # Return success JSON

    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error during forecast for user {uid if 'uid' in locals() else 'N/A'}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000) # Render uses port 10000
