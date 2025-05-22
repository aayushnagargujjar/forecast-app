from prophet import Prophet
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def forecast(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(7)

@app.route('/')
def run_forecast():
    users = db.collection("users").stream()

    for user in users:
        user_data = user.to_dict()
        uid = user.id

        co2_data = user_data.get("co2_data", [])
        water_data = user_data.get("water_data", [])

        if not co2_data or not water_data:
            continue

        co2_df = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})
        water_df = pd.DataFrame(water_data).rename(columns={"date": "ds", "value": "y"})

        co2_pred = forecast(co2_df)
        water_pred = forecast(water_df)

        result = []
        for i in range(3):
            result.append({
                "date": str(co2_pred.iloc[i]["ds"].date()),
                "co2_pred": round(co2_pred.iloc[i]["yhat"], 2),
                "water_pred": round(water_pred.iloc[i]["yhat"], 2)
            })

        db.collection("users").document(uid).update({"user_forecast": result})

    return "Forecast updated for all users."

if __name__ == "__main__":
    app.run()
