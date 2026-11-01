from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
with open("models/customer_churn_model.pkl", "rb") as file:
    model = pickle.load(file)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    customer = pd.DataFrame([{
        "gender": request.form["gender"],
        "SeniorCitizen": 1 if request.form["senior_citizen"] == "Yes" else 0,
        "Partner": request.form["partner"],
        "Dependents": request.form["dependents"],
        "tenure": float(request.form["tenure"]),
        "PhoneService": request.form["phone_service"],
        "MultipleLines": request.form["multiple_lines"],
        "InternetService": request.form["internet_service"],
        "OnlineSecurity": request.form["online_security"],
        "OnlineBackup": request.form["online_backup"],
        "DeviceProtection": request.form["device_protection"],
        "TechSupport": request.form["tech_support"],
        "StreamingTV": request.form["streaming_tv"],
        "StreamingMovies": request.form["streaming_movies"],
        "Contract": request.form["contract"],
        "PaperlessBilling": request.form["paperless_billing"],
        "PaymentMethod": request.form["payment_method"],
        "MonthlyCharges": float(request.form["monthly_charges"]),
        "TotalCharges": float(request.form["total_charges"])
    }])
    prediction = model.predict(customer)[0]
    probability = round(model.predict_proba(customer)[0][1] * 100, 2)
    if prediction == 1:
        prediction_text = "🔴 Customer has a HIGH chance of Churning"
    else:
        prediction_text = "🟢 Customer is likely to Continue the Service"
    if probability >= 70:
        risk_level = "HIGH"

    elif probability >= 40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    if risk_level == "HIGH":

        recommendations = [
            "Offer a personalized loyalty discount.",
            "Provide priority customer support.",
            "Upgrade the customer to a yearly contract.",
            "Assign the customer to the retention team.",
            "Offer exclusive promotional benefits."
        ]

    elif risk_level == "MEDIUM":

        recommendations = [
            "Provide personalized offers.",
            "Increase customer engagement.",
            "Recommend suitable service upgrades.",
            "Monitor customer satisfaction regularly."
        ]

    else:

        recommendations = [
            "Maintain regular customer engagement.",
            "Continue providing excellent customer service.",
            "Reward customer loyalty through bonus offers.",
            "Send appreciation and promotional emails."
        ]

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        probability=probability,
        risk_level=risk_level,
        recommendation=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)