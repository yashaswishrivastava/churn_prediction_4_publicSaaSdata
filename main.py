from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load model and feature names
with open("models/xgb_churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

class Customer(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: Customer):
    input_dict = customer.dict()
    input_df = pd.DataFrame([input_dict])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    prob = model.predict_proba(input_df)[0][1]
    risk = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"

    return {
        "churn_probability": round(float(prob), 4),
        "risk_tier": risk,
        "message": f"This customer has a {risk} risk of churning"
    }

@app.get("/health")
def health():
    return {"status": "ok"}