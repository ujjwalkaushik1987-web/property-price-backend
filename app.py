from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model + scaler
model = joblib.load("numeric_model.pkl")
scaler = joblib.load("scaler.pkl")

class PropertyInput(BaseModel):
    latitude: float
    longitude: float
    area_sqft: float
    bedrooms: int
    bathrooms: int

@app.get("/")
def root():
    return {"message": "Numeric Property Price Prediction API is running!"}

@app.post("/predict")
def predict_price(data: PropertyInput):
    x = np.array([[data.latitude, data.longitude, data.area_sqft, data.bedrooms, data.bathrooms]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return {"predicted_price": float(pred)}
