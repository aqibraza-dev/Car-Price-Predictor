from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle

# --- Initialize FastAPI app ---
app = FastAPI(title="Car Price Prediction API", version="1.0")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, e.g., localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model and data ---
try:
    model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
except FileNotFoundError:
    print("="*50)
    print("ERROR: 'LinearRegressionModel.pkl' not found.")
    print("Please run the 'train_model.py' script first to create it.")
    print("="*50)
    model = None

try:
    car = pd.read_csv("Cleaned_Car_data.csv")
except FileNotFoundError:
    print("="*50)
    print("ERROR: 'Cleaned_Car_data.csv' not found.")
    print("Make sure it's in the same directory as main.py")
    print("="*50)
    car = pd.DataFrame(columns=["company", "name", "year", "fuel_type"])


# --- Root endpoint (metadata or UI data) ---
@app.get("/")
def get_metadata():
    """
    Returns dropdown options for frontend UI.
    """
    # --- FIX 1: Convert all numpy types to standard Python lists ---
    companies = sorted(car["company"].unique().tolist())
    car_models = sorted(car["name"].unique().tolist())
    years = sorted(car["year"].unique().tolist(), reverse=True)
    fuel_types = car["fuel_type"].unique().tolist()

    # --- FIX 2: Removed `companies.insert(0, 'Select Company')` ---
    # The frontend will handle this logic to prevent crashes.

    return {
        "companies": companies,
        "car_models": car_models,
        "years": years,
        "fuel_types": fuel_types
    }

# --- Prediction endpoint ---
@app.post("/predict")
def predict_price(
    company: str = Form(...),
    car_model: str = Form(...),
    # --- FIX 3: Changed types to `str` to match your model ---
    year: str = Form(...),
    fuel_type: str = Form(...),
    kilo_driven: str = Form(...)
):
    """
    Predicts car price based on input parameters.
    """
    
    if model is None:
        return {"error": "Model not loaded. Please run train_model.py and restart the server."}, 500

    # Prepare input data
    # This creates an array of strings, which your model expects
    input_df = pd.DataFrame(
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
        data=np.array([car_model, company, year, kilo_driven, fuel_type]).reshape(1, 5)
    )

    # Predict
    prediction = model.predict(input_df)
    
    # --- FIX 4: Removed `np.exp()` ---
    # Your model predicts the price directly, not the log price.
    price = np.round(prediction[0], 2)

    # Convert final price to standard Python float
    return {"predicted_price": float(price)}

# --- Run with: uvicorn main:app --reload ---
