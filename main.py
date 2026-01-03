import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load the trained pipeline
model_pipeline = joblib.load("cardio_ensemble_model.joblib")

app = FastAPI()

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema - must match the 12 features used in training
class PatientData(BaseModel):
    age: int          
    gender: int       
    height: int       # in cm
    weight: float     # in kg
    ap_hi: int        # systolic blood pressure
    ap_lo: int        # diastolic blood pressure
    cholesterol: int  # 1=normal, 2=above normal, 3=well above normal
    gluc: int         # 1=normal, 2=above normal, 3=well above normal
    smoke: int        # 0 or 1
    alco: int         # 0 or 1
    active: int       # 0 or 1
    bmi: float        

@app.get("/")
def home():
    return {"message": "Cardio Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    # Create DataFrame with columns in the EXACT order used during training
    input_df = pd.DataFrame([[
        data.age,
        data.gender,
        data.height,
        data.weight,
        data.ap_hi,
        data.ap_lo,
        data.cholesterol,
        data.gluc,
        data.smoke,
        data.alco,
        data.active,
        data.bmi
    ]], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi'])
    
    prediction = model_pipeline.predict(input_df)
    return {"risk": int(prediction[0])}
