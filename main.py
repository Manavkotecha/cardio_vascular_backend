import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORT THIS

# 1. Load the trained pipeline
model_pipeline = joblib.load("cardio_ensemble_model.joblib")

app = FastAPI()

# 2. ADD THIS BLOCK TO FIX THE 405 ERROR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (simplest for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Define Input Schema 
class PatientData(BaseModel):
    age: int          
    gender: int       
    ap_hi: int        
    ap_lo: int        
    cholesterol: int  
    gluc: int         
    smoke: int        
    alco: int         
    active: int       
    bmi: float        

@app.get("/")
def home():
    return {"message": "Cardio Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model_pipeline.predict(input_df)
    
    # Note: Your frontend expects "risk" but here you return "cardio_disease_prediction"
    # It's safer to match what the frontend expects:
    return {"risk": int(prediction[0])}