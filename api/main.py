from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI()

# Function to load model
def load_model():
    possible_paths = ['models/model.pkl', '../models/model.pkl']
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None

model = load_model()

# 1. Root Endpoint
@app.get("/")
def read_root():
    return {"message": "MLOps API is running!"}

# 2. Health Endpoint (REQUIRED for Task 5.1)
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 3. Predict Endpoint
@app.post("/predict")
def predict(id: int, value: int):
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    features = pd.DataFrame([[id, value]], columns=['id', 'value'])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}