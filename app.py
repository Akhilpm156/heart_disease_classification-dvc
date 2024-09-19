from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
import joblib
import os
from src.data_preprocessing import main as preprocess
from src.model_training import main as training

def load_model(model_path='models/random_forest_model.pkl'):
    """Loads a trained model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

app = FastAPI()

class PredictionRequest(BaseModel):
    data: List[List[float]]

@app.get("/")
def read_root():
    return {"message": "Model API is running!"}

@app.post("/train")
def train():
    preprocess()  # Call preprocessing function
    training()    # Call training function
    return {"message": "Model training completed"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame(request.data, columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    
    model_path = 'models/random_forest_model.pkl'
    model = load_model(model_path)

    predictions = model.predict(data)
    
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
