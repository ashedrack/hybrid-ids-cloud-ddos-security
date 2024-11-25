from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Load the Random Forest model
try:
    with open("webattack_detection_rf_model.pkl", "rb") as file:
        rf_model = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Failed to load Random Forest model: {str(e)}")

# Define the FastAPI app
app = FastAPI()

# Input schema for the prediction
class AttackFeatures(BaseModel):
    Packet_Length_Std: float
    Bwd_Packet_Length_Std: float
    Packet_Length_Variance: float
    Bwd_Packet_Length_Mean: float
    Avg_Bwd_Segment_Size: float
    Bwd_Packet_Length_Max: float
    Max_Packet_Length: float
    Fwd_IAT_Std: float
    Average_Packet_Size: float
    Fwd_Header_Length_1: float

@app.post("/predict")
async def predict(features: AttackFeatures):
    try:
        # Convert input to numpy array
        data = np.array([[ 
            features.Packet_Length_Std,
            features.Bwd_Packet_Length_Std,
            features.Packet_Length_Variance,
            features.Bwd_Packet_Length_Mean,
            features.Avg_Bwd_Segment_Size,
            features.Bwd_Packet_Length_Max,
            features.Max_Packet_Length,
            features.Fwd_IAT_Std,
            features.Average_Packet_Size,
            features.Fwd_Header_Length_1
        ]])

        # Check if input data shape matches the Random Forest model
        expected_features = rf_model.n_features_in_
        if data.shape[1] != expected_features:
            raise ValueError(f"Invalid input shape for Random Forest model. Expected {expected_features} features, got {data.shape[1]}.")

        # Random Forest prediction
        rf_pred = rf_model.predict(data)
        rf_pred_label = int(rf_pred[0])  # Extract the predicted class label

        return {
            "RF_Prediction": rf_pred_label
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
