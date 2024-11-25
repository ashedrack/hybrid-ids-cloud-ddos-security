import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the pre-trained Random Forest model
with open("webattack_detection_rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Define the FastAPI app
app = FastAPI()

# Input features for the prediction
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
        data = np.array([[features.Packet_Length_Std, features.Bwd_Packet_Length_Std,
                          features.Packet_Length_Variance, features.Bwd_Packet_Length_Mean,
                          features.Avg_Bwd_Segment_Size, features.Bwd_Packet_Length_Max,
                          features.Max_Packet_Length, features.Fwd_IAT_Std,
                          features.Average_Packet_Size, features.Fwd_Header_Length_1]])

        # Perform prediction using the Random Forest model
        rf_pred = rf_model.predict(data)
        rf_pred_label = int(rf_pred[0])  # Convert prediction to integer

        return {"RF_Prediction": rf_pred_label}
    except Exception as e:
        return {"detail": f"Error during prediction: {str(e)}"}

# Entry point for deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT from environment variable or default to 8000
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
