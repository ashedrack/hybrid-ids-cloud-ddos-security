import streamlit as st
import requests

# Streamlit app title
st.title("Web Attack Detection with Random Forest")

# User input fields for the selected 10 features
st.header("Enter Feature Values:")
Packet_Length_Std = st.number_input("Packet Length Std", value=0.0)
Bwd_Packet_Length_Std = st.number_input("Bwd Packet Length Std", value=0.0)
Packet_Length_Variance = st.number_input("Packet Length Variance", value=0.0)
Bwd_Packet_Length_Mean = st.number_input("Bwd Packet Length Mean", value=0.0)
Avg_Bwd_Segment_Size = st.number_input("Avg Bwd Segment Size", value=0.0)
Bwd_Packet_Length_Max = st.number_input("Bwd Packet Length Max", value=0.0)
Max_Packet_Length = st.number_input("Max Packet Length", value=0.0)
Fwd_IAT_Std = st.number_input("Fwd IAT Std", value=0.0)
Average_Packet_Size = st.number_input("Average Packet Size", value=0.0)
Fwd_Header_Length_1 = st.number_input("Fwd Header Length.1", value=0.0)

# Button for prediction
if st.button("Predict"):
    # Prepare input data
    input_data = {
        "Packet_Length_Std": Packet_Length_Std,
        "Bwd_Packet_Length_Std": Bwd_Packet_Length_Std,
        "Packet_Length_Variance": Packet_Length_Variance,
        "Bwd_Packet_Length_Mean": Bwd_Packet_Length_Mean,
        "Avg_Bwd_Segment_Size": Avg_Bwd_Segment_Size,
        "Bwd_Packet_Length_Max": Bwd_Packet_Length_Max,
        "Max_Packet_Length": Max_Packet_Length,
        "Fwd_IAT_Std": Fwd_IAT_Std,
        "Average_Packet_Size": Average_Packet_Size,
        "Fwd_Header_Length_1": Fwd_Header_Length_1,
    }

    # Send POST request to FastAPI
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

        # Parse response
        if response.status_code == 200:
            prediction = response.json()
            prediction_label = "Attack" if prediction["RF_Prediction"] == 1 else "Benign"
            st.success(f"Random Forest Prediction: {prediction_label}")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
