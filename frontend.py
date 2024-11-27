import streamlit as st
import requests
from datetime import datetime

# Streamlit app title
st.title("📡 INTRUSION DETECTION SYSTEM")
st.write("### A Machine Learning-powered tool to detect network intrusions.")
st.markdown("---")

# User input fields for the selected 10 features
st.header("🔢 Enter Feature Values")
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
if st.button("Predict 🚀"):
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
        response = requests.post("https://hybrid-ids-cloud-ddos-security-2.onrender.com/predict", json=input_data)

        # Parse response
        if response.status_code == 200:
            prediction = response.json()
            prediction_label = "🔴 Attack" if prediction["RF_Prediction"] == 1 else "🟢 Normal"
            severity = prediction.get("Severity", "low")  # Default to "low" if Severity key is not present
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Display results
            st.success(f"**Model Prediction:** {prediction_label}")
            #st.write(f"**Severity Level:** {severity.capitalize()}")
            st.write(f"**Timestamp:** {timestamp}")

            # Custom status messages based on severity level
            if prediction["RF_Prediction"] == 1:
                if severity.lower() == "high":
                    st.error("🔴 **High severity intrusion detected! Immediate action is required to mitigate potential risks.**")
                elif severity.lower() == "high":
                    st.info("🔴 **High severity intrusion detected! Immediate action is required to mitigate potential risks.**")
                else:
                    st.warning("🔴 **High severity intrusion detected! Immediate action is required to mitigate potential risks.**")
            else:
                st.success("✅ No intrusion detected. The network is secure.")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.write("💡 **Note**: This tool uses advanced machine learning algorithms to predict network intrusions.")
