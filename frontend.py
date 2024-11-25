import streamlit as st
import requests
from PIL import Image

# Load and display a banner image
image = Image.open("ai_ml_banner.jpg")  # Ensure you have a suitable image named 'ai_ml_banner.jpg' in your working directory
st.image(image, use_column_width=True)

# Streamlit app title with styled header
st.markdown(
    """
    <div style="background-color:#002b5c;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">INTRUSION DETECTION SYSTEM</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Instruction text
st.write(
    """
    This system leverages machine learning to predict whether a network behavior is **Benign** or indicates a **Potential Attack**.
    Please fill in the following feature values to make a prediction.
    """
)

# User input fields for the selected 10 features
st.sidebar.header("Enter Feature Values:")
Packet_Length_Std = st.sidebar.number_input("Packet Length Std", value=0.0, format="%.2f")
Bwd_Packet_Length_Std = st.sidebar.number_input("Bwd Packet Length Std", value=0.0, format="%.2f")
Packet_Length_Variance = st.sidebar.number_input("Packet Length Variance", value=0.0, format="%.2f")
Bwd_Packet_Length_Mean = st.sidebar.number_input("Bwd Packet Length Mean", value=0.0, format="%.2f")
Avg_Bwd_Segment_Size = st.sidebar.number_input("Avg Bwd Segment Size", value=0.0, format="%.2f")
Bwd_Packet_Length_Max = st.sidebar.number_input("Bwd Packet Length Max", value=0.0, format="%.2f")
Max_Packet_Length = st.sidebar.number_input("Max Packet Length", value=0.0, format="%.2f")
Fwd_IAT_Std = st.sidebar.number_input("Fwd IAT Std", value=0.0, format="%.2f")
Average_Packet_Size = st.sidebar.number_input("Average Packet Size", value=0.0, format="%.2f")
Fwd_Header_Length_1 = st.sidebar.number_input("Fwd Header Length.1", value=0.0, format="%.2f")

# Predict button with animation
if st.button("Predict"):
    st.markdown(
        """
        <div style="background-color:#e6f7ff;padding:10px;border-radius:5px">
            <h3 style="color:#00509e;text-align:center;">Processing your prediction...</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            prediction_label = "🚨 **Attack**" if prediction["RF_Prediction"] == 1 else "✅ **Benign**"
            st.success(f"Model Prediction: {prediction_label}")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

# Footer with link
st.markdown(
    """
    <hr>
    <p style="text-align:center;">
        <small>Powered by <b>Machine Learning</b> & <b>FastAPI</b></small>
    </p>
    """,
    unsafe_allow_html=True,
)
