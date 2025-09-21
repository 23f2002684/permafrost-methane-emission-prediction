import streamlit as st
import os
from predict import predict_png
from PIL import Image

st.set_page_config(page_title="Permafrost Methane Predictor", layout="wide")

st.title("🛰️ Permafrost Methane Emission Predictor")
st.write("""
Upload a satellite image (PNG or JPG) of a permafrost region to predict its methane emission risk.
**Note:** This is a demo model trained on a very small dataset and is for illustrative purposes only.
""")

# --- Instructions & Disclaimers ---
with st.expander("About the Classes & Model"):
    st.info("""
    - **High Emission Risk:** Areas with visible signs of major permafrost thaw.
    - **Low Emission Risk:** Stable, undisturbed tundra regions.
    - **Moderate Emission Risk / Null:** These classes were not included in the initial training data. The model will classify all images as either 'High' or 'Low' risk.
    """)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a PNG or JPG file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.success(f"Analyzing '{uploaded_file.name}'...")

    # Run prediction
    try:
        predicted_class, confidence = predict_png(uploaded_file)

        st.subheader("Prediction Result")
        if "High" in predicted_class:
            st.error(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")