import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page title and description
st.set_page_config(page_title="University Cutoff Predictor", layout="centered")

st.title("🎓 University Cutoff Score Predictor")
st.write("This web application predicts university admission cutoff scores based on structural features. Provide input features in the sidebar or main panel below to get real-time predictions.")

# Cached Model Loader
@st.cache_resource
def load_model(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None

model_file = 'model.joblib'
model = load_model(model_file)

# Fallback file upload if model.joblib is not present
if model is None:
    st.warning(f"Could not find '{model_file}' in the directory.")
    uploaded_file = st.file_uploader("Please upload your trained 'model.joblib' file:", type=['joblib'])
    if uploaded_file is not None:
        with open(model_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("Model uploaded successfully! Reloading...")
        st.rerun()
else:
    st.sidebar.success("Model loaded successfully!")

if model is not None:
    st.subheader("Configure Input Parameters")

    # Dynamically setup user inputs based on expected categorical/numerical features
    col1, col2 = st.columns(2)

    with col1:
        university = st.text_input("University Name", "Banaras Hindu University")
        university_id = st.text_input("University ID (slug)", "banaras-hindu-university")
        programme_id = st.text_input("Programme ID", "banaras-hindu-university-b-a-honours")
        college = st.text_input("College Name", "Unknown")
        variant = st.text_input("Variant/Branch", "Unknown")
        year = st.number_input("Admission Year", min_value=2024, max_value=2030, value=2025, step=1)

    with col2:
        round_num = st.selectbox("Round", ['1', '2', '3', '4', 'spot'])
        category = st.selectbox("Category", ['UR', 'SC', 'ST', 'OBC', 'EWS', 'PwBD'])
        seat_type = st.text_input("Seat Type", "general")
        basis = st.selectbox("Basis", ['official', 'merit-list'])
        source_id = st.text_input("Source ID", "banaras-hindu-university-2024-r1")

    # Map variables to correct feature order
    input_data = pd.DataFrame({
        'university_id': [university_id],
        'programme_id': [programme_id],
        'college': [college],
        'variant': [variant],
        'round': [str(round_num)],
        'category': [category],
        'seat_type': [seat_type],
        'basis': [basis],
        'source_id': [source_id],
        'year': [year]
    })

    st.write("### Input Preview")
    st.dataframe(input_data)

    if st.button("🎯 Predict Cutoff Score", type="primary"):
        try:
            # Generate prediction using our pipeline
            prediction = model.predict(input_data)[0]

            st.balloons()
            st.metric(label="Predicted Cutoff Score", value=f"{prediction:.4f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure that your input format matches the pipeline's expectations.")