
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import sklearn

from pathlib import Path

st.set_page_config(
    page_title="University Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 University Cutoff Score Predictor")
st.write(
    "Predict university admission cutoff scores using "
    "your trained machine learning model."
)

# Always search in the same directory as app.py
BASE_DIR = Path(__file__).resolve().parent

# Support either filename
MODEL_CANDIDATES = [
    BASE_DIR / "model.joblib",
    BASE_DIR / "model.pkl"
]

@st.cache_resource
def load_model(filepath):
    return joblib.load(filepath)

model = None
model_path = next(
    (path for path in MODEL_CANDIDATES if path.is_file()),
    None
)

if model_path is None:
    st.error(
        "No trained model file was found. "
        "Upload model.joblib or model.pkl."
    )

    uploaded_file = st.file_uploader(
        "Upload your trained model",
        type=["joblib", "pkl"]
    )

    if uploaded_file is not None:
        # Save uploaded file using its original filename
        safe_name = Path(uploaded_file.name).name
        destination = BASE_DIR / safe_name

        with open(destination, "wb") as file:
            file.write(uploaded_file.getbuffer())

        st.success("Model file saved. Reloading the app...")
        st.rerun()

    st.stop()

try:
    model = load_model(str(model_path))
    st.sidebar.success(f"Model loaded: {model_path.name}")

except Exception as exc:
    st.error("Unable to load the trained model.")

    st.code(
        f"Error type: {type(exc).__name__}\n"
        f"Error message: {exc}"
    )

    with st.expander("Environment details"):
        st.write("Python:", sys.version)
        st.write("NumPy:", np.__version__)
        st.write("Pandas:", pd.__version__)
        st.write("Scikit-learn:", sklearn.__version__)
        st.write("Joblib:", joblib.__version__)
        st.write("Model file:", model_path.name)

    st.info(
        "Check the original training environment and the "
        "missing class/module named in the error. "
        "Do not retrain until the cause is identified."
    )
    st.stop()


st.subheader("Configure Input Parameters")

col1, col2 = st.columns(2)

with col1:
    university = st.text_input(
        "University Name", "Banaras Hindu University"
    )
    university_id = st.text_input(
        "University ID", "banaras-hindu-university"
    )
    programme_id = st.text_input(
        "Programme ID",
        "banaras-hindu-university-b-a-honours"
    )
    college = st.text_input("College Name", "Unknown")
    variant = st.text_input("Variant/Branch", "Unknown")
    year = st.number_input(
        "Admission Year",
        min_value=2024,
        max_value=2030,
        value=2025,
        step=1
    )

with col2:
    round_num = st.selectbox(
        "Round", ["1", "2", "3", "4", "spot"]
    )
    category = st.selectbox(
        "Category", ["UR", "SC", "ST", "OBC", "EWS", "PwBD"]
    )
    seat_type = st.text_input("Seat Type", "general")
    basis = st.selectbox("Basis", ["official", "merit-list"])
    source_id = st.text_input(
        "Source ID", "banaras-hindu-university-2024-r1"
    )

input_data = pd.DataFrame({
    "university_id": [university_id],
    "programme_id": [programme_id],
    "college": [college],
    "variant": [variant],
    "round": [str(round_num)],
    "category": [category],
    "seat_type": [seat_type],
    "basis": [basis],
    "source_id": [source_id],
    "year": [year]
})

st.write("### Input Preview")
st.dataframe(input_data, use_container_width=True)

if st.button("🎯 Predict Cutoff Score", type="primary"):
    try:
        # Validate expected input columns when available
        expected = getattr(model, "feature_names_in_", None)

        if expected is not None:
            missing = set(expected) - set(input_data.columns)
            extra = set(input_data.columns) - set(expected)

            if missing or extra:
                st.error(
                    f"Feature mismatch. Missing: {sorted(missing)}; "
                    f"Extra: {sorted(extra)}"
                )
                st.stop()

            input_data = input_data[list(expected)]

        prediction = model.predict(input_data)[0]

        st.balloons()
        st.metric(
            label="Predicted Cutoff Score",
            value=f"{float(prediction):.4f}"
        )

    except Exception as exc:
        st.error(
            f"Prediction failed: {type(exc).__name__}: {exc}"
        )
        st.info(
            "Verify that the model was trained using the same "
            "input columns, data types, and preprocessing pipeline."
        )
