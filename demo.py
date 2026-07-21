import streamlit as st
import joblib
import pandas as pd

# Load model once
rf_model = joblib.load("rf_5class_full.joblib")

st.title("IoMT Network Intrusion Detection")
sample = st.text_area("Feature JSON", height=300)

if st.button("Predict"):
    try:
        # Parse JSON string into dict
        features = eval(sample)
        input_df = pd.DataFrame([features])
        input_df = input_df[rf_model.feature_names_in_]  # enforce column order

        prediction = rf_model.predict(input_df)[0]
        proba = rf_model.predict_proba(input_df)[0]
        confidence = dict(zip(rf_model.classes_.tolist(), proba.tolist()))

        st.write("Prediction:", prediction)
        st.json(confidence)
    except Exception as e:
        st.error(f"Error: {e}")
