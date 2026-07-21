import streamlit as st
import requests

st.title("IoMT Network Intrusion Detection")
st.write("Paste a flow's feature values (JSON) to classify it.")

sample = st.text_area("Feature JSON", height=300)

if st.button("Predict"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=eval(sample))
        st.json(response.json())
    except Exception as e:
        st.error(f"Error: {e}")