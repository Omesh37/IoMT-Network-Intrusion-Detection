import streamlit as st
import requests

st.title("IoMT Network Intrusion Detection")
st.write("Paste a flow's feature values (JSON) to classify it.")

sample = st.text_area("Feature JSON", height=300)

st.write("""
Sample JSON format:
{
  "Header_Length": 3609.0,
  "Protocol Type": 17.0,
  "Duration": 64.0,
  "Rate": 212.9614977723726,
  "Srate": 212.9614977723726,
  "Drate": 0.0,
  "fin_flag_number": 0.0,
  "syn_flag_number": 0.0,
  "rst_flag_number": 0.0,
  "psh_flag_number": 0.0,
  "ack_flag_number": 0.0,
  "ece_flag_number": 0.0,
  "cwr_flag_number": 0.0,
  "ack_count": 0.0,
  "syn_count": 0.0,
  "fin_count": 0.0,
  "rst_count": 0.0,
  "HTTP": 0.0,
  "HTTPS": 0.0,
  "DNS": 0.0,
  "Telnet": 0.0,
  "SMTP": 0.0,
  "SSH": 0.0,
  "IRC": 0.0,
  "TCP": 0.0,
  "UDP": 1.0,
  "DHCP": 0.0,
  "ARP": 0.0,
  "ICMP": 0.0,
  "IGMP": 0.0,
  "IPv": 1.0,
  "LLC": 1.0,
  "Tot sum": 3609.0,
  "Min": 154.8,
  "Max": 1228.4,
  "AVG": 610.0088888888888,
  "Std": 487.4092996797132,
  "Tot size": 673.2,
  "IAT": 169402747.0809622,
  "Number": 5.5,
  "Magnitue": 34.69535543626026,
  "Radius": 689.3008420338226,
  "Covariance": 279086.09952733683,
  "Variance": 0.9,
  "Weight": 38.5
}
""")
if st.button("Predict"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=eval(sample))
        st.json(response.json())
    except Exception as e:
        st.error(f"Error: {e}")