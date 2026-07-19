from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

rf_model = joblib.load("rf_5class_full.joblib")

class FlowFeatures(BaseModel):
    model_config = {"populate_by_name": True}
    protocol_type: float = Field(..., alias="Protocol Type")
    duration: float = Field(..., alias="Duration")
    header_length: float = Field(..., alias="Header_Length")
    rate: float = Field(..., alias="Rate")
    srate: float = Field(..., alias="Srate")
    drate: float = Field(..., alias="Drate")
    fin_flag_number: float = Field(..., alias="fin_flag_number")
    syn_flag_number: float = Field(..., alias="syn_flag_number")
    rst_flag_number: float = Field(..., alias="rst_flag_number")
    psh_flag_number: float = Field(..., alias="psh_flag_number")
    ack_flag_number: float = Field(..., alias="ack_flag_number")
    ece_flag_number: float = Field(..., alias="ece_flag_number")
    cwr_flag_number: float = Field(..., alias="cwr_flag_number")
    ack_count: float = Field(..., alias="ack_count")
    syn_count: float = Field(..., alias="syn_count")
    fin_count: float = Field(..., alias="fin_count")
    rst_count: float = Field(..., alias="rst_count")
    http: float = Field(..., alias="HTTP")
    https: float = Field(..., alias="HTTPS")
    dns: float = Field(..., alias="DNS")
    telnet: float = Field(..., alias="Telnet")
    smtp: float = Field(..., alias="SMTP")
    ssh: float = Field(..., alias="SSH")
    irc: float = Field(..., alias="IRC")
    tcp: float = Field(..., alias="TCP")
    udp: float = Field(..., alias="UDP")
    dhcp: float = Field(..., alias="DHCP")
    arp: float = Field(..., alias="ARP")
    icmp: float = Field(..., alias="ICMP")
    igmp: float = Field(..., alias="IGMP")
    ipv: float = Field(..., alias="IPv")
    llc: float = Field(..., alias="LLC")
    tot_sum: float = Field(..., alias="Tot sum")
    min: float = Field(..., alias="Min")
    max: float = Field(..., alias="Max")
    avg: float = Field(..., alias="AVG")
    std: float = Field(..., alias="Std")
    tot_size: float = Field(..., alias="Tot size")
    iat: float = Field(..., alias="IAT")
    number: float = Field(..., alias="Number")
    magnitue: float = Field(..., alias="Magnitue")
    radius: float = Field(..., alias="Radius")
    covariance: float = Field(..., alias="Covariance")
    variance: float = Field(..., alias="Variance")
    weight: float = Field(..., alias="Weight")

@app.post("/predict")
def predict(data: FlowFeatures):
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])
    input_df = input_df[rf_model.feature_names_in_]  # match training column order exactly

    prediction = rf_model.predict(input_df)[0]
    proba = rf_model.predict_proba(input_df)[0]
    confidence = dict(zip(rf_model.classes_.tolist(), proba.tolist()))

    return {
        "prediction": str(prediction),
        "confidence": confidence
    }