"""
Data loader module for IoMT Network Intrusion Detection Dashboard.
Provides robust data loading from file/env/uploader with stratified sampling,
caching, and realistic synthetic demo fallback.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# Canonical list of 45 numeric flow features from CICIoMT2024 Random Forest model
FEATURE_NAMES: List[str] = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight"
]

CANONICAL_CLASSES: List[str] = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]

DEFAULT_DATA_CANDIDATES = [
    os.environ.get("IOMT_DATA_PATH"),
    "../test_enriched.parquet",
    "../train_enriched.parquet",
    "test_enriched.parquet",
    "train_enriched.parquet",
    "../test.parquet",
    "../train.parquet",
    "test.parquet",
    "train.parquet",
    "data/test_enriched.parquet"
]


def resolve_default_path() -> Optional[str]:
    """Finds the first existing candidate data path on disk."""
    for path in DEFAULT_DATA_CANDIDATES:
        if path and Path(path).exists():
            return str(Path(path).resolve())
    return None


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic IoMT network intrusion dataset matching the CICIoMT2024 schema.
    Reproduces known domain relationships:
    - ARP, LLC, and IPv are perfectly correlated (r = 1.0)
    - Volumetric attacks (DDoS/DoS) have high Rate, Tot sum, and packet count
    - Spoofing closely overlaps Benign in Duration, Rate, and IAT
    - Binary flag columns are Bernoulli-distributed
    """
    rng = np.random.default_rng(seed)
    
    # Class proportions roughly reflecting test split
    proportions = [0.15, 0.45, 0.25, 0.08, 0.05, 0.02]
    classes = rng.choice(CANONICAL_CLASSES, size=n_samples, p=proportions)
    
    df = pd.DataFrame({"category": classes})
    
    # Base continuous features
    df["Header_Length"] = rng.gamma(shape=10.0, scale=100.0, size=n_samples)
    df["Duration"] = rng.exponential(scale=64.0, size=n_samples)
    df["Rate"] = rng.lognormal(mean=5.0, sigma=2.0, size=n_samples)
    df["Srate"] = df["Rate"] * rng.uniform(0.7, 1.0, size=n_samples)
    df["Drate"] = df["Rate"] - df["Srate"]
    
    # Rates and Tot sum spiked during DDoS and DoS
    ddos_mask = df["category"].isin(["DDoS", "DoS"])
    df.loc[ddos_mask, "Rate"] *= rng.uniform(5.0, 50.0, size=int(ddos_mask.sum()))
    df.loc[ddos_mask, "Header_Length"] *= rng.uniform(1.5, 3.0, size=int(ddos_mask.sum()))
    
    # Spoofing overlaps Benign behavior
    spoof_mask = df["category"] == "Spoofing"
    benign_mask = df["category"] == "Benign"
    df.loc[spoof_mask, "Rate"] = rng.lognormal(mean=4.0, sigma=1.2, size=int(spoof_mask.sum()))
    df.loc[benign_mask, "Rate"] = rng.lognormal(mean=3.8, sigma=1.1, size=int(benign_mask.sum()))
    
    # Perfectly correlated trio (ARP, LLC, IPv)
    arp_values = rng.choice([0.0, 1.0], size=n_samples, p=[0.9, 0.1])
    df.loc[spoof_mask, "ARP"] = 1.0
    arp_values[spoof_mask] = 1.0
    df["ARP"] = arp_values
    df["LLC"] = df["ARP"]
    df["IPv"] = df["ARP"]
    
    # Transport protocol indicators
    protocols = rng.choice([6.0, 17.0, 1.0], size=n_samples, p=[0.55, 0.40, 0.05])
    df["Protocol Type"] = protocols
    df["TCP"] = (df["Protocol Type"] == 6.0).astype(float)
    df["UDP"] = (df["Protocol Type"] == 17.0).astype(float)
    df["ICMP"] = (df["Protocol Type"] == 1.0).astype(float)
    
    # Binary TCP flags
    for flag in ["fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number"]:
        df[flag] = rng.choice([0.0, 1.0], size=n_samples, p=[0.85, 0.15]) * df["TCP"]
        
    # High SYN flags during DoS/DDoS SYN variants
    df.loc[ddos_mask, "syn_flag_number"] = rng.choice([0.0, 1.0], size=int(ddos_mask.sum()), p=[0.3, 0.7])
    
    # Count features
    for cnt in ["ack_count", "syn_count", "fin_count", "rst_count"]:
        df[cnt] = rng.poisson(lam=12.0, size=n_samples).astype(float)
        
    # Application layer protocols
    for app in ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "DHCP", "IGMP"]:
        df[app] = rng.choice([0.0, 1.0], size=n_samples, p=[0.95, 0.05])
        
    # Flow aggregations
    df["Tot sum"] = df["Rate"] * df["Duration"] + rng.normal(100, 20, size=n_samples)
    df["Min"] = rng.uniform(20.0, 60.0, size=n_samples)
    df["Max"] = df["Min"] + rng.gamma(shape=5.0, scale=150.0, size=n_samples)
    df["AVG"] = (df["Min"] + df["Max"]) / 2.0 + rng.normal(0, 10, size=n_samples)
    df["Std"] = (df["Max"] - df["Min"]) / 4.0
    df["Tot size"] = df["Tot sum"] / (df["Rate"] + 1.0)
    df["IAT"] = rng.exponential(scale=1e7, size=n_samples)
    df["Number"] = rng.uniform(1.0, 10.0, size=n_samples)
    df["Magnitue"] = np.sqrt(df["AVG"]**2 + df["Std"]**2)
    df["Radius"] = df["Std"] * 1.414
    df["Variance"] = df["Std"] ** 2
    df["Covariance"] = df["Variance"] * rng.uniform(0.8, 1.2, size=n_samples)
    df["Weight"] = rng.uniform(10.0, 50.0, size=n_samples)
    
    # Metadata columns
    df["label"] = df["category"] + "_simulated"
    df["canonical_label"] = df["category"]
    df["binary_label"] = np.where(df["category"] == "Benign", "Benign", "Attack")
    
    return df


@st.cache_data(show_spinner="Loading and preprocessing IoMT dataset...")
def load_raw_dataset(file_path_or_buffer: Any) -> Tuple[pd.DataFrame, bool, str]:
    """
    Loads dataset from parquet/csv or returns synthetic dataset fallback.
    Returns: (DataFrame, is_synthetic_flag, status_message)
    """
    if file_path_or_buffer is None:
        default_p = resolve_default_path()
        if default_p:
            file_path_or_buffer = default_p
        else:
            synth_df = generate_synthetic_data(5000)
            return synth_df, True, "No dataset path detected. Generated synthetic demo dataset."
            
    try:
        # Check if it's a file path string or an uploaded buffer
        if isinstance(file_path_or_buffer, str):
            p = Path(file_path_or_buffer)
            if not p.exists():
                synth_df = generate_synthetic_data(5000)
                return synth_df, True, f"File '{file_path_or_buffer}' not found. Using synthetic demo data."
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            source_desc = f"Loaded from disk: {p.name} ({len(df):,} rows)"
        else:
            # Uploaded buffer
            filename = getattr(file_path_or_buffer, "name", "uploaded_file")
            if filename.endswith(".parquet"):
                df = pd.read_parquet(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
            source_desc = f"Loaded uploaded file: {filename} ({len(df):,} rows)"

        # Ensure label/category columns exist
        if "category" not in df.columns:
            if "label" in df.columns:
                # Map detailed labels to canonical 5-class categories if possible
                category_map = {
                    "TCP_IP-DDoS-UDP": "DDoS", "TCP_IP-DDoS-ICMP": "DDoS",
                    "TCP_IP-DDoS-TCP": "DDoS", "TCP_IP-DDoS-SYN": "DDoS",
                    "TCP_IP-DoS-UDP": "DoS", "TCP_IP-DoS-SYN": "DoS",
                    "TCP_IP-DoS-ICMP": "DoS", "TCP_IP-DoS-TCP": "DoS",
                    "MQTT-DDoS-Connect_Flood": "MQTT", "MQTT-DDoS-Publish_Flood": "MQTT",
                    "MQTT-DoS-Publish_Flood": "MQTT", "MQTT-DoS-Connect_Flood": "MQTT",
                    "MQTT-Malformed_Data": "MQTT", "Recon-Port_Scan": "Recon",
                    "Recon-OS_Scan": "Recon", "Recon-VulScan": "Recon",
                    "Recon-Ping_Sweep": "Recon", "ARP_Spoofing": "Spoofing",
                    "Benign": "Benign"
                }
                stripped = df["label"].astype(str).str.rstrip("0123456789")
                df["canonical_label"] = stripped
                df["category"] = df["canonical_label"].map(category_map).fillna(stripped)
            else:
                df["category"] = "Benign"

        if "binary_label" not in df.columns:
            df["binary_label"] = np.where(df["category"] == "Benign", "Benign", "Attack")
            
        # Clean infinite values and replace with NaN
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        
        return df, False, source_desc

    except Exception as e:
        synth_df = generate_synthetic_data(5000)
        return synth_df, True, f"Error loading file ({str(e)}). Fallen back to synthetic demo data."


@st.cache_data(show_spinner="Applying stratified sampling...")
def get_stratified_sample(
    df: pd.DataFrame,
    target_col: str = "category",
    sample_size: int = 100000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Performs stratified sampling to ensure rare attack classes (such as Spoofing)
    are preserved proportionally in the analysis sample.
    """
    if len(df) <= sample_size:
        return df
        
    rng = np.random.default_rng(seed)
    
    # Calculate per-class allocation
    class_counts = df[target_col].value_counts()
    fractions = class_counts / len(df)
    
    sampled_indices = []
    for cls, frac in fractions.items():
        cls_indices = df[df[target_col] == cls].index.values
        # Allocate proportional sample with a guarantee of at least min(len, 50)
        cls_target_n = max(min(len(cls_indices), 50), int(np.round(frac * sample_size)))
        cls_target_n = min(cls_target_n, len(cls_indices))
        
        sampled_indices.extend(rng.choice(cls_indices, size=cls_target_n, replace=False))
        
    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)
    
    return df.loc[sampled_indices].copy()
