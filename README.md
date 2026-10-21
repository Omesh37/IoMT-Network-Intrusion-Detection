# Securing Medical IoT: A Statistical and Machine Learning Approach to Network Intrusion Detection

ML-based intrusion detection for medical IoT (IoMT) network traffic, using the CICIoMT2024 dataset (Canadian Institute for Cybersecurity, UNB). Classifies network flows as Benign or one of 5 attack categories (DDoS, DoS, MQTT, Recon, Spoofing), served via a FastAPI endpoint, containerized with Docker.

## Problem
IoMT devices are increasingly targeted, this project builds a detector for 18 real attack variants across 5 categories against a realistic medical-device network testbed.

## Data Engineering
- Source: CICIoMT2024 `WiFi_and_MQTT/csv` split — 51 raw files, ~1.9GB, one file per attack-capture scenario
- Built a loader to reconstruct labels from filenames (no label column in the raw data)
- Collapsed chunked capture files (e.g. `UDP1`...`UDP8`) into canonical labels via string processing
- Mapped 19 canonical labels to 5 top-level categories using the paper's taxonomy
- Persisted to Parquet at each stage for fast reload
## EDA
- Class balance: The catergory of attack is imbalance, example the DDoS is around 5 time more than Spoofing and same with for Benign, MQTT and Recon. Number of Attack is 7 time more than Benign.<img width="678" height="421" alt="download" src="https://github.com/user-attachments/assets/04b22d35-dac4-438c-8a9d-8c8ff129941d" /><img width="678" height="430" alt="download" src="https://github.com/user-attachments/assets/27eeb62a-4884-4b4e-9fba-b53cbe19af5b" />

- Duration/Rate/IAT overlap: Spoofing distribution almost overlap the distribution of Benign, which make it more harder to differenticate between them.
<img width="700" height="468" alt="download" src="https://github.com/user-attachments/assets/383e9b51-a9b5-466b-9fce-e82541e1748d" />

- Correlation:
ARP     IPv       1.000000
LLC     ARP       1.000000
IPv     ARP       1.000000
ARP     LLC       1.000000
These are highly correlated pair as per Heat map.<img width="1127" height="989" alt="download" src="https://github.com/user-attachments/assets/a29dc114-ccc5-495a-a3e3-3a316cf0212c" />

Distribution of Traffic Categories and binary classification balance
<img width="1243" height="590" alt="image" src="https://github.com/user-attachments/assets/f6e2b628-8522-4b55-a9c4-98e9311c2dfb" />

<img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/ad114b29-edd2-45f2-84f6-31bce88696dd" />

- Conclusion: This EDA supports the 0.74 precision on Spoofing seen in the classification report.
## Modeling
Staged approach —> binary first to validate the pipeline, then 5-class:
- **Binary (Benign vs Attack)**: Attack recall ~1.00, Benign recall 0.95
- **5-class**: near-perfect on high-volume attacks (DDoS/DoS/MQTT ≈ 1.00 F1); weaker on Spoofing (precision 0.74, recall 0.82) — the smallest class (16K rows) and behaviorally closer to benign traffic than volumetric attacks
- Used a majority-class baseline given the 36:1 Benign:Attack imbalance — reported macro-avg alongside weighted-avg, since weighted-avg (0.99) hides the Spoofing weakness that macro-avg (0.95) reveals
- Verified feature importances (Max, Magnitude, Variance, rst_count) to rule out data leakage before trusting results

## Serving
FastAPI `/predict` endpoint, Pydantic-validated 44-feature schema, returns predicted category + per-class confidence. Containerized with Docker.

## Run it
\`\`\`
docker build -t iomt-api .
docker run -p 8000:8000 iomt-api
\`\`\`
Then visit `http://127.0.0.1:8000/docs`.

## Limitations & future work
- Spoofing detection needs more data or targeted resampling (SMOTE untested)
- No persistent logging/monitoring of live predictions yet
- Adversarial robustness against evasion untested — planned next step
- Frontend is a minimal Streamlit demo; a proper UI is future work

## Dataset citation
Dadkhah et al., CICIoMT2024, Canadian Institute for Cybersecurity, University of New Brunswick.
