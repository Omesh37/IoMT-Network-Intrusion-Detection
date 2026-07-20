# IoMT Network Intrusion Detection

ML-based intrusion detection for medical IoT (IoMT) network traffic, using the CICIoMT2024 dataset (Canadian Institute for Cybersecurity, UNB). Classifies network flows as Benign or one of 5 attack categories (DDoS, DoS, MQTT, Recon, Spoofing), served via a FastAPI endpoint, containerized with Docker.

## Problem
IoMT devices are increasingly targeted, this project builds a detector for 18 real attack variants across 5 categories against a realistic medical-device network testbed.

## Data Engineering
- Source: CICIoMT2024 `WiFi_and_MQTT/csv` split — 51 raw files, ~1.9GB, one file per attack-capture scenario
- Built a loader to reconstruct labels from filenames (no label column in the raw data)
- Collapsed chunked capture files (e.g. `UDP1`...`UDP8`) into canonical labels via string processing
- Mapped 19 canonical labels to 5 top-level categories using the paper's taxonomy
- Persisted to Parquet at each stage for fast reload

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