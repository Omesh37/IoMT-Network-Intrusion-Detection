"""
Tab 6: Model & Error Analysis (Unit 6 / ML Link)
Evaluates rf_5class_full.joblib on held-out samples, plots confusion matrix,
analyzes Macro vs Weighted averages, computes Type I (FPR) and Type II (FNR) error rates
with 95% Wilson Score CIs, plots feature importances, and provides an interactive live predictor.
"""

from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px

from utils.stats_utils import classification_error_analysis
from utils.plot_utils import plot_confusion_matrix, apply_theme


MODEL_PATH = Path("rf_5class_full.joblib")


@st.cache_resource(show_spinner="Loading trained Random Forest model...")
def load_rf_model():
    """Loads the pre-trained Random Forest model from disk if present."""
    if MODEL_PATH.exists():
        try:
            model = joblib.load(str(MODEL_PATH))
            return model, None
        except Exception as e:
            return None, f"Failed to deserialize model: {str(e)}"
    return None, f"Model file '{MODEL_PATH}' not found in repo root."


def render_tab6(df: pd.DataFrame, full_df: pd.DataFrame):
    st.header("Unit 6: Machine Learning Intrusion Detector & Statistical Error Analysis")

    # Concept Expander
    with st.expander("📘 Statistical Classification & Machine Learning Error Metrics (Unit 6)", expanded=False):
        st.markdown(r"""
        ### Statistical Evaluation of Multiclass Intrusion Detectors
        - **Type I Error Rate (False Positive Rate, FPR)**:
          $$FPR_i = \frac{FP_i}{FP_i + TN_i} \implies P(\text{Flagged as Class } i \mid \text{True Class } \neq i)$$
        - **Type II Error Rate (False Negative Rate, FNR)**:
          $$FNR_i = \frac{FN_i}{FN_i + TP_i} = 1 - \text{Recall}_i \implies P(\text{Missed Class } i \mid \text{True Class } = i)$$
        - **Wilson Score 95% Confidence Interval for Errors**:
          Quantifies uncertainty around error estimates, especially for small sample sizes like Spoofing ($N \approx 1,700$ vs DDoS $N \approx 1,000,000$).
        - **Macro-Average vs Weighted-Average**:
          $$\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^K F1_k, \quad \text{Weighted-F1} = \sum_{k=1}^K \left(\frac{N_k}{N}\right) F1_k$$
          Weighted average is dominated by majority classes and hides severe vulnerabilities in minority classes.
        """)

    st.markdown("---")

    rf_model, err_msg = load_rf_model()
    if rf_model is None:
        st.warning(f"⚠️ {err_msg}. You can continue exploring Tabs 1-5 without the model.")
        return

    st.success(f"✅ Loaded Random Forest classifier with `{len(rf_model.classes_)}` classes and `{len(rf_model.feature_names_in_)}` features.")

    # Section 1: Evaluation on Held-Out Sample
    st.subheader("1. Multiclass Evaluation & Confusion Matrix")
    
    eval_subsample_size = st.slider("Evaluation Test Sample Size:", min_value=1000, max_value=25000, value=5000, step=1000)
    
    # Stratified test sample
    test_sample = df.groupby("category", group_keys=False).apply(
        lambda x: x.sample(max(10, int(np.round(len(x) / len(df) * eval_subsample_size))), random_state=42)
    ).copy()

    # Align features exactly with model.feature_names_in_
    X_test = test_sample[rf_model.feature_names_in_].fillna(0)
    y_true = test_sample["category"].values

    with st.spinner("Generating predictions..."):
        y_pred = rf_model.predict(X_test)

    eval_res = classification_error_analysis(y_true, y_pred, classes=list(rf_model.classes_))

    c_m1, c_m2 = st.columns([1.2, 1])
    with c_m1:
        fig_cm = plot_confusion_matrix(eval_res["confusion_matrix"], list(rf_model.classes_))
        st.plotly_chart(fig_cm, use_container_width=True, key="tab6_confusion_matrix_chart")

    with c_m2:
        st.markdown("#### Macro vs Weighted Performance")
        macro_f1 = eval_res["macro_avg"].get("f1-score", 0.0)
        weighted_f1 = eval_res["weighted_avg"].get("f1-score", 0.0)
        
        st.metric("Macro-Average F1-Score", f"{macro_f1:.4f}")
        st.metric("Weighted-Average F1-Score", f"{weighted_f1:.4f}")
        
        st.markdown(f"""
        > [!IMPORTANT]
        > **Why Macro-Average is Essential in IoMT**:  
        > Notice how **Weighted F1 ({weighted_f1:.2f})** appears almost perfect, because massive DDoS/DoS attacks represent over 80% of traffic.  
        > However, **Macro F1 ({macro_f1:.2f})** weights all classes equally, correctly penalizing weakness on stealthy attacks like **Spoofing** (Recall ~80%).
        """)

    st.markdown("---")

    # Section 2: Statistical Error Analysis (Type I & Type II per class)
    st.subheader("2. Statistical Error Rates (Type I FPR & Type II FNR) with 95% Wilson CIs")
    st.dataframe(
        eval_res["per_class_df"].style.format({
            "Support": "{:,}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Type I Error (FPR)": "{:.4f}",
            "Type II Error (FNR)": "{:.4f}"
        }).highlight_max(subset=["Type II Error (FNR)"], color="#ffcdd2"),
        use_container_width=True
    )

    st.info("""
    🛡️ **Cybersecurity Interpretation**:  
    - Volumetric attacks (DDoS/DoS/MQTT) achieve near-zero Type II error rates ($FNR < 0.01$).
    - **Spoofing** exhibits the highest Type II error ($FNR \\approx 0.18-0.22$), meaning ~1 in 5 spoofed flows penetrates the detector! 
      This occurs because ARP spoofing uses normal packet rates and durations, blending in with benign device telemetry.
    """)

    st.markdown("---")

    # Section 3: Feature Importances
    st.subheader("3. Random Forest Feature Importances")
    if hasattr(rf_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": rf_model.feature_names_in_,
            "Gini Importance": rf_model.feature_importances_
        }).sort_values("Gini Importance", ascending=False).reset_index(drop=True)

        col_fi1, col_fi2 = st.columns([1.5, 1])
        with col_fi1:
            fig_fi = px.bar(
                fi_df.head(15),
                x="Gini Importance",
                y="Feature",
                orientation="h",
                color="Gini Importance",
                color_continuous_scale="Blues"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            apply_theme(fig_fi, "Top 15 Most Informative Flow Features", "Gini Importance Score", "Feature")
            st.plotly_chart(fig_fi, use_container_width=True, key="tab6_feature_importances_chart")

        with col_fi2:
            st.markdown("**Top Predictors:**")
            st.dataframe(fi_df.head(10).style.format({"Gini Importance": "{:.4f}"}), use_container_width=True)
            st.caption("Top features like `Max`, `Magnitue`, `Variance`, and `rst_count` capture flow burst size and TCP reset spikes during volumetric attacks.")

    st.markdown("---")

    # Section 4: Live Interactive Prediction Form & JSON Classifier
    st.subheader("4. Live Flow Classifier & API Integration")
    st.markdown("Test single-flow predictions using interactive parameter controls or raw JSON payloads (folded from demo.py).")

    input_mode = st.radio("Input Mode:", options=["Interactive Sliders (Key Features)", "Raw JSON Flow (demo.py format)"], horizontal=True)

    # Compute baseline medians for features
    medians = df[rf_model.feature_names_in_].median().to_dict()

    if "Interactive Sliders" in input_mode:
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            val_duration = st.slider("Duration (seconds):", min_value=0.0, max_value=128.0, value=float(medians.get("Duration", 64.0)))
            val_rate = st.slider("Rate (packets/s):", min_value=0.0, max_value=50000.0, value=float(medians.get("Rate", 212.0)))
            val_header = st.slider("Header_Length:", min_value=0.0, max_value=10000.0, value=float(medians.get("Header_Length", 3600.0)))
        with c_i2:
            val_syn = st.selectbox("syn_flag_number:", options=[0.0, 1.0], index=int(medians.get("syn_flag_number", 0.0)))
            val_rst_cnt = st.slider("rst_count:", min_value=0.0, max_value=50.0, value=float(medians.get("rst_count", 0.0)))
            val_proto = st.selectbox("Protocol Type:", options=[6.0, 17.0, 1.0, 0.0], format_func=lambda x: f"{x} (TCP)" if x==6 else f"{x} (UDP)" if x==17 else f"{x} (ICMP)" if x==1 else str(x))
        with c_i3:
            val_max = st.slider("Max Packet Size:", min_value=0.0, max_value=2000.0, value=float(medians.get("Max", 1200.0)))
            val_var = st.slider("Variance:", min_value=0.0, max_value=5.0, value=float(medians.get("Variance", 0.9)))
            val_iat = st.slider("IAT (Inter-arrival time):", min_value=0.0, max_value=2e8, value=float(medians.get("IAT", 1.6e8)))

        # Update input payload
        live_payload = medians.copy()
        live_payload["Duration"] = val_duration
        live_payload["Rate"] = val_rate
        live_payload["Header_Length"] = val_header
        live_payload["syn_flag_number"] = val_syn
        live_payload["rst_count"] = val_rst_cnt
        live_payload["Protocol Type"] = val_proto
        live_payload["Max"] = val_max
        live_payload["Variance"] = val_var
        live_payload["IAT"] = val_iat

    else:
        sample_json_text = json.dumps({
            "Header_Length": 3609.0, "Protocol Type": 17.0, "Duration": 64.0, "Rate": 212.96,
            "Srate": 212.96, "Drate": 0.0, "fin_flag_number": 0.0, "syn_flag_number": 0.0,
            "rst_flag_number": 0.0, "psh_flag_number": 0.0, "ack_flag_number": 0.0,
            "ece_flag_number": 0.0, "cwr_flag_number": 0.0, "ack_count": 0.0, "syn_count": 0.0,
            "fin_count": 0.0, "rst_count": 0.0, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0,
            "Telnet": 0.0, "SMTP": 0.0, "SSH": 0.0, "IRC": 0.0, "TCP": 0.0, "UDP": 1.0,
            "DHCP": 0.0, "ARP": 0.0, "ICMP": 0.0, "IGMP": 0.0, "IPv": 1.0, "LLC": 1.0,
            "Tot sum": 3609.0, "Min": 154.8, "Max": 1228.4, "AVG": 610.0, "Std": 487.4,
            "Tot size": 673.2, "IAT": 169402747.0, "Number": 5.5, "Magnitue": 34.69,
            "Radius": 689.3, "Covariance": 279086.0, "Variance": 0.9, "Weight": 38.5
        }, indent=2)

        user_json = st.text_area("Paste Flow JSON:", value=sample_json_text, height=220)
        try:
            parsed = json.loads(user_json)
            live_payload = medians.copy()
            live_payload.update(parsed)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            live_payload = medians.copy()

    col_btn1, col_btn2 = st.columns([1, 1.5])
    with col_btn1:
        predict_local = st.button("🚀 Classify Flow Locally (Loaded Model)", type="primary")
    with col_btn2:
        call_api = st.checkbox("Also Query FastAPI Endpoint (http://127.0.0.1:8000/predict)", value=False)

    if predict_local:
        input_row = pd.DataFrame([live_payload])[rf_model.feature_names_in_].fillna(0)
        pred_label = rf_model.predict(input_row)[0]
        proba = rf_model.predict_proba(input_row)[0]
        confidence_dict = dict(zip(rf_model.classes_, proba))

        pred_color = "#2ECC71" if pred_label == "Benign" else "#E74C3C"
        st.markdown(f"""
        <div style="background-color: {pred_color}22; border-left: 6px solid {pred_color}; padding: 15px; border-radius: 6px; margin: 15px 0;">
            <h3 style="margin: 0; color: {pred_color};">Predicted Classification: <strong>{pred_label}</strong></h3>
            <p style="margin: 5px 0 0 0;">Model Confidence: <strong>{confidence_dict[pred_label]*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Plot confidence probability bar
        conf_df = pd.DataFrame(list(confidence_dict.items()), columns=["Category", "Probability"])
        fig_conf = px.bar(
            conf_df,
            x="Probability",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_map={"Benign": "#2ECC71", "DDoS": "#E74C3C", "DoS": "#E67E22", "MQTT": "#9B59B6", "Recon": "#F1C40F", "Spoofing": "#1ABC9C"},
            range_x=[0, 1]
        )
        apply_theme(fig_conf, "Class Posterior Probabilities", "Posterior Probability", "Class")
        st.plotly_chart(fig_conf, use_container_width=True, key="tab6_live_confidence_chart")

        # Optional FastAPI request
        if call_api:
            try:
                # Format json matching alias schema
                api_resp = requests.post("http://127.0.0.1:8000/predict", json=live_payload, timeout=2.0)
                if api_resp.status_code == 200:
                    st.success("FastAPI /predict Response: " + str(api_resp.json()))
                else:
                    st.warning(f"FastAPI responded with status code {api_resp.status_code}: {api_resp.text}")
            except Exception as ex:
                st.info(f"FastAPI endpoint not reachable (Run `uvicorn main:app` to start it): {ex}")
