"""
Main Entry Point for the IoMT Network Intrusion Detection Statistics Dashboard.
Orchestrates global sidebar controls, cached data loading, stratified sampling,
and renders the university syllabus unit modules in interactive tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="IoMT Statistics & ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.data_loader import (
    FEATURE_NAMES,
    resolve_default_path,
    load_raw_dataset,
    get_stratified_sample
)
from home import render_home_page
from tabs.tab1_descriptive import render_tab1
from tabs.tab2_probability import render_tab2
from tabs.tab3_normal_sampling import render_tab3
from tabs.tab4_inference_tests import render_tab4
from tabs.tab5_correlation_regression import render_tab5
from tabs.tab6_model_errors import render_tab6


def main():
    # ---------------------------------------------------------
    # GLOBAL SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.title("🛡️ IoMT Analytics Controls")
    st.sidebar.caption("University Probability & Statistics Testbed")

    # 1. Data Source Selection
    st.sidebar.subheader("1. Data Source")
    default_detected_path = resolve_default_path() or ""
    
    source_type = st.sidebar.radio(
        "Load Method:",
        options=["Disk Path / Auto-Detect", "Upload Parquet / CSV File"],
        label_visibility="collapsed"
    )

    file_or_path = None
    if source_type == "Disk Path / Auto-Detect":
        path_input = st.sidebar.text_input(
            "Parquet/CSV File Path:",
            value=default_detected_path,
            placeholder="e.g. test_enriched.parquet or ../test_enriched.parquet",
            help="Path to real CICIoMT2024 dataset. Leave blank to auto-detect or fall back to demo synthetic data."
        )
        file_or_path = path_input.strip() if path_input.strip() else None
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Dataset File:",
            type=["parquet", "csv"],
            help="Upload custom CICIoMT2024 parquet or CSV file."
        )
        file_or_path = uploaded_file

    # Load dataset with caching
    full_df, is_synthetic, source_status = load_raw_dataset(file_or_path)

    if is_synthetic:
        st.sidebar.warning("⚠️ Using Synthetic Demo Data")
    else:
        st.sidebar.success(f"✅ {source_status}")

    st.sidebar.markdown("---")

    # 2. Sample Size & Stratification
    st.sidebar.subheader("2. Sample Size & Sampling")
    total_rows = len(full_df)
    
    use_full = st.sidebar.checkbox(
        "Load Full Dataset",
        value=False,
        help="Warning: Disables sampling. May consume significant memory if analyzing all 1.6M+ flows."
    )
    
    if use_full:
        st.sidebar.warning(f"⚠️ Full dataset active ({total_rows:,} rows).")
        df = full_df.copy()
    else:
        default_sample = min(100000, total_rows)
        sample_size = st.sidebar.slider(
            "Sample Size (Stratified):",
            min_value=min(1000, total_rows),
            max_value=total_rows,
            value=default_sample,
            step=max(1000, int(total_rows / 100)),
            help="Stratified sampling guarantees rare classes (e.g. Spoofing) are preserved proportionally."
        )
        random_seed = st.sidebar.number_input("Sampling Random Seed:", value=42, step=1)
        df = get_stratified_sample(full_df, target_col="category", sample_size=sample_size, seed=int(random_seed))

    st.sidebar.caption(f"Active Sample: **{len(df):,}** flows (of {total_rows:,} available)")

    st.sidebar.markdown("---")

    # 3. Class Filter
    st.sidebar.subheader("3. Class Filter")
    all_categories = sorted(df["category"].unique())
    selected_classes = st.sidebar.multiselect(
        "Include Traffic Classes:",
        options=all_categories,
        default=all_categories,
        help="Filter dashboard metrics to specific traffic families."
    )
    if not selected_classes:
        st.sidebar.warning("Please select at least one class.")
        selected_classes = all_categories

    filtered_df = df[df["category"].isin(selected_classes)].copy()

    st.sidebar.markdown("---")

    # 4. Feature Selector & Transformation
    st.sidebar.subheader("4. Feature & Transformations")
    numeric_features = [col for col in FEATURE_NAMES if col in filtered_df.columns]
    
    primary_feature = st.sidebar.selectbox(
        "Primary Feature for Deep Dive:",
        options=numeric_features,
        index=numeric_features.index("Rate") if "Rate" in numeric_features else 0,
        help="Select numeric feature to analyze across histograms, normality tests, and control charts."
    )

    log_transform = st.sidebar.toggle(
        "Apply log1p Transform (for Skewed Features)",
        value=False,
        help="Applies y = log(1 + x) to stabilize extreme right skewness in features like Rate, Duration, IAT."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("IoMT Intrusion Detection System • Course Probability & Statistics")

    # ---------------------------------------------------------
    # MAIN APPLICATION NAVIGATION & TABS
    # ---------------------------------------------------------
    dashboard_tabs = st.tabs([
        "🏠 Project Overview",
        "📊 Unit 1: Descriptive Stats",
        "🎲 Unit 2: Probability & Distributions",
        "📈 Unit 3: Normal & Sampling",
        "🔬 Unit 4: Inference & ANOVA",
        "🔗 Unit 5: Correlation & Regression",
        "🤖 Unit 6: Model & Error Analysis"
    ])

    # Tab 0: Home / Overview
    with dashboard_tabs[0]:
        render_home_page(filtered_df, is_synthetic, source_status)

    # Tab 1: Descriptive Statistics (Unit 1)
    with dashboard_tabs[1]:
        render_tab1(filtered_df, full_df, primary_feature, log_transform)

    # Tab 2: Probability & Distributions (Unit 2)
    with dashboard_tabs[2]:
        render_tab2(filtered_df, full_df, primary_feature)

    # Tab 3: Normal & Sampling Theory (Unit 3)
    with dashboard_tabs[3]:
        render_tab3(filtered_df, full_df, primary_feature, log_transform)

    # Tab 4: Estimation, Hypothesis Testing & ANOVA (Unit 4)
    with dashboard_tabs[4]:
        render_tab4(filtered_df, full_df, primary_feature)

    # Tab 5: Correlation & Regression (Unit 5)
    with dashboard_tabs[5]:
        render_tab5(filtered_df, full_df, primary_feature)

    # Tab 6: Machine Learning Model & Statistical Errors (Unit 6)
    with dashboard_tabs[6]:
        render_tab6(filtered_df, full_df)


if __name__ == "__main__":
    main()
