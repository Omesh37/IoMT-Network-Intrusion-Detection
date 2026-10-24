"""
Home Page: Project Overview for the IoMT Statistics Dashboard.
Presents the project background, key dataset metrics, and class distribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.stats_utils import calc_class_frequency_table
from utils.plot_utils import plot_frequency_bar


def render_home_page(df: pd.DataFrame, is_synthetic: bool, source_info: str):
    """Renders a simple, clean project overview."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%); padding: 22px; border-radius: 10px; margin-bottom: 20px; color: #0d231b;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">Securing Medical IoT: A Statistical and Machine Learning Approach</h1>
        <p style="margin-top: 8px; font-size: 1.05rem; opacity: 0.95;">
            An interactive dashboard applying introductory <strong>Probability and Statistics</strong> to real-world 
            Internet of Medical Things (IoMT) network traffic from the Canadian Institute for Cybersecurity (CICIoMT2024).
        </p>
    </div>
    """, unsafe_allow_html=True)

    if is_synthetic:
        st.warning("""
        ⚠️ **Notice: Synthetic Demo Dataset in Active Use**  
        The real CICIoMT2024 dataset was not detected in local standard paths. A demo synthetic dataset with the same columns 
        is currently active. You can specify a file path or upload your dataset in the sidebar.
        """)

    # Quick Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flows in Sample", f"{len(df):,}")
    with col2:
        st.metric("Flow Features", "45")
    with col3:
        st.metric("Traffic Classes", f"{df['category'].nunique()}")
    with col4:
        nan_count = df.select_dtypes(include=[np.number]).isna().sum().sum()
        st.metric("Data Quality", "Clean (0 NaNs)" if nan_count == 0 else f"{nan_count} NaNs")

    st.markdown("---")

    # Class Breakdown Table & Chart
    st.subheader("📊 Class Distribution")
    st.caption("Distribution of Benign medical device telemetry versus attack families in the current sample.")

    freq_df = calc_class_frequency_table(df["category"])
    c_left, c_right = st.columns([1, 1.2])
    with c_left:
        st.dataframe(
            freq_df.style.format({
                "Count": "{:,}",
                "Relative Frequency (%)": "{:.2f}%",
                "Cumulative Frequency (%)": "{:.2f}%"
            }),
            use_container_width=True
        )
        benign_cnt = freq_df[freq_df["Class"] == "Benign"]["Count"].sum()
        attack_cnt = len(df) - benign_cnt
        st.info(f"**Sample Count**: {benign_cnt:,} Benign flows and {attack_cnt:,} Attack flows.")

    with c_right:
        fig = plot_frequency_bar(freq_df)
        st.plotly_chart(fig, use_container_width=True, key="home_class_freq_chart")

    st.markdown("---")

    # Simple Guide to Dashboard Tabs
    st.subheader("🧭 Dashboard Guide")
    st.markdown("""
    Use the tabs above to explore standard probability and statistics concepts:
    - **Unit 1: Descriptive Stats**: Data types, mean, median, mode, standard deviation, histograms, and box plots.
    - **Unit 2: Probability & Distributions**: Empirical probabilities, addition rule, Bayes' theorem, and standard distributions (Binomial, Poisson, Exponential).
    - **Unit 3: Normal & Sampling**: Normal distribution, Z-scores, and the Central Limit Theorem (CLT).
    - **Unit 4: Inference & ANOVA**: Confidence intervals, hypothesis testing (Z-test, t-test), Chi-Square tests, and One-Way ANOVA.
    - **Unit 5: Correlation & Regression**: Scatter plot, correlation coefficient $r$, and linear regression line.
    - **Unit 6: Model & Error Analysis**: Evaluation of the Random Forest model and live flow prediction.
    """)
