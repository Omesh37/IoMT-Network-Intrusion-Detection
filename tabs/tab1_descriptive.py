"""
Tab 1: Descriptive Statistics & Data Types (Unit 1)
Interactive exploration of IoMT feature scales, central tendency, dispersion,
shape, frequency distributions, and outlier detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.stats_utils import (
    calc_descriptive_stats,
    calc_class_frequency_table,
    classify_features
)
from utils.plot_utils import (
    plot_frequency_bar,
    plot_feature_histogram,
    plot_box_by_class
)


def render_tab1(df: pd.DataFrame, full_df: pd.DataFrame, primary_feature: str, log_transform: bool):
    st.header("Unit 1: Descriptive Statistics & Exploratory Data Analysis")

    # Mathematical Concept Expander
    with st.expander("📘 Mathematical Concepts & Syllabus Definitions (Unit 1)", expanded=False):
        st.markdown(r"""
        ### Fundamental Descriptive Measures
        - **Sample Mean ($\bar{x}$)**: Centroid of values, sensitive to extreme outliers:
          $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$
        - **Sample Variance ($s^2$) with Bessel's Correction**: Unbiased estimator of population variance:
          $$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2$$
        - **Standard Deviation ($s$)**: Square root of variance in native physical units ($s = \sqrt{s^2}$).
        - **Coefficient of Variation ($CV$)**: Dimensionless relative dispersion metric:
          $$CV = \frac{s}{|\bar{x}|} \times 100\%$$
        - **Fisher-Pearson Standardized Skewness**: Measure of distribution asymmetry:
          $$\text{Skewness} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{\left[\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2\right]^{3/2}}$$
        - **Fisher Excess Kurtosis**: Tail weight relative to normal distribution ($\text{Normal} = 0$):
          $$\text{Kurtosis} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^4}{s^4} - 3$$
        - **Tukey Interquartile Range (IQR) & Outlier Fences**:
          $$IQR = Q_3 - Q_1, \quad \text{Outlier Fences} = [Q_1 - 1.5 \cdot IQR, \; Q_3 + 1.5 \cdot IQR]$$
        """)

    st.markdown("---")

    # Section 1: Variable Scales & Types Table
    st.subheader("1. Variable Classification & Scales of Measurement")
    st.caption("Comprehensive audit of all 45 telemetry features plus target labels.")

    all_cols = [c for c in df.columns if c not in ["label", "canonical_label", "binary_label"]]
    types_df = classify_features(all_cols)

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        type_filter = st.multiselect(
            "Filter by Variable Type:",
            options=types_df["Variable Type"].unique(),
            default=types_df["Variable Type"].unique()
        )
    with col_t2:
        search_kw = st.text_input("Search Feature Name:", placeholder="e.g. Protocol, Rate, syn...")

    filtered_types = types_df[types_df["Variable Type"].isin(type_filter)]
    if search_kw:
        filtered_types = filtered_types[filtered_types["Feature Name"].str.contains(search_kw, case=False)]

    st.dataframe(
        filtered_types,
        column_config={
            "Feature Name": st.column_config.TextColumn("Feature", width="medium"),
            "Variable Type": st.column_config.TextColumn("Type", width="small"),
            "Sub-classification": st.column_config.TextColumn("Sub-type", width="medium"),
            "Measurement Scale": st.column_config.TextColumn("Scale", width="medium"),
            "IoMT Security Role": st.column_config.TextColumn("Domain Context", width="large"),
        },
        use_container_width=True,
        hide_index=True
    )

    st.info(r"""
    💡 **IoMT Security Note on Measurement Scales**:
    - **Protocol Type** is numerically recorded (e.g. 6, 17, 1) but represents a **Qualitative Nominal** variable (IANA protocol numbers). Treating it as continuous ratio data in regression yields meaningless equations!
    - **Header flags** (e.g., `syn_flag_number`) are **Bernoulli indicators** ($X \in \{0, 1\}$) whose mean $\bar{x} = p$ reflects empirical probability of the flag's presence.
    """)

    st.markdown("---")

    # Section 2: Frequency Distribution of Target Classes
    st.subheader("2. Class Frequency Distribution")
    freq_df = calc_class_frequency_table(df["category"])
    
    c_f1, c_f2 = st.columns([1, 1.4])
    with c_f1:
        st.dataframe(
            freq_df.style.format({
                "Count": "{:,}",
                "Relative Frequency (%)": "{:.2f}%",
                "Cumulative Frequency (%)": "{:.2f}%"
            }),
            use_container_width=True
        )
    with c_f2:
        fig_freq = plot_frequency_bar(freq_df)
        st.plotly_chart(fig_freq, use_container_width=True, key="tab1_class_freq_chart")

    st.markdown("---")

    # Section 3: Distribution Plots (Histogram, Box, Violin)
    st.subheader(f"3. Distribution Visualizer: {primary_feature}")
    
    # Feature data preparation
    feature_data = df[primary_feature].dropna()
    feature_data = feature_data[np.isfinite(feature_data)]
    if log_transform and (feature_data >= 0).all():
        plot_data = np.log1p(feature_data)
        plot_label = f"log1p({primary_feature})"
    else:
        plot_data = feature_data
        plot_label = primary_feature

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        bins = st.slider("Histogram Bin Count:", min_value=10, max_value=120, value=40, step=5)
    with col_ctrl2:
        show_violin = st.checkbox("Show Violin Plot alongside Box Plot", value=False)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_hist = plot_feature_histogram(plot_data, plot_label, bins=bins, show_kde=True)
        st.plotly_chart(fig_hist, use_container_width=True, key="tab1_feature_hist_chart")
    with col_p2:
        plot_df = df.copy()
        if log_transform and (feature_data >= 0).all():
            plot_df[plot_label] = np.log1p(plot_df[primary_feature])
        fig_box = plot_box_by_class(plot_df, plot_label, class_col="category", show_violin=show_violin)
        st.plotly_chart(fig_box, use_container_width=True, key="tab1_box_violin_chart")

    st.markdown("---")

    # Section 4: Comprehensive Summary Statistics Table
    st.subheader(f"4. Summary Statistics for {primary_feature} (Overall & Per Class)")
    
    # Overall statistics
    overall_stats = calc_descriptive_stats(feature_data)
    
    # Per-class statistics
    per_class_list = []
    for cls_name, grp in df.groupby("category"):
        s = grp[primary_feature].dropna()
        st_dict = calc_descriptive_stats(s)
        st_dict["Class"] = cls_name
        per_class_list.append(st_dict)

    overall_row = overall_stats.copy()
    overall_row["Class"] = "ALL (Overall)"
    summary_df = pd.DataFrame([overall_row] + per_class_list)
    
    # Reorder columns
    display_cols = ["Class", "count", "mean", "median", "mode", "variance", "std", "cv", "skewness", "kurtosis", "iqr", "outliers_count", "outlier_percentage"]
    summary_df = summary_df[display_cols]
    summary_df.columns = ["Class", "Count", "Mean", "Median", "Mode", "Variance", "Std Dev", "CV (%)", "Skewness", "Kurtosis", "IQR", "Outliers", "Outlier %"]

    st.dataframe(
        summary_df.style.format({
            "Count": "{:,}",
            "Mean": "{:.4f}",
            "Median": "{:.4f}",
            "Mode": "{:.4f}",
            "Variance": "{:.4f}",
            "Std Dev": "{:.4f}",
            "CV (%)": "{:.2f}%",
            "Skewness": "{:.3f}",
            "Kurtosis": "{:.3f}",
            "IQR": "{:.4f}",
            "Outliers": "{:,}",
            "Outlier %": "{:.2f}%"
        }).highlight_max(subset=["CV (%)"], color="#ffcdd2"),
        use_container_width=True
    )

    # Highlight feature with highest CV across entire dataset
    num_cols = df.select_dtypes(include=[np.number]).columns
    cv_records = {}
    for col in num_cols:
        s = df[col].dropna()
        m = s.mean()
        if abs(m) > 1e-5:
            cv_records[col] = (s.std(ddof=1) / abs(m)) * 100.0
            
    if cv_records:
        highest_cv_feat = max(cv_records, key=cv_records.get)
        highest_cv_val = cv_records[highest_cv_feat]
        st.metric(
            label="Highest Coefficient of Variation (CV) Across All Features",
            value=f"{highest_cv_feat}: {highest_cv_val:.1f}%",
            help="High CV indicates extreme relative dispersion and burstiness, typical of attack surge traffic."
        )

    # Automated Dynamic Interpretation
    st.subheader("5. Automated Statistical Interpretation")
    skew_val = overall_stats["skewness"]
    mean_val = overall_stats["mean"]
    median_val = overall_stats["median"]
    
    if skew_val > 1.0:
        skew_desc = f"**heavily right-skewed** ($\text{{Skewness}} = {skew_val:.2f} > 1$, Mean {mean_val:.2f} > Median {median_val:.2f}). Extreme bursts of traffic create a heavy right tail."
    elif skew_val < -1.0:
        skew_desc = f"**heavily left-skewed** ($\text{{Skewness}} = {skew_val:.2f} < -1$, Mean {mean_val:.2f} < Median {median_val:.2f})."
    else:
        skew_desc = f"**approximately symmetric** ($\text{{Skewness}} = {skew_val:.2f} \approx 0$)."

    st.markdown(f"""
    > [!NOTE]
    > **Data Shape Insight**: The feature **`{primary_feature}`** is {skew_desc}
    > Across classes, the Coefficient of Variation ranges from {summary_df['CV (%)'].min():.1f}% to {summary_df['CV (%)'].max():.1f}%.
    > 
    > **What this means for IoMT Security**:  
    > In medical IoT environments, benign telemetry from devices like infusion pumps or heart-rate monitors is highly rhythmic and predictable (low CV, modest IQR). 
    > Volumetric attacks (DDoS/DoS) inject enormous bursts of packets, generating huge outlier counts ({overall_stats['outliers_count']:,} flows flagged by 1.5×IQR fences).
    > However, stealth attacks like **Spoofing** mimic benign traffic rates, demanding statistical hypothesis testing and multivariate modeling to unmask.
    """)
