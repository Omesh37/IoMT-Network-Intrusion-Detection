"""
Tab 5: Correlation & Simple Linear Regression (Unit 5)
Provides a straightforward, easy-to-understand demonstration of linear correlation
and simple linear regression (y = mx + c) with a single clean scatter plot and trendline.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

from utils.plot_utils import apply_theme


def render_tab5(df: pd.DataFrame, full_df: pd.DataFrame, primary_feature: str):
    st.header("Unit 5: Correlation & Simple Linear Regression")

    # Simple concept expander with introductory syllabus concepts
    with st.expander("📘 Basic Principles: Correlation & Linear Regression (Unit 5)", expanded=False):
        st.markdown(r"""
        ### Fundamental Concepts
        - **Correlation Coefficient ($r$)**:
          Measures the direction and strength of the linear relationship between two continuous variables.
          Values range from $-1.0$ (perfect negative correlation) to $+1.0$ (perfect positive correlation), with $0$ indicating no linear relationship.
        - **Simple Linear Regression Line**:
          Fits a straight line through the data to model how a response variable ($Y$) changes with a predictor ($X$):
          $$\hat{y} = b_0 + b_1 x$$
          where $b_1$ is the slope and $b_0$ is the intercept.
        - **Coefficient of Determination ($R^2$)**:
          The square of the correlation coefficient ($r^2$), indicating the percentage of variation in $Y$ that is explained by $X$.
        """)

    st.markdown("---")

    st.subheader("Simple Linear Regression Explorer")
    st.markdown("Select two continuous network telemetry features to explore their linear relationship.")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].std() > 0]
    
    # Sensible default choices
    default_x = "Rate" if "Rate" in numeric_cols else numeric_cols[0]
    default_y = "Tot sum" if "Tot sum" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])

    col_x, col_y = st.columns(2)
    with col_x:
        x_col = st.selectbox(
            "Predictor Variable (X):",
            options=numeric_cols,
            index=numeric_cols.index(default_x) if default_x in numeric_cols else 0,
            key="tab5_x_var_select"
        )
    with col_y:
        y_col = st.selectbox(
            "Response Variable (Y):",
            options=numeric_cols,
            index=numeric_cols.index(default_y) if default_y in numeric_cols else min(1, len(numeric_cols) - 1),
            key="tab5_y_var_select"
        )

    # Clean valid finite data
    valid_mask = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    sub_df = df.loc[valid_mask, [x_col, y_col, "category"]]

    if len(sub_df) < 5 or sub_df[x_col].std() == 0 or sub_df[y_col].std() == 0:
        st.warning("Insufficient variance or valid data points to perform linear regression on the chosen variables.")
        return

    # Subsample for responsive, crisp plotting if dataset is large
    sample_size = min(len(sub_df), 1500)
    plot_df = sub_df.sample(sample_size, random_state=42)

    x_vals = sub_df[x_col].values
    y_vals = sub_df[y_col].values

    # Compute regression parameters
    reg = stats.linregress(x_vals, y_vals)
    slope = reg.slope
    intercept = reg.intercept
    r_val = reg.rvalue
    r_squared = r_val ** 2
    p_val = reg.pvalue

    # Display KPI cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Correlation (r)", f"{r_val:.4f}")
    col_m2.metric("R² (Variance Explained)", f"{r_squared * 100:.2f}%")
    col_m3.metric("Slope (b₁)", f"{slope:.4f}")
    col_m4.metric("Intercept (b₀)", f"{intercept:.4f}")

    # Fitted equation display
    sign = "+" if intercept >= 0 else "-"
    st.info(f"**Fitted Regression Equation**: $\\hat{{{y_col}}} = {slope:.4f} \\cdot {x_col} {sign} {abs(intercept):.4f}$  (p-value = `{p_val:.4e}`)")

    # Only one clean static graph: Scatter Plot with Regression Line
    fig_scatter = go.Figure()

    # Data points
    fig_scatter.add_trace(go.Scatter(
        x=plot_df[x_col],
        y=plot_df[y_col],
        mode="markers",
        name="Telemetry Flows",
        marker=dict(
            color="#3498DB",
            size=6,
            opacity=0.55,
            line=dict(width=0.5, color="#1B4F72")
        )
    ))

    # Regression line
    x_line = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
    y_line = slope * x_line + intercept

    fig_scatter.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Fitted Line (ŷ)",
        line=dict(color="#E74C3C", width=3)
    ))

    apply_theme(
        fig_scatter,
        f"Scatter Plot & Fitted Regression Line: {y_col} vs {x_col}",
        x_col,
        y_col
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="tab5_single_scatter_chart")

    # Plain-English security interpretation
    corr_desc = "strong positive" if r_val > 0.7 else ("moderate positive" if r_val > 0.3 else ("weak/no" if abs(r_val) <= 0.3 else ("moderate negative" if r_val > -0.7 else "strong negative")))
    st.success(f"""
    🛡️ **What this means for IoMT Security**:  
    The correlation between `{x_col}` and `{y_col}` is **{r_val:.3f}** ({corr_desc} linear relationship).  
    When network attacks like Flooding (DDoS/DoS) occur, packet transmission rates surge rapidly, driving a proportional increase in aggregate byte volumes and buffer sizes. 
    A higher $R^2$ value means `{x_col}` can reliably serve as a predictive indicator for monitoring surges in `{y_col}` across medical IoT devices.
    """)
