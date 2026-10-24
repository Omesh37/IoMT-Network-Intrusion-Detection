"""
Plotting utilities for the IoMT Network Intrusion Detection Statistics Dashboard.
Provides consistent, responsive Plotly figures with dark/light theme compatibility.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Distinct domain-aligned colors for IoMT classes
CLASS_COLORS = {
    "Benign": "#2ECC71",    # Emerald green
    "DDoS": "#E74C3C",      # Red
    "DoS": "#E67E22",       # Orange
    "MQTT": "#9B59B6",      # Purple
    "Recon": "#F1C40F",     # Yellow-gold
    "Spoofing": "#1ABC9C"   # Turquoise
}

THEME_LAYOUT = {
    "template": "plotly_white",
    "margin": dict(l=40, r=40, t=50, b=40),
    "font": dict(family="Inter, -apple-system, sans-serif", size=12),
    "hoverlabel": dict(font_size=12, font_family="Inter, sans-serif")
}


def apply_theme(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    """Applies standardized styling and layout parameters to a Plotly figure."""
    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text=title, font=dict(size=15, weight=600)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_frequency_bar(freq_df: pd.DataFrame) -> go.Figure:
    """Creates a class frequency bar chart with count and percentage annotations."""
    colors = [CLASS_COLORS.get(cls, "#3498DB") for cls in freq_df["Class"]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=freq_df["Class"],
            y=freq_df["Count"],
            text=[f"{pct:.2f}%" for pct in freq_df["Relative Frequency (%)"]],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#2C3E50", width=1))
        )
    ])
    return apply_theme(fig, "Class Frequency Distribution (CICIoMT2024)", "Network Traffic Class", "Flow Count")


def plot_feature_histogram(series: pd.Series, feature_name: str, bins: int = 40, show_kde: bool = True) -> go.Figure:
    """Plots feature histogram with optional normal distribution overlay."""
    clean_s = series.dropna()
    clean_s = clean_s[np.isfinite(clean_s)]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clean_s,
        nbinsx=bins,
        histnorm="probability density",
        name="Empirical Density",
        marker=dict(color="#3498DB", opacity=0.7, line=dict(color="#2980B9", width=1))
    ))
    
    if show_kde and len(clean_s) > 5:
        mu = float(np.mean(clean_s))
        sigma = float(np.std(clean_s, ddof=1))
        if sigma > 0:
            x_range = np.linspace(float(clean_s.min()), float(clean_s.max()), 200)
            y_normal = stats.norm.pdf(x_range, mu, sigma)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_normal,
                mode="lines",
                name=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})",
                line=dict(color="#E74C3C", width=2.5, dash="dash")
            ))
            
    return apply_theme(fig, f"Distribution of {feature_name}", feature_name, "Probability Density")


def plot_box_by_class(df: pd.DataFrame, feature_name: str, class_col: str = "category", show_violin: bool = False) -> go.Figure:
    """Plots box plots or violin plots grouped by traffic class."""
    clean_df = df[[class_col, feature_name]].dropna()
    clean_df = clean_df[np.isfinite(clean_df[feature_name])]
    
    if show_violin:
        fig = px.violin(
            clean_df,
            x=class_col,
            y=feature_name,
            color=class_col,
            color_discrete_map=CLASS_COLORS,
            box=True,
            points=False
        )
        title = f"Violin Distribution of {feature_name} by Class"
    else:
        fig = px.box(
            clean_df,
            x=class_col,
            y=feature_name,
            color=class_col,
            color_discrete_map=CLASS_COLORS,
            points="outliers"
        )
        title = f"Box Plot of {feature_name} by Class (Tukey 1.5×IQR Fences)"
        
    return apply_theme(fig, title, "Traffic Category", feature_name)


def plot_venn_breakdown(p_a: float, p_b: float, p_a_and_b: float, label_a: str, label_b: str) -> go.Figure:
    """Renders an intuitive partitioned bar diagram visualizing addition rule P(A or B)."""
    p_only_a = max(0.0, p_a - p_a_and_b)
    p_only_b = max(0.0, p_b - p_a_and_b)
    p_both = p_a_and_b
    p_neither = max(0.0, 1.0 - (p_only_a + p_only_b + p_both))
    
    categories = [f"Only {label_a}", "Intersection (A ∩ B)", f"Only {label_b}", "Neither (Complement)"]
    probs = [p_only_a, p_both, p_only_b, p_neither]
    colors = ["#3498DB", "#9B59B6", "#E67E22", "#BDC3C7"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probs,
            text=[f"{p*100:.2f}%" for p in probs],
            textposition="auto",
            marker=dict(color=colors)
        )
    ])
    return apply_theme(fig, f"Empirical Probability Breakdown: {label_a} vs {label_b}", "Event Region", "Probability P(E)")


def plot_qq_norm(data: np.ndarray, feature_name: str) -> go.Figure:
    """Generates a Quantile-Quantile (Q-Q) plot against theoretical standard normal quantiles."""
    clean_data = data[np.isfinite(data)]
    n = min(len(clean_data), 3000)
    sub = np.random.choice(clean_data, size=n, replace=False) if len(clean_data) > n else clean_data
    
    (osm, osr), (slope, intercept, r) = stats.probplot(sub, dist="norm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode="markers",
        name="Empirical Quantiles",
        marker=dict(color="#3498DB", size=4, opacity=0.7)
    ))
    
    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"Theoretical Reference (R²={r**2:.3f})",
        line=dict(color="#E74C3C", width=2.5)
    ))
    
    return apply_theme(fig, f"Normal Q-Q Plot: {feature_name}", "Theoretical Quantiles", "Empirical Quantiles")


def plot_control_chart(series: pd.Series, labels: pd.Series, cl: float, ucl: float, lcl: float, feature_name: str) -> go.Figure:
    """Plots a Shewhart-style X-bar Quality Control chart with upper and lower 3-sigma limits."""
    n = min(len(series), 1000)
    sub_s = series.iloc[:n].values
    sub_l = labels.iloc[:n].values
    idx = np.arange(1, n + 1)
    
    out_of_control = (sub_s > ucl) | (sub_s < lcl)
    in_control = ~out_of_control
    
    fig = go.Figure()
    # In-control line
    fig.add_trace(go.Scatter(
        x=idx,
        y=sub_s,
        mode="lines",
        name="Observed Flow Metric",
        line=dict(color="#7F8C8D", width=1.2)
    ))
    # In-control points
    fig.add_trace(go.Scatter(
        x=idx[in_control],
        y=sub_s[in_control],
        mode="markers",
        name="Normal Traffic",
        marker=dict(color="#2ECC71", size=4, opacity=0.6)
    ))
    # Out of control anomalies
    fig.add_trace(go.Scatter(
        x=idx[out_of_control],
        y=sub_s[out_of_control],
        mode="markers",
        name="Out-of-Control (Anomaly)",
        marker=dict(color="#E74C3C", size=7, symbol="x"),
        text=[f"Class: {lbl}" for lbl in sub_l[out_of_control]]
    ))
    
    # Control limit lines
    fig.add_hline(y=ucl, line=dict(color="#C0392B", dash="dash", width=2), annotation_text=f"UCL (+3σ) = {ucl:.2f}")
    fig.add_hline(y=cl, line=dict(color="#2980B9", dash="dot", width=1.8), annotation_text=f"Center Line (CL) = {cl:.2f}")
    fig.add_hline(y=lcl, line=dict(color="#C0392B", dash="dash", width=2), annotation_text=f"LCL (-3σ) = {lcl:.2f}")
    
    return apply_theme(fig, f"Statistical Process Control Chart: {feature_name}", "Flow Sequence Index", feature_name)


def plot_rejection_regions(crit_val: float, test_stat: float, alternative: str = "two-sided", alpha: float = 0.05) -> go.Figure:
    """Visualizes rejection regions and test statistic on the standard normal or t-distribution."""
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Null Distribution H0: N(0, 1)", line=dict(color="#2C3E50", width=2)))
    
    if alternative == "two-sided":
        # Left tail
        x_left = x[x <= -abs(crit_val)]
        fig.add_trace(go.Scatter(x=x_left, y=stats.norm.pdf(x_left), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(231, 76, 60, 0)"), name=f"Rejection Region (α/2 = {alpha/2})"))
        # Right tail
        x_right = x[x >= abs(crit_val)]
        fig.add_trace(go.Scatter(x=x_right, y=stats.norm.pdf(x_right), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(231, 76, 60, 0)"), showlegend=False))
    elif alternative == "greater":
        x_right = x[x >= crit_val]
        fig.add_trace(go.Scatter(x=x_right, y=stats.norm.pdf(x_right), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(231, 76, 60, 0)"), name=f"Rejection Region (α = {alpha})"))
    else:  # less
        x_left = x[x <= crit_val]
        fig.add_trace(go.Scatter(x=x_left, y=stats.norm.pdf(x_left), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(231, 76, 60, 0)"), name=f"Rejection Region (α = {alpha})"))
        
    # Mark test statistic
    stat_clipped = np.clip(test_stat, -4, 4)
    fig.add_vline(x=stat_clipped, line=dict(color="#2980B9", width=3), annotation_text=f"Test Stat = {test_stat:.2f}")
    
    return apply_theme(fig, f"Hypothesis Testing Rejection Regions ({alternative.capitalize()}, α={alpha})", "Z / t Score", "Density")


def plot_two_regression_lines(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, reg_res: Dict[str, Any]) -> go.Figure:
    """Plots scatter plot along with two regression lines: Y on X and X on Y, intersecting at centroid."""
    n = min(len(x), 1500)
    indices = np.random.choice(len(x), size=n, replace=False) if len(x) > n else np.arange(len(x))
    x_sub = x[indices]
    y_sub = y[indices]
    
    fig = go.Figure()
    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_sub,
        y=y_sub,
        mode="markers",
        name="Data Points",
        marker=dict(color="#7F8C8D", size=4, opacity=0.5)
    ))
    
    # Regression line of Y on X
    x_range = np.linspace(float(np.min(x_sub)), float(np.max(x_sub)), 100)
    y_on_x = reg_res["intercept_yx"] + reg_res["slope_yx"] * x_range
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_on_x,
        mode="lines",
        name=f"Line 1 (Y on X): ŷ = {reg_res['intercept_yx']:.2f} + {reg_res['slope_yx']:.3f}x",
        line=dict(color="#E74C3C", width=3)
    ))
    
    # Regression line of X on Y: x = b_xy * y + a_xy => y = (x - a_xy) / b_xy
    if reg_res["slope_xy"] != 0:
        y_from_x_on_y = (x_range - reg_res["intercept_xy"]) / reg_res["slope_xy"]
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_from_x_on_y,
            mode="lines",
            name=f"Line 2 (X on Y): x̂ = {reg_res['intercept_xy']:.2f} + {reg_res['slope_xy']:.3f}y",
            line=dict(color="#2980B9", width=2.5, dash="dash")
        ))
        
    # Centroid (x_bar, y_bar)
    fig.add_trace(go.Scatter(
        x=[reg_res["x_bar"]],
        y=[reg_res["y_bar"]],
        mode="markers",
        name=f"Centroid (x̄={reg_res['x_bar']:.2f}, ȳ={reg_res['y_bar']:.2f})",
        marker=dict(color="#F39C12", size=12, symbol="star")
    ))
    
    return apply_theme(fig, f"Two Regression Lines: {y_label} vs {x_label} (r² = {reg_res['r_squared']:.3f})", x_label, y_label)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> go.Figure:
    """Renders interactive confusion matrix heatmap with count and normalized percentages."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    text = [[f"{cm[i, j]:,}<br>({cm_norm[i, j]*100:.1f}%)" for j in range(len(classes))] for i in range(len(classes))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True
    ))
    
    return apply_theme(fig, "Random Forest 5-Class Confusion Matrix", "Predicted Category", "True Category")
