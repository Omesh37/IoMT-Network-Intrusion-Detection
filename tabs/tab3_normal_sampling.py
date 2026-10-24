"""
Tab 3: Normal Distribution & Sampling Theory (Unit 3)
Normal fitting, Q-Q plots, normality tests, standard normal & Z-scores,
normal approximation to discrete distributions, Central Limit Theorem (CLT), and rejection regions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

from utils.stats_utils import (
    fit_normal_and_tests,
    clt_simulation
)
from utils.plot_utils import (
    plot_qq_norm,
    plot_rejection_regions,
    apply_theme
)


def render_tab3(df: pd.DataFrame, full_df: pd.DataFrame, primary_feature: str, log_transform: bool):
    st.header("Unit 3: Normal Distribution & Sampling Theory")

    with st.expander("📘 Mathematical Principles: Normal Distribution & CLT (Unit 3)", expanded=False):
        st.markdown(r"""
        ### Gaussian Distribution & Sampling Fundamentals
        - **Normal (Gaussian) Probability Density Function**:
          $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right), \quad x \in (-\infty, \infty)$$
        - **Standard Normal Transformation ($Z$-score)**:
          $$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$
        - **Central Limit Theorem (CLT)**:
          For any independent, identically distributed (i.i.d.) random variables with mean $\mu$ and finite variance $\sigma^2$:
          $$\bar{X}_n \xrightarrow{d} N\left(\mu, \; \frac{\sigma^2}{n}\right), \quad \text{Standard Error } SE = \frac{\sigma}{\sqrt{n}}$$
        - **Normal Approximation to Binomial & Poisson**:
          - Binomial($n, p$) $\approx N(\mu = np, \sigma = \sqrt{np(1-p)})$ when $np \ge 5$ and $n(1-p) \ge 5$
          - Continuity correction: $P(X \le k) \approx P(Y \le k + 0.5)$
        """)

    st.markdown("---")

    # Section 1: Normal Fit, Q-Q Plot, and Normality Tests
    st.subheader(f"1. Normal Distribution Fit & Diagnostics: {primary_feature}")
    feat_data = df[primary_feature].dropna()
    feat_data = feat_data[np.isfinite(feat_data)]
    
    if log_transform and (feat_data >= 0).all():
        active_data = np.log1p(feat_data)
        var_label = f"log1p({primary_feature})"
    else:
        active_data = feat_data
        var_label = primary_feature

    norm_res = fit_normal_and_tests(active_data.values)

    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    col_n1.metric("Fitted Mean (μ̂)", f"{norm_res['mean']:.4f}")
    col_n2.metric("Fitted Std Dev (σ̂)", f"{norm_res['std']:.4f}")
    col_n3.metric("Shapiro-Wilk p-value", f"{norm_res['shapiro_p']:.4e}")
    col_n4.metric("D'Agostino-Pearson p-value", f"{norm_res['dagostino_p']:.4e}")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        # Histogram with fitted normal curve
        sub_sample = active_data.sample(min(len(active_data), 3000), random_state=42)
        fig_norm_hist = go.Figure()
        fig_norm_hist.add_trace(go.Histogram(
            x=sub_sample, histnorm="probability density", nbinsx=45,
            name="Empirical Density", marker=dict(color="#3498DB", opacity=0.7)
        ))
        x_grid = np.linspace(sub_sample.min(), sub_sample.max(), 200)
        y_grid = stats.norm.pdf(x_grid, loc=norm_res["mean"], scale=norm_res["std"])
        fig_norm_hist.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode="lines",
            name=f"Fitted N(μ={norm_res['mean']:.1f}, σ={norm_res['std']:.1f})",
            line=dict(color="#E74C3C", width=2.5)
        ))
        apply_theme(fig_norm_hist, f"Normal Density Fit: {var_label}", var_label, "Density")
        st.plotly_chart(fig_norm_hist, use_container_width=True, key="tab3_norm_hist_chart")

    with col_q2:
        fig_qq = plot_qq_norm(active_data.values, var_label)
        st.plotly_chart(fig_qq, use_container_width=True, key="tab3_qq_chart")

    st.info(f"**Normality Diagnostic Verdict**: `{norm_res['verdict']}`. Note that high kurtosis in cyber telemetry often creates S-shaped departures on the Q-Q plot.")

    st.markdown("---")

    # Section 2: Standard Normal & Z-Scores
    st.subheader("2. Standard Normal Distribution & Z-Score Anomaly Detection")
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        z_threshold = st.slider("Z-Score Anomaly Threshold (|Z| > threshold):", min_value=1.5, max_value=5.0, value=3.0, step=0.25)
    with col_z2:
        calc_z_val = st.number_input("Interactive Z-Score Calculator (Input Z):", min_value=-4.0, max_value=4.0, value=1.96, step=0.01)

    # Calculate probabilities
    p_less_z = stats.norm.cdf(calc_z_val)
    p_two_tail = 2 * (1 - stats.norm.cdf(abs(calc_z_val)))
    
    st.markdown(f"""
    - **$P(Z < {calc_z_val:.2f})$**: `{p_less_z:.4f}` ({p_less_z*100:.2f}% cumulative area)
    - **Two-tailed tail area $P(|Z| > |{calc_z_val:.2f}|)$**: `{p_two_tail:.4f}` ({p_two_tail*100:.2f}%)
    """)

    # Anomaly detection via Z-Score on original feature
    mu_orig = float(feat_data.mean())
    std_orig = float(feat_data.std(ddof=1))
    if std_orig > 0:
        z_scores = np.abs((feat_data - mu_orig) / std_orig)
        outlier_mask = z_scores > z_threshold
        outlier_classes = df.loc[feat_data.index[outlier_mask], "category"].value_counts()
        total_outliers = int(outlier_mask.sum())
        
        st.write(f"**Anomalous Flows Detected with $|Z| > {z_threshold}$**: `{total_outliers:,}` out of `{len(feat_data):,}` flows ({total_outliers/len(feat_data)*100:.2f}%).")
        
        c_anom1, c_anom2 = st.columns([1, 1.2])
        with c_anom1:
            st.markdown("**Class Composition of Z-Score Outliers:**")
            st.dataframe(outlier_classes.rename("Count"), use_container_width=True)
            attack_outliers = outlier_classes[outlier_classes.index != "Benign"].sum()
            attack_fraction = (attack_outliers / total_outliers * 100.0) if total_outliers > 0 else 0
            st.metric("Attack Precision Among Outliers", f"{attack_fraction:.1f}%")
        with c_anom2:
            fig_z = go.Figure()
            z_x = np.linspace(-4, 4, 300)
            z_y = stats.norm.pdf(z_x)
            fig_z.add_trace(go.Scatter(x=z_x, y=z_y, mode="lines", name="Standard Normal N(0, 1)", line=dict(color="#2C3E50", width=2)))
            # Shade anomaly tail
            z_shade_left = z_x[z_x <= -z_threshold]
            z_shade_right = z_x[z_x >= z_threshold]
            fig_z.add_trace(go.Scatter(x=z_shade_left, y=stats.norm.pdf(z_shade_left), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(0,0,0,0)"), name=f"|Z| > {z_threshold}"))
            fig_z.add_trace(go.Scatter(x=z_shade_right, y=stats.norm.pdf(z_shade_right), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            apply_theme(fig_z, f"Standard Normal Anomaly Tails (|Z| > {z_threshold})", "Z-Score", "Density")
            st.plotly_chart(fig_z, use_container_width=True, key="tab3_z_anomaly_chart")

    st.markdown("---")



    # Section 4: Normal Approximation to Binomial and Poisson
    st.subheader("3. Normal Approximation to Discrete Distributions")
    approx_tabs = st.tabs(["Approximation to Binomial(n, p)", "Approximation to Poisson(λ)"])

    with approx_tabs[0]:
        c_app1, c_app2, c_app3 = st.columns(3)
        with c_app1:
            app_n = st.slider("Trials n:", min_value=10, max_value=200, value=50, step=5)
        with c_app2:
            app_p = st.slider("Probability p:", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
        with c_app3:
            use_cc = st.checkbox("Apply Continuity Correction (±0.5)", value=True)

        mu_bin = app_n * app_p
        sigma_bin = np.sqrt(app_n * app_p * (1 - app_p))
        rule_passed = (mu_bin >= 5) and (app_n * (1 - app_p) >= 5)

        k_grid = np.arange(0, app_n + 1)
        binom_pmf = stats.binom.pmf(k_grid, app_n, app_p)
        norm_pdf = stats.norm.pdf(k_grid, loc=mu_bin, scale=sigma_bin)

        # Max error between CDFs
        binom_cdf = stats.binom.cdf(k_grid, app_n, app_p)
        eval_points = (k_grid + 0.5) if use_cc else k_grid
        norm_cdf = stats.norm.cdf(eval_points, loc=mu_bin, scale=sigma_bin)
        max_cdf_err = float(np.max(np.abs(binom_cdf - norm_cdf)))

        fig_app_bin = go.Figure()
        fig_app_bin.add_trace(go.Bar(x=k_grid, y=binom_pmf, name="Exact Binomial PMF", marker=dict(color="#3498DB", opacity=0.7)))
        x_smooth = np.linspace(0, app_n, 300)
        fig_app_bin.add_trace(go.Scatter(x=x_smooth, y=stats.norm.pdf(x_smooth, mu_bin, sigma_bin), mode="lines", name=f"Normal N({mu_bin:.1f}, {sigma_bin:.1f})", line=dict(color="#E74C3C", width=2.5)))
        apply_theme(fig_app_bin, f"Normal Approx to Binomial(n={app_n}, p={app_p})", "k Successes", "Probability")
        st.plotly_chart(fig_app_bin, use_container_width=True, key="tab3_app_binom_chart")

        st.markdown(rf"""
        - **Rule of Thumb ($np \ge 5$ and $n(1-p) \ge 5$)**: `{rule_passed}` ($np = {mu_bin:.1f}$, $n(1-p) = {app_n*(1-app_p):.1f}$)
        - **Max Absolute CDF Discrepancy**: `{max_cdf_err:.4f}`
        """)

    with approx_tabs[1]:
        app_lam = st.slider("Poisson Mean λ:", min_value=2.0, max_value=60.0, value=20.0, step=2.0)
        mu_poi = app_lam
        sigma_poi = np.sqrt(app_lam)

        k_poi_grid = np.arange(0, int(app_lam + 4 * sigma_poi))
        poi_pmf = stats.poisson.pmf(k_poi_grid, app_lam)
        
        fig_app_poi = go.Figure()
        fig_app_poi.add_trace(go.Bar(x=k_poi_grid, y=poi_pmf, name="Exact Poisson PMF", marker=dict(color="#9B59B6", opacity=0.7)))
        x_p_smooth = np.linspace(0, max(k_poi_grid), 300)
        fig_app_poi.add_trace(go.Scatter(x=x_p_smooth, y=stats.norm.pdf(x_p_smooth, mu_poi, sigma_poi), mode="lines", name=f"Normal N(λ={mu_poi:.1f}, σ={sigma_poi:.1f})", line=dict(color="#F39C12", width=2.5)))
        apply_theme(fig_app_poi, f"Normal Approx to Poisson(λ={app_lam})", "Count k", "Probability")
        st.plotly_chart(fig_app_poi, use_container_width=True, key="tab3_app_poisson_chart")

    st.markdown("---")

    # Section 5: Central Limit Theorem (CLT) Interactive Simulation
    st.subheader("4. Central Limit Theorem (CLT) & Sampling Distribution of the Mean")
    st.markdown("""
    The Central Limit Theorem guarantees that the distribution of sample means approaches a Gaussian distribution 
    as sample size $n$ increases, **regardless of how heavily skewed or non-normal the underlying population is**.
    """)

    clt_feat = st.selectbox("Select Skewed Population Feature:", options=["Rate", "Duration", "IAT", "Tot sum"])
    clt_pop = df[clt_feat].dropna().values

    col_clt1, col_clt2, col_clt3 = st.columns(3)
    with col_clt1:
        clt_n = st.slider("Sample Size n (Per Sample):", min_value=2, max_value=200, value=30, step=2)
    with col_clt2:
        clt_samples = st.slider("Number of Repeated Samples M:", min_value=200, max_value=3000, value=1000, step=100)
    with col_clt3:
        clt_seed = st.number_input("Simulation Seed:", value=42, step=1)

    clt_res = clt_simulation(clt_pop, sample_size=clt_n, num_samples=clt_samples, seed=int(clt_seed))

    c_clt_kpi1, c_clt_kpi2, c_clt_kpi3, c_clt_kpi4 = st.columns(4)
    c_clt_kpi1.metric("Population Mean (μ)", f"{clt_res['pop_mean']:.2f}")
    c_clt_kpi2.metric("Mean of Sample Means (x̄̄)", f"{clt_res['mean_of_means']:.2f}")
    c_clt_kpi3.metric("Theoretical SE (σ / √n)", f"{clt_res['theoretical_se']:.2f}")
    c_clt_kpi4.metric("Empirical SD of Means (s_x̄)", f"{clt_res['empirical_se']:.2f}")

    # Plot sampling distribution with normal overlay
    fig_clt = go.Figure()
    fig_clt.add_trace(go.Histogram(
        x=clt_res["sample_means"],
        histnorm="probability density",
        nbinsx=40,
        name=f"Sample Means (n={clt_n})",
        marker=dict(color="#1ABC9C", opacity=0.7)
    ))
    x_clt_grid = np.linspace(clt_res["sample_means"].min(), clt_res["sample_means"].max(), 200)
    y_clt_grid = stats.norm.pdf(x_clt_grid, loc=clt_res["pop_mean"], scale=clt_res["theoretical_se"])
    fig_clt.add_trace(go.Scatter(
        x=x_clt_grid, y=y_clt_grid, mode="lines",
        name=f"Theoretical Normal N(μ, SE={clt_res['theoretical_se']:.2f})",
        line=dict(color="#E74C3C", width=3)
    ))
    apply_theme(fig_clt, f"Sampling Distribution of Mean for {clt_feat} (M={clt_samples}, n={clt_n})", "Sample Mean x̄", "Density")
    st.plotly_chart(fig_clt, use_container_width=True, key="tab3_clt_hist_chart")


    st.markdown("---")

    # Section 6: One-Tailed vs Two-Tailed Rejection Regions Visualizer
    st.subheader("5. Hypothesis Testing Rejection Regions on the Standard Normal Curve")
    col_rej1, col_rej2 = st.columns(2)
    with col_rej1:
        alpha_choice = st.select_slider("Significance Level α:", options=[0.01, 0.05, 0.10], value=0.05)
    with col_rej2:
        tail_choice = st.radio("Test Direction:", options=["two-sided", "greater", "less"], horizontal=True)

    if tail_choice == "two-sided":
        c_val = stats.norm.ppf(1 - alpha_choice / 2)
        crit_text = f"Critical Values: Z = ±{c_val:.3f}"
    elif tail_choice == "greater":
        c_val = stats.norm.ppf(1 - alpha_choice)
        crit_text = f"Critical Value: Z = +{c_val:.3f}"
    else:
        c_val = stats.norm.ppf(alpha_choice)
        crit_text = f"Critical Value: Z = {c_val:.3f}"

    st.markdown(f"**{crit_text}** for {tail_choice} test at $\\alpha = {alpha_choice}$.")
    fig_rej = plot_rejection_regions(crit_val=c_val, test_stat=2.15, alternative=tail_choice, alpha=alpha_choice)
    st.plotly_chart(fig_rej, use_container_width=True, key="tab3_rejection_regions_chart")
