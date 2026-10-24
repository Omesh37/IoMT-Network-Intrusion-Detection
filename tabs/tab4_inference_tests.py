"""
Tab 4: Estimation, Hypothesis Testing & ANOVA (Unit 4)
Point estimation (Bessel's correction proof),
One-Sample & Two-Sample Welch t-tests, Type I/II Error and Power analysis,
Chi-Square Goodness-of-Fit and Independence tests, and One-Way ANOVA with Tukey HSD.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

from utils.stats_utils import (
    simulate_variance_bias,
    one_sample_test,
    two_sample_welch,
    type_errors_power,
    chi_square_gof,
    chi_square_independence,
    one_way_anova
)
from utils.plot_utils import (
    plot_rejection_regions,
    apply_theme
)


def render_tab4(df: pd.DataFrame, full_df: pd.DataFrame, primary_feature: str):
    st.header("Unit 4: Estimation, Statistical Hypothesis Testing & ANOVA")

    # Concept Expander
    with st.expander("📘 Statistical Inference, Testing & ANOVA Principles (Unit 4)", expanded=False):
        st.markdown(r"""
        ### Principles of Estimation & Hypothesis Testing
        - **Unbiased Variance Estimator (Bessel's Correction)**:
          $$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2 \quad \text{guarantees} \quad E[s^2] = \sigma^2$$
        - **Two-Sample Welch's t-Test & Cohen's d**:
          $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}, \quad \text{Cohen's } d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
        - **Type I ($\alpha$) and Type II ($\beta$) Error**:
          - $\alpha = P(\text{Reject } H_0 \mid H_0 \text{ true}) \implies \text{False Alarm (Benign flagged as attack)}$
          - $\beta = P(\text{Fail to reject } H_0 \mid H_1 \text{ true}) \implies \text{Missed Detection (Attack accepted as benign)}$
          - $\text{Statistical Power} = 1 - \beta$
        - **Chi-Square Tests**:
          $$\chi^2 = \sum \frac{(O - E)^2}{E}, \quad \text{Cramér's } V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$
        - **One-Way ANOVA & F-Statistic**:
          $$F = \frac{MSB}{MSW} = \frac{SSB / (k - 1)}{SSW / (N - k)}$$
        """)

    st.markdown("---")

    # Big Data P-Value Saturation Warning & Subsample Slider
    st.subheader("⚙️ Large-Sample Effect Size & Subsampling Control")
    st.markdown(r"""
    > [!IMPORTANT]
    > **The "Big Data" P-Value Trap**: With hundreds of thousands of network flows, standard error shrinks ($SE \propto 1/\sqrt{n}$), 
    > causing $p$-values to become trivially small ($p < 10^{-15}$) even for practically meaningless deviations. 
    > In modern data science, **effect sizes** (Cohen's $d$, Cramér's $V$, $\eta^2$) must be evaluated alongside $p$-values. 
    > Use the subsample slider below to observe how test statistics behave under standard experimental sample sizes.
    """)

    sub_n = st.slider("Inference Subsample Size (for Hypothesis Tests):", min_value=100, max_value=5000, value=1000, step=100)
    rng_seed = st.number_input("Random Seed:", value=42, step=1)
    
    # Stratified subsample for inference testing
    inf_df = df.groupby("category", group_keys=False).apply(
        lambda x: x.sample(max(20, int(np.round(len(x) / len(df) * sub_n))), random_state=int(rng_seed))
    ).copy()
    
    st.caption(f"Active inference sample: {len(inf_df):,} flows across {inf_df['category'].nunique()} categories.")

    st.markdown("---")

    # Section 1: Point Estimation & Proof of Bessel's Correction
    st.subheader("1. Point Estimation & Proof of Bessel's Correction (n - 1)")
    st.markdown(r"Why do statisticians divide by $n - 1$ instead of $n$ when calculating sample variance?")

    col_b1, col_b2 = st.columns([1, 1.2])
    with col_b1:
        sim_n = st.slider("Sample Size in Simulation (n):", min_value=3, max_value=15, value=5)
        sim_trials = st.slider("Simulation Trials:", min_value=500, max_value=3000, value=1500, step=250)

    pop_feature_data = df[primary_feature].dropna().values
    bias_sim = simulate_variance_bias(pop_feature_data, sample_size=sim_n, num_trials=sim_trials, seed=int(rng_seed))

    with col_b2:
        st.markdown(rf"""
        - **True Population Variance $\sigma^2$**: `{bias_sim['true_pop_var']:.4f}`
        - **Average of Biased Estimator $S_n^2$ (dividing by $n$)**: `{bias_sim['mean_biased_var']:.4f}`
        - **Theoretical Expected Biased Value $\frac{{n-1}}{{n}}\sigma^2$**: `{bias_sim['theoretical_biased_expected']:.4f}`
        - **Average of Unbiased Estimator $s^2$ (dividing by $n-1$)**: `{bias_sim['mean_unbiased_var']:.4f}`
        - **Empirical Bias in $S_n^2$**: `{bias_sim['bias_percentage']:.2f}%` downward bias!
        """)
        st.success(f"Bessel's correction ($n-1$) successfully removes the downward bias of {abs(bias_sim['bias_percentage']):.1f}%, yielding an exactly unbiased estimator.")

    st.markdown("---")

    # Section 2: One-Sample Z-Test & t-Test
    st.subheader(f"2. One-Sample Hypothesis Testing: {primary_feature}")
    c_h1, c_h2, c_h3, c_h4 = st.columns(4)
    with c_h1:
        mu_0 = st.number_input("Hypothesized Mean (μ0):", value=float(inf_df[primary_feature].mean() * 0.90), format="%.2f")
    with c_h2:
        alpha_val = st.selectbox("Significance Level α:", options=[0.01, 0.05, 0.10], index=1)
    with c_h3:
        test_type = st.radio("Test Method:", options=["t (Student's t-test)", "z (Z-test)"])
    with c_h4:
        alt_hyp = st.radio("Alternative Hypothesis H1:", options=["two-sided (μ ≠ μ0)", "greater (μ > μ0)", "less (μ < μ0)"])

    raw_test_type = "t" if "t" in test_type else "z"
    raw_alt = "two-sided" if "two-sided" in alt_hyp else "greater" if "greater" in alt_hyp else "less"

    ht_res = one_sample_test(inf_df[primary_feature].values, mu_0=mu_0, alpha=alpha_val, test_type=raw_test_type, alternative=raw_alt)

    st.markdown(rf"""
    - **Null Hypothesis ($H_0$)**: $\mu = {mu_0:.2f}$ (Mean {primary_feature} equals {mu_0:.2f})
    - **Alternative Hypothesis ($H_1$)**: $\mu {('≠' if raw_alt == 'two-sided' else '>' if raw_alt == 'greater' else '<')} {mu_0:.2f}$
    - **Test Statistic**: `{ht_res['test_type'].upper()} = {ht_res['stat']:.4f}`
    - **p-value**: `{ht_res['p_value']:.4e}` (Critical value: `{ht_res['critical_value']:.4f}`)
    - **Decision**: **{ht_res['decision']}** at $\alpha = {alpha_val}$.
    """)

    fig_ht = plot_rejection_regions(crit_val=ht_res["critical_value"], test_stat=ht_res["stat"], alternative=raw_alt, alpha=alpha_val)
    st.plotly_chart(fig_ht, use_container_width=True, key="tab4_ht_rejection_chart")

    st.markdown("---")

    # Section 3: Two-Sample Welch t-Test (Benign vs Attack)
    st.subheader("3. Two-Sample Welch's t-Test & Cohen's d Effect Size")
    st.markdown("Compare feature distributions between **Benign** baseline traffic and a target attack category (e.g. Benign vs Spoofing).")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        target_attack = st.selectbox("Compare Benign against Attack Category:", options=[c for c in df["category"].unique() if c != "Benign"])
    with col_w2:
        welch_feature = st.selectbox("Select Feature to Compare:", options=["Rate", "Duration", "IAT", "Header_Length", "Max", "Variance"], index=0)

    benign_vals = inf_df[inf_df["category"] == "Benign"][welch_feature].dropna().values
    attack_vals = inf_df[inf_df["category"] == target_attack][welch_feature].dropna().values

    if len(benign_vals) > 5 and len(attack_vals) > 5:
        welch_res = two_sample_welch(benign_vals, attack_vals)
        
        w_kpi1, w_kpi2, w_kpi3, w_kpi4 = st.columns(4)
        w_kpi1.metric("Benign Mean", f"{welch_res['mean1']:.2f}")
        w_kpi2.metric(f"{target_attack} Mean", f"{welch_res['mean2']:.2f}")
        w_kpi3.metric("Welch t-stat", f"{welch_res['t_stat']:.3f}")
        w_kpi4.metric("Cohen's d Effect Size", f"{welch_res['cohens_d']:.3f}")

        st.info(f"""
        **Inference Verdict**: Welch's $t = {welch_res['t_stat']:.3f}$ ($df = {welch_res['df']:.1f}$, $p = {welch_res['p_value']:.4e}$).  
        **Effect Size**: Cohen's $d = {welch_res['cohens_d']:.3f}$ ({welch_res['d_interpretation']}).  
        *Cybersecurity Insight*: Spoofing attacks yield small-to-medium Cohen's $d$ against Benign flows on volumetric features, making them notoriously hard to separate with simple thresholding compared to volumetric DDoS attacks ($d > 2.0$).
        """)
    else:
        st.warning("Insufficient samples in chosen categories.")

    st.markdown("---")

    # Section 4: Type I and Type II Errors & Statistical Power
    st.subheader("4. Type I Error (α), Type II Error (β), & Statistical Power (1 - β)")
    st.markdown("""
    In intrusion detection:
    - **Type I Error ($\alpha$, False Alarm)**: Null hypothesis is true (Benign traffic), but test rejects $H_0$ and triggers an alert.
    - **Type II Error ($\beta$, Missed Detection)**: Alternative hypothesis is true (Malicious attack), but test fails to reject $H_0$.
    - **Power ($1 - \beta$)**: Probability of successfully detecting an attack when one is present.
    """)

    col_err1, col_err2, col_err3 = st.columns(3)
    with col_err1:
        err_alpha = st.slider("Significance Level α (False Alarm Tolerance):", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    with col_err2:
        err_n = st.slider("Sample Size n (Packets inspected):", min_value=10, max_value=150, value=30, step=5)
    with col_err3:
        err_delta = st.slider("Attack Effect Size Shift (μ1 - μ0):", min_value=0.5, max_value=3.0, value=1.0, step=0.25)

    err_res = type_errors_power(mu0=0.0, mu1=err_delta, sigma=1.0, n=err_n, alpha=err_alpha, alternative="greater")

    e_kpi1, e_kpi2, e_kpi3 = st.columns(3)
    e_kpi1.metric("Type I Error Rate (α)", f"{err_res['alpha']:.3f} ({err_res['alpha']*100:.1f}%)")
    e_kpi2.metric("Type II Error Rate (β)", f"{err_res['beta']:.3f} ({err_res['beta']*100:.1f}%)")
    e_kpi3.metric("Statistical Power (1 - β)", f"{err_res['power']:.3f} ({err_res['power']*100:.1f}%)")

    # Plot dual distributions
    x_err_grid = np.linspace(-3, err_delta + 3, 500)
    se_val = err_res["se"]
    y_null = stats.norm.pdf(x_err_grid, loc=0.0, scale=se_val)
    y_alt = stats.norm.pdf(x_err_grid, loc=err_delta, scale=se_val)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=x_err_grid, y=y_null, mode="lines", name="Null H0: Benign (μ=0)", line=dict(color="#2980B9", width=2.5)))
    fig_err.add_trace(go.Scatter(x=x_err_grid, y=y_alt, mode="lines", name=f"Alt H1: Attack (μ={err_delta})", line=dict(color="#E74C3C", width=2.5)))
    
    # Shade alpha (Type I)
    x_alpha_shade = x_err_grid[x_err_grid >= err_res["x_crit"]]
    fig_err.add_trace(go.Scatter(x=x_alpha_shade, y=stats.norm.pdf(x_alpha_shade, loc=0.0, scale=se_val), fill="tozeroy", fillcolor="rgba(41, 128, 185, 0.4)", line=dict(color="rgba(0,0,0,0)"), name=f"Type I Error α ({err_res['alpha']:.2f})"))
    
    # Shade beta (Type II)
    x_beta_shade = x_err_grid[x_err_grid <= err_res["x_crit"]]
    fig_err.add_trace(go.Scatter(x=x_beta_shade, y=stats.norm.pdf(x_beta_shade, loc=err_delta, scale=se_val), fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.4)", line=dict(color="rgba(0,0,0,0)"), name=f"Type II Error β ({err_res['beta']:.2f})"))
    
    fig_err.add_vline(x=err_res["x_crit"], line=dict(color="#2C3E50", width=2, dash="dash"), annotation_text=f"Decision Boundary = {err_res['x_crit']:.2f}")
    apply_theme(fig_err, "Type I & Type II Error Distribution Overlap", "Observed Test Statistic x̄", "Density")
    st.plotly_chart(fig_err, use_container_width=True, key="tab4_type_errors_chart")

    st.markdown("---")

    # Section 5: Chi-Square Tests (Goodness-of-Fit & Independence)
    st.subheader("5. Chi-Square Goodness-of-Fit & Independence Tests")
    chi_tabs = st.tabs(["Goodness-of-Fit Test", "Test of Independence (Protocol vs Attack)"])

    with chi_tabs[0]:
        st.markdown("#### Test whether traffic category distribution conforms to a Uniform expectation")
        gof_res = chi_square_gof(inf_df["category"])
        st.dataframe(gof_res["table"].style.format({
            "Observed (O)": "{:,}",
            "Expected (E)": "{:,.1f}",
            "Residual (O - E)": "{:+,.1f}",
            "Chi2 Component ((O-E)^2 / E)": "{:.2f}"
        }), use_container_width=True)
        st.write(f"**Chi-Square Statistic**: $\\chi^2 = {gof_res['chi2_stat']:.2f}$, $df = {gof_res['df']}$, $p = {gof_res['p_value']:.4e}$.")
        if gof_res["reject_null"]:
            st.warning("Reject Null Hypothesis: Empirical traffic distribution severely deviates from uniform distribution.")

    with chi_tabs[1]:
        st.markdown("#### Test Independence between Network Protocol (TCP/UDP/ICMP/ARP) and Attack Category")
        # Build protocol label column
        proto_series = np.select(
            [inf_df["TCP"] == 1.0, inf_df["UDP"] == 1.0, inf_df["ARP"] == 1.0, inf_df["ICMP"] == 1.0],
            ["TCP", "UDP", "ARP", "ICMP"],
            default="Other"
        )
        inf_df["Protocol_Name"] = proto_series
        
        indep_res = chi_square_independence(inf_df, "Protocol_Name", "category")
        
        col_ind1, col_ind2 = st.columns(2)
        with col_ind1:
            st.markdown("**Contingency Table (Observed Counts):**")
            st.dataframe(indep_res["contingency"], use_container_width=True)
            st.metric("Cramér's V (Effect Size)", f"{indep_res['cramers_v']:.4f}")
        with col_ind2:
            st.markdown("**Standardized Pearson Residuals Heatmap:**")
            fig_resid = px.imshow(
                indep_res["std_residuals"],
                color_continuous_scale="RdBu_r",
                labels=dict(x="Traffic Category", y="Protocol", color="Std Residual")
            )
            apply_theme(fig_resid, "Pearson Residuals ((O - E) / √E)", "Traffic Category", "Protocol")
            st.plotly_chart(fig_resid, use_container_width=True, key="tab4_chi2_residuals_chart")

        st.info(f"Chi-square independence test: $\\chi^2 = {indep_res['chi2_stat']:.1f}$, $df = {indep_res['df']}$, $p = {indep_res['p_value']:.4e}$, Cramér's $V = {indep_res['cramers_v']:.3f}$. Strongly rejects independence!")

    st.markdown("---")

    # Section 6: One-Way ANOVA with F-Test & Tukey HSD
    st.subheader(f"6. One-Way ANOVA with F-Test: {primary_feature} Across Classes")
    anova_res = one_way_anova(inf_df, "category", primary_feature)

    if "error" not in anova_res:
        st.markdown("#### Complete ANOVA Table")
        st.dataframe(anova_res["anova_table"].style.format({
            "Sum of Squares (SS)": "{:,.2f}",
            "Degrees of Freedom (df)": "{:,}",
            "Mean Square (MS)": "{:,.2f}",
            "F-Statistic": "{:.3f}",
            "p-value": "{:.4e}"
        }), use_container_width=True)

        a_kpi1, a_kpi2, a_kpi3 = st.columns(3)
        a_kpi1.metric("F-Statistic", f"{anova_res['f_stat']:.3f}")
        a_kpi2.metric("p-value", f"{anova_res['p_val']:.4e}")
        a_kpi3.metric("Eta-Squared (η² Variance Explained)", f"{anova_res['eta_squared']:.4f}")

        # Assumption checks
        st.markdown(f"""
        - **Levene's Test for Homogeneity of Variance**: Stat = `{anova_res['levene_stat']:.3f}`, p-value = `{anova_res['levene_p']:.4e}`.
        - **Kruskal-Wallis Non-Parametric Fallback**: Stat = `{anova_res['kruskal_stat']:.3f}`, p-value = `{anova_res['kruskal_p']:.4e}`.
        """)
        
        with st.expander("🔬 Tukey HSD Post-Hoc Pairwise Comparisons", expanded=False):
            st.dataframe(anova_res["tukey_df"], use_container_width=True)
    else:
        st.warning("Insufficient data across classes to perform ANOVA.")
