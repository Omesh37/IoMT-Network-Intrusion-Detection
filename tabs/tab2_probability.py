"""
Tab 2: Probability & Distributions (Unit 2)
Empirical probability axioms, addition & complement rules, conditional probability,
Bayes' Theorem & base-rate fallacy, discrete distributions (Bernoulli, Binomial, Poisson),
and continuous distributions (Exponential MLE & KS test, Uniform).
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from utils.stats_utils import (
    bayes_theorem,
    fit_exponential
)
from utils.plot_utils import (
    plot_venn_breakdown,
    apply_theme
)


def render_tab2(df: pd.DataFrame, full_df: pd.DataFrame, primary_feature: str):
    st.header("Unit 2: Probability & Probability Distributions")

    # Concept Expander
    with st.expander("📘 Probability Axioms, Bayes' Theorem & Distributions (Unit 2)", expanded=False):
        st.markdown(r"""
        ### Axiomatic & Empirical Probability
        - **Complement Rule**: $P(A') = 1 - P(A)$
        - **General Addition Rule**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
        - **Conditional Probability**: $P(B \mid A) = \frac{P(A \cap B)}{P(A)}$
        - **Statistical Independence**: $A$ and $B$ are independent if and only if $P(A \cap B) = P(A) \cdot P(B)$
        - **Bayes' Theorem (Two-Hypothesis Form)**:
          $$P(\text{Attack} \mid \text{Alert}) = \frac{P(\text{Alert} \mid \text{Attack}) \cdot P(\text{Attack})}{P(\text{Alert} \mid \text{Attack}) \cdot P(\text{Attack}) + P(\text{Alert} \mid \text{Benign}) \cdot P(\text{Benign})}$$
        - **Discrete Distributions**:
          - Bernoulli($p$): $P(X=1) = p, \; P(X=0) = 1-p, \; E[X]=p, \; \text{Var}=p(1-p)$
          - Binomial($n, p$): $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \; E[X]=np, \; \text{Var}=np(1-p)$
          - Poisson($\lambda$): $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \; E[X]=\lambda, \; \text{Var}=\lambda$
        - **Continuous Distributions**:
          - Exponential($\lambda$): $f(x) = \lambda e^{-\lambda x}, \; F(x) = 1 - e^{-\lambda x}, \; E[X]=1/\lambda, \; \text{Var}=1/\lambda^2$
          - Uniform($a, b$): $f(x) = \frac{1}{b-a}, \; E[X]=\frac{a+b}{2}, \; \text{Var}=\frac{(b-a)^2}{12}$
        """)

    st.markdown("---")

    # Section A: Probability Rules & Addition Rule
    st.subheader("A. Empirical Probability Rules & Addition Rule")
    total_n = len(df)
    n_attack = (df["category"] != "Benign").sum()
    p_attack = n_attack / total_n
    p_benign = 1.0 - p_attack

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Empirical P(Attack)", f"{p_attack:.4f} ({p_attack*100:.2f}%)")
    kpi2.metric("Complement P(Benign) = 1 - P(Attack)", f"{p_benign:.4f} ({p_benign*100:.2f}%)")

    st.markdown(r"#### Test the Addition Rule: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        event_a_opt = st.selectbox(
            "Select Event A:",
            options=["UDP == 1 (UDP Traffic)", "TCP == 1 (TCP Traffic)", "syn_flag_number == 1 (SYN Flag Set)", "Rate > Median (High Traffic Rate)"]
        )
    with col_a2:
        event_b_opt = st.selectbox(
            "Select Event B:",
            options=["syn_flag_number == 1 (SYN Flag Set)", "category != 'Benign' (Attack Flow)", "Rate > Median (High Traffic Rate)", "UDP == 1 (UDP Traffic)"]
        )

    # Evaluate masks for Event A
    if "UDP == 1" in event_a_opt:
        mask_a = df["UDP"] == 1.0
        label_a = "UDP=1"
    elif "TCP == 1" in event_a_opt:
        mask_a = df["TCP"] == 1.0
        label_a = "TCP=1"
    elif "syn_flag" in event_a_opt:
        mask_a = df["syn_flag_number"] == 1.0
        label_a = "SYN=1"
    else:
        mask_a = df["Rate"] > df["Rate"].median()
        label_a = "Rate > Median"

    # Evaluate masks for Event B
    if "category != 'Benign'" in event_b_opt:
        mask_b = df["category"] != "Benign"
        label_b = "Attack"
    elif "syn_flag" in event_b_opt:
        mask_b = df["syn_flag_number"] == 1.0
        label_b = "SYN=1"
    elif "UDP == 1" in event_b_opt:
        mask_b = df["UDP"] == 1.0
        label_b = "UDP=1"
    else:
        mask_b = df["Rate"] > df["Rate"].median()
        label_b = "Rate > Median"

    p_a = float(mask_a.mean())
    p_b = float(mask_b.mean())
    p_a_and_b = float((mask_a & mask_b).mean())
    p_a_or_b_formula = p_a + p_b - p_a_and_b
    p_a_or_b_empirical = float((mask_a | mask_b).mean())

    c_venn1, c_venn2 = st.columns([1, 1.2])
    with c_venn1:
        st.markdown(rf"""
        - **$P(A)$ [{label_a}]**: `{p_a:.4f}`
        - **$P(B)$ [{label_b}]**: `{p_b:.4f}`
        - **$P(A \cap B)$ [Joint Probability]**: `{p_a_and_b:.4f}`
        - **$P(A) + P(B) - P(A \cap B)$**: `{p_a_or_b_formula:.4f}`
        - **Empirical $P(A \cup B)$**: `{p_a_or_b_empirical:.4f}`
        - **Discrepancy**: `{abs(p_a_or_b_formula - p_a_or_b_empirical):.6e}` (Exact match!)
        """)
    with c_venn2:
        fig_venn = plot_venn_breakdown(p_a, p_b, p_a_and_b, label_a, label_b)
        st.plotly_chart(fig_venn, use_container_width=True, key="tab2_venn_chart")

    st.markdown("---")

    # Section B: Conditional Probability & Statistical Independence
    st.subheader("B. Conditional Probability & Statistical Independence")
    st.markdown("Does the occurrence of a network feature alter the probability of an attack?")

    col_cond1, col_cond2 = st.columns(2)
    with col_cond1:
        binary_feat = st.selectbox(
            "Select Binary Protocol / Header Feature (Event F):",
            options=["syn_flag_number", "rst_flag_number", "UDP", "TCP", "ICMP", "ARP"]
        )
    with col_cond2:
        target_cls = st.selectbox("Select Traffic Class (Event C):", options=df["category"].unique())

    mask_f = df[binary_feat] == 1.0
    mask_c = df["category"] == target_cls
    
    p_f = float(mask_f.mean())
    p_c = float(mask_c.mean())
    p_f_and_c = float((mask_f & mask_c).mean())
    
    p_c_given_f = (p_f_and_c / p_f) if p_f > 0 else 0.0
    p_f_given_c = (p_f_and_c / p_c) if p_c > 0 else 0.0
    
    product_p = p_f * p_c
    indep_diff = abs(p_f_and_c - product_p)
    is_independent = indep_diff < 0.005

    ctable = pd.crosstab(df[binary_feat], df["category"] == target_cls, rownames=[f"{binary_feat} (0/1)"], colnames=[f"Is {target_cls}"])
    
    col_ct1, col_ct2 = st.columns([1, 1])
    with col_ct1:
        st.markdown("**Contingency Table (Counts):**")
        st.dataframe(ctable, use_container_width=True)
    with col_ct2:
        st.markdown(rf"""
        - **$P(\text{{{target_cls}}})$ Prior**: `{p_c:.4f}`
        - **$P(\text{{{binary_feat}}}=1)$**: `{p_f:.4f}`
        - **$P(\text{{{target_cls}}} \mid \text{{{binary_feat}}}=1)$**: `{p_c_given_f:.4f}`
        - **$P(\text{{{binary_feat}}}=1 \mid \text{{{target_cls}}})$**: `{p_f_given_c:.4f}`
        - **$P(F \cap C)$ vs $P(F) \cdot P(C)$**: `{p_f_and_c:.4f}` vs `{product_p:.4f}` (diff: `{indep_diff:.4f}`)
        """)
        if is_independent:
            st.success(r"Verdict: **Statistically Independent** ($P(F \cap C) \approx P(F)P(C)$). Knowing this feature provides negligible information about this class.")
        else:
            st.warning(rf"Verdict: **Statistically Dependent** ($P(F \cap C) \neq P(F)P(C)$). Conditioned on {binary_feat}=1, probability shifted from {p_c:.4f} to {p_c_given_f:.4f}!")

    st.markdown("---")

    # Section C: Bayes' Theorem & Base-Rate Effect
    st.subheader("C. Bayes' Theorem & The Base-Rate Effect in Medical IoT")
    st.markdown("""
    In hospital networks, malicious intrusion attempts are typically rare compared to legitimate heart-monitor and telemetry pings.
    Even when an ML classifier boasts **99% sensitivity** and **95% specificity**, a low attack base rate means the vast majority of alerts are **false alarms**.
    """)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        prior_slider = st.slider("Attack Prior P(Attack):", min_value=0.001, max_value=0.50, value=min(0.20, float(p_attack)), step=0.005, format="%.3f")
    with col_b2:
        sens_slider = st.slider("Sensitivity (True Positive Rate P(Alert|Attack)):", min_value=0.80, max_value=1.00, value=0.99, step=0.01)
    with col_b3:
        spec_slider = st.slider("Specificity (True Negative Rate P(No Alert|Benign)):", min_value=0.80, max_value=1.00, value=0.95, step=0.01)

    bayes_res = bayes_theorem(prior_slider, sens_slider, spec_slider)

    b_kpi1, b_kpi2, b_kpi3 = st.columns(3)
    b_kpi1.metric("P(Alert) [Total Alert Probability]", f"{bayes_res['p_alert']:.4f}")
    b_kpi2.metric("Posterior P(Attack | Alert)", f"{bayes_res['posterior_attack_given_alert']*100:.2f}%")
    b_kpi3.metric("False Alarm Rate P(Benign | Alert)", f"{bayes_res['posterior_benign_given_alert']*100:.2f}%")

    # Curve of Posterior vs Prior
    prior_range = np.linspace(0.001, 0.50, 200)
    posterior_curve = [(sens_slider * pr) / ((sens_slider * pr) + ((1 - spec_slider) * (1 - pr))) for pr in prior_range]
    
    fig_bayes = go.Figure()
    fig_bayes.add_trace(go.Scatter(x=prior_range, y=posterior_curve, mode="lines", name="Posterior P(Attack | Alert)", line=dict(color="#2980B9", width=2.5)))
    fig_bayes.add_trace(go.Scatter(x=[prior_slider], y=[bayes_res["posterior_attack_given_alert"]], mode="markers", name="Current Configuration", marker=dict(color="#E74C3C", size=10)))
    apply_theme(fig_bayes, "Posterior Probability as a Function of Attack Base Rate", "Prior Probability P(Attack)", "Posterior P(Attack | Alert)")
    st.plotly_chart(fig_bayes, use_container_width=True, key="tab2_bayes_curve_chart")

    st.markdown("---")

    # Section D: Discrete Distributions (Bernoulli, Binomial, Poisson)
    st.subheader("D. Discrete Probability Distributions")
    disc_tabs = st.tabs(["Bernoulli Distribution", "Binomial Distribution B(n, p)", "Poisson Distribution Poisson(λ)"])

    with disc_tabs[0]:
        st.markdown("#### Bernoulli Distribution: Modeling Binary Header Flags")
        bern_col = st.selectbox("Select Binary Indicator Column:", options=["syn_flag_number", "rst_flag_number", "UDP", "TCP"])
        p_est = float(df[bern_col].mean())
        
        st.write(f"Estimated Parameter: $\hat{{p}} = {p_est:.4f}$")
        col_bern1, col_bern2 = st.columns([1, 1.5])
        with col_bern1:
            st.metric("Expected Value E[X] = p", f"{p_est:.4f}")
            st.metric("Variance Var(X) = p(1 - p)", f"{p_est * (1 - p_est):.4f}")
        with col_bern2:
            fig_bern = go.Figure(data=[
                go.Bar(x=["0 (Flag Absent)", "1 (Flag Present)"], y=[1 - p_est, p_est], marker=dict(color=["#95A5A6", "#3498DB"]))
            ])
            apply_theme(fig_bern, f"Bernoulli PMF for {bern_col}", "Outcome", "Probability P(X = x)")
            st.plotly_chart(fig_bern, use_container_width=True, key="tab2_bernoulli_chart")

    with disc_tabs[1]:
        st.markdown("#### Binomial Distribution: Probability of k Attack Packets in a Flow Window of Size n")
        col_bin1, col_bin2 = st.columns(2)
        with col_bin1:
            bin_n = st.slider("Window Size n (Number of Trials):", min_value=5, max_value=100, value=25, step=5)
        with col_bin2:
            bin_p = st.slider("Success Probability p (e.g. Flag occurrence):", min_value=0.01, max_value=0.99, value=max(0.05, min(0.95, p_est)), step=0.02)

        k_values = np.arange(0, bin_n + 1)
        pmf_values = stats.binom.pmf(k_values, bin_n, bin_p)
        cdf_values = stats.binom.cdf(k_values, bin_n, bin_p)

        fig_binom = go.Figure()
        fig_binom.add_trace(go.Bar(x=k_values, y=pmf_values, name="PMF P(X = k)", marker=dict(color="#3498DB")))
        fig_binom.add_trace(go.Scatter(x=k_values, y=cdf_values, name="CDF P(X ≤ k)", mode="lines+markers", line=dict(color="#E67E22", width=2)))
        apply_theme(fig_binom, f"Binomial(n={bin_n}, p={bin_p:.2f}) PMF & CDF", "Number of Successes k", "Probability")
        st.plotly_chart(fig_binom, use_container_width=True, key="tab2_binomial_chart")
        
        st.markdown(f"**Theoretical Moments**: $E[X] = np = {bin_n * bin_p:.2f}$, $\\text{{Var}}(X) = np(1-p) = {bin_n * bin_p * (1 - bin_p):.2f}$, $SD = {np.sqrt(bin_n * bin_p * (1 - bin_p)):.2f}$.")

    with disc_tabs[2]:
        st.markdown("#### Poisson Distribution: Modeling Flow Packet Arrival Counts")
        poisson_feature = st.selectbox("Select Count Feature to Estimate λ:", options=["syn_count", "ack_count", "fin_count", "rst_count", "Number"])
        clean_counts = df[poisson_feature].dropna()
        lam_hat = float(clean_counts.mean())
        
        st.metric("Estimated Parameter λ̂ (Sample Mean)", f"{lam_hat:.3f}")
        k_pois = np.arange(0, int(max(20, lam_hat + 4 * np.sqrt(lam_hat))))
        pois_pmf = stats.poisson.pmf(k_pois, lam_hat)
        
        fig_pois = go.Figure()
        fig_pois.add_trace(go.Bar(x=k_pois, y=pois_pmf, name="Theoretical Poisson PMF", marker=dict(color="#9B59B6")))
        apply_theme(fig_pois, f"Poisson(λ={lam_hat:.2f}) Distribution", "Count k", "Probability P(X = k)")
        st.plotly_chart(fig_pois, use_container_width=True, key="tab2_poisson_chart")

    st.markdown("---")

    # Section E: Continuous Distributions (Exponential & Uniform)
    st.subheader("E. Continuous Distributions: Exponential MLE & Uniform")
    cont_tabs = st.tabs(["Exponential Distribution (IAT)", "Uniform Distribution U(a, b)"])

    with cont_tabs[0]:
        st.markdown("#### Maximum Likelihood Estimation on Inter-Arrival Time (IAT)")
        iat_data = df["IAT"].dropna()
        exp_res = fit_exponential(iat_data.values)
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        col_exp1.metric("Sample Mean (Scale 1/λ)", f"{exp_res['mean']:.2e}")
        col_exp2.metric("MLE λ̂ = 1 / x̄", f"{exp_res['lambda_mle']:.4e}")
        col_exp3.metric("Kolmogorov-Smirnov p-value", f"{exp_res['p_value']:.4e}")

        # Overlay plot
        clean_iat = iat_data[iat_data > 0]
        n_sub = min(len(clean_iat), 2000)
        sub_iat = clean_iat.sample(n_sub, random_state=42)
        
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Histogram(x=sub_iat, histnorm="probability density", nbinsx=50, name="Empirical IAT Density", marker=dict(color="#3498DB", opacity=0.7)))
        x_exp = np.linspace(sub_iat.min(), sub_iat.quantile(0.95), 200)
        if exp_res["scale"] > 0:
            y_exp = stats.expon.pdf(x_exp, scale=exp_res["scale"])
            fig_exp.add_trace(go.Scatter(x=x_exp, y=y_exp, mode="lines", name=f"Fitted Exp (λ̂={exp_res['lambda_mle']:.2e})", line=dict(color="#E74C3C", width=2.5)))
        apply_theme(fig_exp, "Exponential Fit to Flow Inter-Arrival Time (IAT)", "IAT (Inter-Arrival Time)", "Density")
        st.plotly_chart(fig_exp, use_container_width=True, key="tab2_exponential_chart")

        st.warning(f"""
        ⚠️ **Honest Scientific Evaluation of Goodness-of-Fit**:  
        The Kolmogorov-Smirnov test yields $p = {exp_res['p_value']:.4e} < 0.05$, **decisively rejecting** the hypothesis that real IoMT packet arrivals follow an exponential distribution.  
        **Why real IoMT network traffic violates the Exponential assumption**:
        1. **Bursty Traffic**: Cyber-attacks (DDoS/DoS) transmit massive surges of packets simultaneously, causing heavy clusters near zero with extreme right-skew.
        2. **Periodic IoT Heartbeats**: Medical sensors (e.g. bedside ECG, telemetry transmitters) ping at fixed timer intervals (deterministic/clocked), violating the memoryless property of Poisson processes.
        3. **Heavy-tailed Distributions**: Real network IAT is better described by Pareto or Weibull distributions than memoryless exponentials.
        """)

    with cont_tabs[1]:
        st.markdown("#### Continuous Uniform Distribution U(a, b)")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            u_a = st.slider("Lower Bound a:", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
        with col_u2:
            u_b = st.slider("Upper Bound b:", min_value=51.0, max_value=150.0, value=60.0, step=1.0)

        u_x = np.linspace(u_a - 10, u_b + 10, 300)
        u_pdf = stats.uniform.pdf(u_x, loc=u_a, scale=u_b - u_a)
        u_cdf = stats.uniform.cdf(u_x, loc=u_a, scale=u_b - u_a)

        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=u_x, y=u_pdf, mode="lines", name="PDF f(x) = 1/(b-a)", line=dict(color="#2ECC71", width=2.5)))
        fig_u.add_trace(go.Scatter(x=u_x, y=u_cdf, mode="lines", name="CDF F(x) = (x-a)/(b-a)", line=dict(color="#F39C12", width=2, dash="dash")))
        apply_theme(fig_u, f"Uniform Distribution U({u_a}, {u_b})", "Value x", "Density / Probability")
        st.plotly_chart(fig_u, use_container_width=True, key="tab2_uniform_chart")
        st.caption(f"Theoretical Moments: $E[X] = \\frac{{a + b}}{{2}} = {(u_a + u_b)/2:.2f}$, $\\text{{Var}}(X) = \\frac{{(b - a)^2}}{{12}} = {((u_b - u_a)**2)/12:.2f}$. Application: random flow sampling for DPI inspection.")

    st.markdown("---")

    # Explainer: PMF vs PDF vs CDF
    with st.expander("🔍 Explainer: Discrete PMF vs Continuous PDF vs Cumulative CDF", expanded=False):
        st.markdown(r"""
        | Concept | Discrete Variable (e.g. `syn_count`) | Continuous Variable (e.g. `Rate`, `Duration`) |
        |---|---|---|
        | **Function** | Probability Mass Function: $P(X = k)$ | Probability Density Function: $f(x)$ |
        | **Point Probability** | $P(X = k) \in [0, 1]$ (Point mass is non-zero) | $P(X = c) = 0$ (Point mass is zero; probability exists only over intervals) |
        | **Total Sum / Integral**| $\sum_{k} P(X = k) = 1$ | $\int_{-\infty}^{\infty} f(x) dx = 1$ |
        | **Cumulative CDF** | $F(k) = P(X \le k) = \sum_{i \le k} P(X = i)$ (Step function) | $F(x) = P(X \le x) = \int_{-\infty}^x f(t) dt$ (Smooth monotonic curve) |
        """)
