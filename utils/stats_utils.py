"""
Statistical utilities module for the IoMT Network Intrusion Detection Dashboard.
Provides pure, robust, tested statistical functions adhering to university Probability & Statistics syllabus.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def calc_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Computes comprehensive descriptive statistics for a numerical series.
    Includes central tendency, dispersion, shape, and outlier detection.
    """
    clean_s = series.dropna()
    clean_s = clean_s[np.isfinite(clean_s)]
    
    n = len(clean_s)
    if n == 0:
        return {
            "count": 0, "mean": np.nan, "median": np.nan, "mode": np.nan,
            "variance": np.nan, "std": np.nan, "cv": np.nan,
            "skewness": np.nan, "kurtosis": np.nan, "iqr": np.nan,
            "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan,
            "outliers_count": 0, "outlier_percentage": 0.0
        }

    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    
    # Mode calculation (safely)
    mode_res = stats.mode(clean_s, keepdims=False)
    mode_val = float(mode_res.mode) if hasattr(mode_res, 'mode') else float(clean_s.mode().iloc[0]) if not clean_s.mode().empty else mean_val

    # Sample variance with Bessel's correction (ddof=1)
    var_val = float(clean_s.var(ddof=1)) if n > 1 else 0.0
    std_val = float(clean_s.std(ddof=1)) if n > 1 else 0.0
    
    # Coefficient of Variation: (SD / |Mean|) * 100
    cv_val = float((std_val / abs(mean_val)) * 100.0) if mean_val != 0 else np.nan

    # Skewness and Fisher Kurtosis (normal = 0)
    skew_val = float(stats.skew(clean_s)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(clean_s)) if n > 3 else 0.0

    # Quartiles & IQR
    q25 = float(clean_s.quantile(0.25))
    q75 = float(clean_s.quantile(0.75))
    iqr_val = float(q75 - q25)

    # 1.5 * IQR Rule Outliers
    lower_bound = q25 - 1.5 * iqr_val
    upper_bound = q75 + 1.5 * iqr_val
    outliers_mask = (clean_s < lower_bound) | (clean_s > upper_bound)
    outliers_count = int(outliers_mask.sum())
    outlier_pct = float((outliers_count / n) * 100.0)

    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": var_val,
        "std": std_val,
        "cv": cv_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "iqr": iqr_val,
        "q25": q25,
        "q75": q75,
        "min": float(clean_s.min()),
        "max": float(clean_s.max()),
        "outliers_count": outliers_count,
        "outlier_percentage": outlier_pct,
        "lower_fence": lower_bound,
        "upper_fence": upper_bound
    }


def calc_class_frequency_table(series: pd.Series) -> pd.DataFrame:
    """
    Computes frequency distribution: Count, Relative Frequency (%), Cumulative Frequency (%).
    """
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = ["Class", "Count"]
    total = counts["Count"].sum()
    counts["Relative Frequency (%)"] = (counts["Count"] / total) * 100.0
    counts["Cumulative Frequency (%)"] = counts["Relative Frequency (%)"].cumsum()
    return counts


def classify_features(feature_names: List[str]) -> pd.DataFrame:
    """
    Classifies all IoMT features into Qualitative vs Quantitative with statistical scale and IoMT context.
    """
    records = []
    
    # Pre-defined domain knowledge for CICIoMT2024 features
    nominal_features = {
        "Protocol Type": "Nominal (numeric-coded IANA protocol: 6=TCP, 17=UDP, 1=ICMP)",
        "label": "Nominal (specific attack scenario or benign capture)",
        "canonical_label": "Nominal (collapsed attack scenario family)",
        "category": "Nominal (top-level 6-class target: Benign + 5 attack families)",
        "binary_label": "Nominal (Binary: Benign vs Attack)"
    }
    
    bernoulli_flags = {
        "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
        "ack_flag_number", "ece_flag_number", "cwr_flag_number",
        "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC"
    }

    discrete_counts = {
        "ack_count", "syn_count", "fin_count", "rst_count", "Number"
    }

    for name in feature_names:
        if name in nominal_features:
            cat = "Qualitative"
            subcat = "Nominal"
            scale = "Nominal"
            notes = nominal_features[name]
        elif name in bernoulli_flags:
            cat = "Qualitative (Indicator)"
            subcat = "Bernoulli / Binary Indicator"
            scale = "Nominal (0/1 binary indicator)"
            notes = "Binary network header flag or protocol presence indicator"
        elif name in discrete_counts:
            cat = "Quantitative"
            subcat = "Discrete"
            scale = "Ratio"
            notes = "Count of packets or TCP flag occurrences within flow window"
        else:
            cat = "Quantitative"
            subcat = "Continuous"
            scale = "Ratio"
            notes = "Flow rate, duration, size, or statistical window summary"
            
        records.append({
            "Feature Name": name,
            "Variable Type": cat,
            "Sub-classification": subcat,
            "Measurement Scale": scale,
            "IoMT Security Role": notes
        })
        
    return pd.DataFrame(records)


def bayes_theorem(prior_attack: float, sensitivity: float, specificity: float) -> Dict[str, float]:
    """
    Calculates posterior probability P(Attack | Alert) via Bayes' Theorem.
    Demonstrates base-rate fallacy / base-rate effect in intrusion detection.
    """
    # P(Alert | Attack) = sensitivity (TPR)
    # P(Alert | Benign) = 1 - specificity (FPR)
    # P(Benign) = 1 - prior_attack
    p_attack = float(prior_attack)
    p_benign = 1.0 - p_attack
    
    tpr = float(sensitivity)
    fpr = 1.0 - float(specificity)
    
    # Total probability of an alert
    p_alert = (tpr * p_attack) + (fpr * p_benign)
    
    if p_alert == 0:
        posterior = 0.0
    else:
        posterior = (tpr * p_attack) / p_alert
        
    p_benign_given_alert = 1.0 - posterior  # False discovery rate
    
    return {
        "prior_attack": p_attack,
        "prior_benign": p_benign,
        "sensitivity": tpr,
        "specificity": float(specificity),
        "fpr": fpr,
        "p_alert": p_alert,
        "posterior_attack_given_alert": posterior,
        "posterior_benign_given_alert": p_benign_given_alert
    }


def fit_exponential(data: np.ndarray) -> Dict[str, Any]:
    """
    Fits Exponential distribution via MLE: lambda_hat = 1 / mean(X).
    Performs Kolmogorov-Smirnov test to assess goodness of fit.
    """
    clean_data = data[np.isfinite(data)]
    clean_data = clean_data[clean_data > 0]
    
    if len(clean_data) < 5:
        return {"lambda_mle": np.nan, "scale": np.nan, "ks_stat": np.nan, "p_value": np.nan}
        
    mean_val = float(np.mean(clean_data))
    lambda_hat = 1.0 / mean_val if mean_val > 0 else np.nan
    scale = mean_val
    
    # Subsample for KS test if dataset is very large to maintain responsiveness
    sample_size = min(len(clean_data), 5000)
    subsample = np.random.choice(clean_data, size=sample_size, replace=False) if len(clean_data) > sample_size else clean_data
    
    ks_res = stats.kstest(subsample, 'expon', args=(0, scale))
    
    return {
        "lambda_mle": lambda_hat,
        "scale": scale,
        "mean": mean_val,
        "variance": mean_val ** 2,
        "ks_stat": float(ks_res.statistic),
        "p_value": float(ks_res.pvalue),
        "is_good_fit": bool(ks_res.pvalue > 0.05)
    }


def linear_transformation_stats(data: np.ndarray, a: float, b: float) -> Dict[str, float]:
    """
    Demonstrates linear transformation properties:
    E[aX + b] = a*E[X] + b
    Var(aX + b) = a^2 * Var(X)
    """
    clean_data = data[np.isfinite(data)]
    e_x = float(np.mean(clean_data))
    var_x = float(np.var(clean_data, ddof=1)) if len(clean_data) > 1 else 0.0
    
    transformed = a * clean_data + b
    e_transformed = float(np.mean(transformed))
    var_transformed = float(np.var(transformed, ddof=1)) if len(clean_data) > 1 else 0.0
    
    theoretical_e = a * e_x + b
    theoretical_var = (a ** 2) * var_x
    
    return {
        "e_x": e_x,
        "var_x": var_x,
        "a": a,
        "b": b,
        "empirical_transformed_mean": e_transformed,
        "theoretical_transformed_mean": theoretical_e,
        "empirical_transformed_var": var_transformed,
        "theoretical_transformed_var": theoretical_var,
        "mean_diff": abs(e_transformed - theoretical_e),
        "var_diff": abs(var_transformed - theoretical_var)
    }


def fit_normal_and_tests(data: np.ndarray) -> Dict[str, Any]:
    """
    Fits Normal distribution and conducts Shapiro-Wilk and D'Agostino normality tests.
    """
    clean_data = data[np.isfinite(data)]
    n = len(clean_data)
    if n < 8:
        return {"mean": np.nan, "std": np.nan, "shapiro_stat": np.nan, "shapiro_p": np.nan,
                "dagostino_stat": np.nan, "dagostino_p": np.nan, "verdict": "Insufficient data"}
                
    mu = float(np.mean(clean_data))
    sigma = float(np.std(clean_data, ddof=1))
    
    # Shapiro-Wilk test works for N <= 5000
    shapiro_n = min(n, 5000)
    shapiro_sample = clean_data if n <= 5000 else np.random.choice(clean_data, size=shapiro_n, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)
    
    # D'Agostino-Pearson omnibus test
    dagostino_stat, dagostino_p = stats.normaltest(clean_data)
    
    if dagostino_p > 0.05 and shapiro_p > 0.05:
        verdict = "Normal (Cannot reject H0 of normality)"
    elif dagostino_p > 0.01:
        verdict = "Mildly Non-Normal (Deviates slightly from Gaussian)"
    else:
        verdict = "Heavily Non-Normal (Significant skewness/kurtosis; non-Gaussian)"

    return {
        "mean": mu,
        "std": sigma,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "dagostino_stat": float(dagostino_stat),
        "dagostino_p": float(dagostino_p),
        "verdict": verdict
    }


def quality_control_limits(data: np.ndarray, sigma_multiplier: float = 3.0) -> Dict[str, float]:
    """
    Computes Shewhart X-bar style statistical process control limits:
    Center Line (CL) = mean
    UCL = mean + 3*sigma
    LCL = max(0, mean - 3*sigma)
    """
    clean_data = data[np.isfinite(data)]
    mu = float(np.mean(clean_data))
    sigma = float(np.std(clean_data, ddof=1))
    
    ucl = mu + sigma_multiplier * sigma
    lcl = max(0.0, mu - sigma_multiplier * sigma)
    
    return {
        "cl": mu,
        "sigma": sigma,
        "ucl": ucl,
        "lcl": lcl
    }


def clt_simulation(data: np.ndarray, sample_size: int, num_samples: int, seed: int = 42) -> Dict[str, Any]:
    """
    Demonstrates Central Limit Theorem by repeatedly drawing random samples of size n.
    Compares empirical standard deviation of sample means with theoretical SE = sigma / sqrt(n).
    """
    clean_data = data[np.isfinite(data)]
    rng = np.random.default_rng(seed)
    
    pop_mean = float(np.mean(clean_data))
    pop_std = float(np.std(clean_data, ddof=1))
    theoretical_se = pop_std / np.sqrt(sample_size)
    
    # Draw repeated samples
    sample_means = np.zeros(num_samples)
    for i in range(num_samples):
        batch = rng.choice(clean_data, size=sample_size, replace=True)
        sample_means[i] = np.mean(batch)
        
    mean_of_means = float(np.mean(sample_means))
    empirical_se = float(np.std(sample_means, ddof=1))
    
    return {
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "sample_size": sample_size,
        "num_samples": num_samples,
        "theoretical_se": theoretical_se,
        "empirical_se": empirical_se,
        "mean_of_means": mean_of_means,
        "sample_means": sample_means
    }


def simulate_variance_bias(data: np.ndarray, sample_size: int = 5, num_trials: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """
    Demonstrates why Bessel's correction (n-1) is necessary for unbiased variance estimation.
    Compares average of S_n^2 (biased, divided by n) vs s^2 (unbiased, divided by n-1).
    """
    clean_data = data[np.isfinite(data)]
    rng = np.random.default_rng(seed)
    
    pop_var = float(np.var(clean_data, ddof=0))
    
    biased_vars = np.zeros(num_trials)
    unbiased_vars = np.zeros(num_trials)
    
    for i in range(num_trials):
        sample = rng.choice(clean_data, size=sample_size, replace=True)
        biased_vars[i] = np.var(sample, ddof=0)
        unbiased_vars[i] = np.var(sample, ddof=1)
        
    mean_biased = float(np.mean(biased_vars))
    mean_unbiased = float(np.mean(unbiased_vars))
    
    # Expected biased variance = ((n-1)/n) * sigma^2
    theoretical_biased_expected = ((sample_size - 1) / sample_size) * pop_var
    
    return {
        "true_pop_var": pop_var,
        "sample_size": sample_size,
        "num_trials": num_trials,
        "mean_biased_var": mean_biased,
        "mean_unbiased_var": mean_unbiased,
        "theoretical_biased_expected": theoretical_biased_expected,
        "bias_percentage": ((mean_biased - pop_var) / pop_var) * 100.0 if pop_var > 0 else 0.0
    }


def confidence_interval_mean(data: np.ndarray, confidence: float = 0.95, sigma_known: Optional[float] = None) -> Dict[str, Any]:
    """
    Computes Confidence Interval for the mean:
    - Z-interval if sigma is known
    - t-interval if sigma is unknown (Student's t, df = n - 1)
    """
    clean_data = data[np.isfinite(data)]
    n = len(clean_data)
    if n < 2:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "margin_of_error": np.nan, "method": "none"}
        
    x_bar = float(np.mean(clean_data))
    alpha = 1.0 - confidence
    
    if sigma_known is not None:
        se = sigma_known / np.sqrt(n)
        crit_val = float(stats.norm.ppf(1.0 - alpha / 2.0))
        method = "Z-interval (sigma known)"
    else:
        s = float(np.std(clean_data, ddof=1))
        se = s / np.sqrt(n)
        crit_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        method = "t-interval (sigma unknown, df=n-1)"
        
    margin = crit_val * se
    return {
        "mean": x_bar,
        "confidence": confidence,
        "se": se,
        "critical_value": crit_val,
        "margin_of_error": margin,
        "ci_lower": x_bar - margin,
        "ci_upper": x_bar + margin,
        "method": method
    }


def confidence_interval_proportion(successes: int, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Ideal for rare categories like Spoofing precision / recall.
    """
    if n == 0:
        return {"p_hat": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin_of_error": 0.0}
        
    p_hat = successes / n
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    spread = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))) / denominator
    
    ci_lower = max(0.0, float(center - spread))
    ci_upper = min(1.0, float(center + spread))
    
    return {
        "p_hat": float(p_hat),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": float(spread)
    }


def one_sample_test(data: np.ndarray, mu_0: float, alpha: float = 0.05, test_type: str = "t", alternative: str = "two-sided") -> Dict[str, Any]:
    """
    Performs One-Sample Hypothesis Test (Z-test or t-test).
    alternative: 'two-sided', 'greater' (right-tailed), or 'less' (left-tailed).
    """
    clean_data = data[np.isfinite(data)]
    n = len(clean_data)
    x_bar = float(np.mean(clean_data))
    s = float(np.std(clean_data, ddof=1))
    se = s / np.sqrt(n)
    
    if se == 0:
        return {"stat": 0.0, "p_value": 1.0, "decision": "Variance is zero"}
        
    stat = (x_bar - mu_0) / se
    df = n - 1
    
    if test_type == "z":
        dist = stats.norm
        crit_kwargs = {}
    else:
        dist = stats.t
        crit_kwargs = {"df": df}
        
    if alternative == "two-sided":
        crit_val = float(dist.ppf(1.0 - alpha / 2.0, **crit_kwargs))
        p_val = float(2.0 * (1.0 - dist.cdf(abs(stat), **crit_kwargs)))
        reject = abs(stat) > crit_val
        critical_interval = (-crit_val, crit_val)
    elif alternative == "greater":
        crit_val = float(dist.ppf(1.0 - alpha, **crit_kwargs))
        p_val = float(1.0 - dist.cdf(stat, **crit_kwargs))
        reject = stat > crit_val
        critical_interval = (crit_val, np.inf)
    else:  # less
        crit_val = float(dist.ppf(alpha, **crit_kwargs))
        p_val = float(dist.cdf(stat, **crit_kwargs))
        reject = stat < crit_val
        critical_interval = (-np.inf, crit_val)
        
    decision = "Reject H0 (Statistically Significant)" if reject else "Fail to Reject H0 (Insufficient Evidence)"
    
    return {
        "sample_mean": x_bar,
        "hypothesized_mean": mu_0,
        "sample_size": n,
        "stat": float(stat),
        "df": df if test_type == "t" else None,
        "p_value": float(p_val),
        "alpha": alpha,
        "critical_value": crit_val,
        "critical_interval": critical_interval,
        "alternative": alternative,
        "test_type": test_type,
        "reject_null": bool(reject),
        "decision": decision
    }


def two_sample_welch(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    Performs Two-Sample Welch's t-test (unequal variances assumed) and computes Cohen's d effect size.
    """
    g1 = group1[np.isfinite(group1)]
    g2 = group2[np.isfinite(group2)]
    
    n1, n2 = len(g1), len(g2)
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1 = float(np.var(g1, ddof=1)) if n1 > 1 else 0.0
    v2 = float(np.var(g2, ddof=1)) if n2 > 1 else 0.0
    
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    
    # Welch-Satterthwaite degrees of freedom
    if (v1 / n1 + v2 / n2) > 0:
        df = ((v1 / n1 + v2 / n2) ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))
    else:
        df = 1.0
        
    # Pooled standard deviation for Cohen's d
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2) > 2 else 1.0
    cohens_d = (m1 - m2) / s_pooled if s_pooled > 0 else 0.0
    
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_interp = "Negligible effect size"
    elif abs_d < 0.5:
        d_interp = "Small effect size"
    elif abs_d < 0.8:
        d_interp = "Medium effect size"
    else:
        d_interp = "Large effect size"

    return {
        "n1": n1, "mean1": m1, "std1": np.sqrt(v1),
        "n2": n2, "mean2": m2, "std2": np.sqrt(v2),
        "mean_diff": m1 - m2,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "df": float(df),
        "cohens_d": float(cohens_d),
        "d_interpretation": d_interp
    }


def type_errors_power(mu0: float, mu1: float, sigma: float, n: int, alpha: float = 0.05, alternative: str = "greater") -> Dict[str, float]:
    """
    Computes Type I error (alpha), Type II error (beta), and statistical Power (1 - beta)
    for testing H0: mu = mu0 vs H1: mu = mu1.
    """
    se = sigma / np.sqrt(n)
    
    if alternative == "greater":
        crit_z = stats.norm.ppf(1.0 - alpha)
        x_crit = mu0 + crit_z * se
        # Beta: probability of falling in the non-rejection region under H1
        beta = float(stats.norm.cdf((x_crit - mu1) / se))
    elif alternative == "less":
        crit_z = stats.norm.ppf(alpha)
        x_crit = mu0 + crit_z * se
        beta = float(1.0 - stats.norm.cdf((x_crit - mu1) / se))
    else:  # two-sided
        crit_z = stats.norm.ppf(1.0 - alpha / 2.0)
        x_crit_high = mu0 + crit_z * se
        x_crit_low = mu0 - crit_z * se
        beta = float(stats.norm.cdf((x_crit_high - mu1) / se) - stats.norm.cdf((x_crit_low - mu1) / se))
        x_crit = x_crit_high
        
    power = 1.0 - beta
    return {
        "mu0": mu0,
        "mu1": mu1,
        "sigma": sigma,
        "n": n,
        "se": se,
        "alpha": alpha,
        "beta": beta,
        "power": power,
        "x_crit": x_crit
    }


def chi_square_gof(observed: pd.Series, expected_proportions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Performs Chi-Square Goodness-of-Fit test against uniform or custom expected distribution.
    """
    obs_counts = observed.value_counts()
    categories = obs_counts.index.tolist()
    k = len(categories)
    total_n = obs_counts.sum()
    
    if expected_proportions is None:
        # Uniform expectation
        exp_counts = np.full(k, total_n / k)
    else:
        exp_counts = np.array([expected_proportions.get(cat, 0.0) * total_n for cat in categories])
        
    chi2_stat, p_val = stats.chisquare(obs_counts.values, f_exp=exp_counts)
    
    df_results = pd.DataFrame({
        "Category": categories,
        "Observed (O)": obs_counts.values,
        "Expected (E)": exp_counts,
        "Residual (O - E)": obs_counts.values - exp_counts,
        "Chi2 Component ((O-E)^2 / E)": ((obs_counts.values - exp_counts) ** 2) / exp_counts
    })
    
    return {
        "table": df_results,
        "chi2_stat": float(chi2_stat),
        "df": k - 1,
        "p_value": float(p_val),
        "reject_null": bool(p_val < 0.05)
    }


def chi_square_independence(df: pd.DataFrame, col_x: str, col_y: str) -> Dict[str, Any]:
    """
    Performs Chi-Square Test of Independence between two categorical variables.
    Computes contingency table, expected frequencies, chi2, Cramér's V, and standardized residuals.
    """
    contingency = pd.crosstab(df[col_x], df[col_y])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    n = contingency.values.sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    
    # Standardized Pearson residuals: (O - E) / sqrt(E)
    std_residuals = (contingency - expected_df) / np.sqrt(expected_df)
    
    return {
        "contingency": contingency,
        "expected": expected_df,
        "std_residuals": std_residuals,
        "chi2_stat": float(chi2),
        "df": int(dof),
        "p_value": float(p_val),
        "cramers_v": float(cramers_v),
        "reject_null": bool(p_val < 0.05)
    }


def one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """
    Conducts One-Way ANOVA, Levene's test for equal variance, Kruskal-Wallis non-parametric test,
    and Tukey's HSD post-hoc analysis.
    """
    clean_df = df[[group_col, value_col]].dropna()
    clean_df = clean_df[np.isfinite(clean_df[value_col])]
    
    groups = [group[value_col].values for _, group in clean_df.groupby(group_col)]
    group_names = [str(name) for name, _ in clean_df.groupby(group_col)]
    k = len(groups)
    n_total = len(clean_df)
    
    if k < 2 or n_total <= k:
        return {"error": "Insufficient groups or data"}
        
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Levene's test for homogeneity of variance
    levene_stat, levene_p = stats.levene(*groups)
    
    # Kruskal-Wallis test (non-parametric fallback)
    kw_stat, kw_p = stats.kruskal(*groups)
    
    # ANOVA Table components
    grand_mean = clean_df[value_col].mean()
    ssb = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups)
    ssw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
    sst = ssb + ssw
    
    df_b = k - 1
    df_w = n_total - k
    df_t = n_total - 1
    
    msb = ssb / df_b if df_b > 0 else 0.0
    msw = ssw / df_w if df_w > 0 else 0.0
    
    eta_squared = ssb / sst if sst > 0 else 0.0
    
    anova_table = pd.DataFrame({
        "Source of Variation": ["Between Groups (SSB)", "Within Groups (SSW)", "Total (SST)"],
        "Sum of Squares (SS)": [ssb, ssw, sst],
        "Degrees of Freedom (df)": [df_b, df_w, df_t],
        "Mean Square (MS)": [msb, msw, np.nan],
        "F-Statistic": [f_stat, np.nan, np.nan],
        "p-value": [p_val, np.nan, np.nan]
    })
    
    # Tukey HSD post-hoc (subsample if n_total > 5000 to keep fast)
    if n_total > 5000:
        sub_sample = clean_df.groupby(group_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), 500), random_state=42)
        )
        tukey = pairwise_tukeyhsd(sub_sample[value_col], sub_sample[group_col], alpha=0.05)
    else:
        tukey = pairwise_tukeyhsd(clean_df[value_col], clean_df[group_col], alpha=0.05)
        
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    return {
        "anova_table": anova_table,
        "f_stat": float(f_stat),
        "p_val": float(p_val),
        "eta_squared": float(eta_squared),
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "kruskal_stat": float(kw_stat),
        "kruskal_p": float(kw_p),
        "tukey_df": tukey_df
    }


def linear_regression_two_lines(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Computes Simple Linear Regression with:
    - Line of Y on X: Y - y_bar = b_yx * (X - x_bar)
    - Line of X on Y: X - x_bar = b_xy * (Y - y_bar)
    - Verifies r^2 = b_yx * b_xy
    - Standard error of estimate, residuals, and prediction intervals
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_c = x[mask]
    y_c = y[mask]
    n = len(x_c)
    
    if n < 3:
        return {"error": "Need at least 3 points"}
        
    x_bar = float(np.mean(x_c))
    y_bar = float(np.mean(y_c))
    
    s_x = float(np.std(x_c, ddof=1))
    s_y = float(np.std(y_c, ddof=1))
    
    # Scipy linear regression (Y on X)
    slope_yx, intercept_yx, r_val, p_val, std_err_slope = stats.linregress(x_c, y_c)
    
    # Regression of X on Y
    slope_xy, intercept_xy, _, _, _ = stats.linregress(y_c, x_c)
    
    r_squared = r_val ** 2
    b_product = slope_yx * slope_xy
    
    # Fitted values and residuals for Y on X
    y_pred = intercept_yx + slope_yx * x_c
    residuals = y_c - y_pred
    
    # Standard error of the estimate: s_e = sqrt( sum(e^2) / (n - 2) )
    ss_res = np.sum(residuals ** 2)
    s_e = np.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
    
    ss_tot = np.sum((y_c - y_bar) ** 2)
    
    return {
        "n": n,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "s_x": s_x,
        "s_y": s_y,
        "slope_yx": float(slope_yx),
        "intercept_yx": float(intercept_yx),
        "slope_xy": float(slope_xy),
        "intercept_xy": float(intercept_xy),
        "r": float(r_val),
        "r_squared": float(r_squared),
        "b_product": float(b_product),
        "p_value": float(p_val),
        "std_err_slope": float(std_err_slope),
        "std_err_estimate": float(s_e),
        "residuals": residuals,
        "y_pred": y_pred,
        "ss_res": float(ss_res),
        "ss_tot": float(ss_tot)
    }


def predict_with_intervals(x_new: float, x_data: np.ndarray, y_data: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """
    Computes point prediction along with 95% Confidence Interval (mean response)
    and 95% Prediction Interval (individual observation) at X = x_new.
    """
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_c = x_data[mask]
    y_c = y_data[mask]
    n = len(x_c)
    
    res = linear_regression_two_lines(x_c, y_c)
    b0 = res["intercept_yx"]
    b1 = res["slope_yx"]
    x_bar = res["x_bar"]
    s_e = res["std_err_estimate"]
    
    y_hat = b0 + b1 * x_new
    
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 2))
    ss_xx = np.sum((x_c - x_bar) ** 2)
    
    # Margin for Confidence Interval of Mean Response
    ci_margin = t_crit * s_e * np.sqrt((1.0 / n) + ((x_new - x_bar) ** 2) / ss_xx) if ss_xx > 0 else 0.0
    
    # Margin for Prediction Interval of Individual Observation
    pi_margin = t_crit * s_e * np.sqrt(1.0 + (1.0 / n) + ((x_new - x_bar) ** 2) / ss_xx) if ss_xx > 0 else 0.0
    
    return {
        "x_new": x_new,
        "y_hat": float(y_hat),
        "ci_lower": float(y_hat - ci_margin),
        "ci_upper": float(y_hat + ci_margin),
        "pi_lower": float(y_hat - pi_margin),
        "pi_upper": float(y_hat + pi_margin),
        "ci_margin": float(ci_margin),
        "pi_margin": float(pi_margin)
    }


def spearman_rank_worked(x: np.ndarray, y: np.ndarray, sample_n: int = 10) -> Dict[str, Any]:
    """
    Produces a worked, step-by-step Spearman rank table and formula calculation.
    rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_sub = x[mask][:sample_n]
    y_sub = y[mask][:sample_n]
    n = len(x_sub)
    
    rank_x = stats.rankdata(x_sub)
    rank_y = stats.rankdata(y_sub)
    
    d = rank_x - rank_y
    d_sq = d ** 2
    sum_d_sq = float(np.sum(d_sq))
    
    # Manual Spearman formula
    rho_manual = 1.0 - (6.0 * sum_d_sq) / (n * (n**2 - 1)) if n > 1 else 1.0
    
    # Scipy verification
    scipy_res = stats.spearmanr(x_sub, y_sub)
    
    table_df = pd.DataFrame({
        "i": np.arange(1, n + 1),
        "X": x_sub,
        "Y": y_sub,
        "Rank(X)": rank_x,
        "Rank(Y)": rank_y,
        "d_i = Rank(X) - Rank(Y)": d,
        "d_i^2": d_sq
    })
    
    return {
        "table": table_df,
        "n": n,
        "sum_d_sq": sum_d_sq,
        "rho_manual": float(rho_manual),
        "rho_scipy": float(scipy_res.correlation),
        "p_value": float(scipy_res.pvalue)
    }


def classification_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """
    Computes per-class Type I error (FPR) and Type II error (FNR) with 95% Wilson Confidence Intervals.
    Also computes overall Macro vs Weighted metrics.
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    per_class_results = []
    
    total_samples = len(y_true)
    for i, cls_name in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total_samples - (tp + fn + fp)
        
        # Type I error rate = FPR = FP / (FP + TN)
        fp_denom = fp + tn
        fpr = fp / fp_denom if fp_denom > 0 else 0.0
        fpr_ci = confidence_interval_proportion(fp, fp_denom, 0.95)
        
        # Type II error rate = FNR = FN / (FN + TP)
        fn_denom = fn + tp
        fnr = fn / fn_denom if fn_denom > 0 else 0.0
        fnr_ci = confidence_interval_proportion(fn, fn_denom, 0.95)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_results.append({
            "Class": cls_name,
            "Support": int(fn_denom),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "Type I Error (FPR)": float(fpr),
            "FPR 95% CI": f"[{fpr_ci['ci_lower']:.4f}, {fpr_ci['ci_upper']:.4f}]",
            "Type II Error (FNR)": float(fnr),
            "FNR 95% CI": f"[{fnr_ci['ci_lower']:.4f}, {fnr_ci['ci_upper']:.4f}]"
        })
        
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    return {
        "confusion_matrix": cm,
        "classes": classes,
        "per_class_df": pd.DataFrame(per_class_results),
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {})
    }
