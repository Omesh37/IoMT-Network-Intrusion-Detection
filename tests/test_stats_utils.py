"""
Unit tests for utils/stats_utils.py.
Validates statistical implementations against standard SciPy and Statsmodels outputs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from utils.stats_utils import (
    calc_descriptive_stats,
    calc_class_frequency_table,
    classify_features,
    bayes_theorem,
    fit_exponential,
    linear_transformation_stats,
    fit_normal_and_tests,
    quality_control_limits,
    clt_simulation,
    simulate_variance_bias,
    confidence_interval_mean,
    confidence_interval_proportion,
    one_sample_test,
    two_sample_welch,
    type_errors_power,
    chi_square_gof,
    chi_square_independence,
    one_way_anova,
    linear_regression_two_lines,
    predict_with_intervals,
    spearman_rank_worked,
    classification_error_analysis
)


@pytest.fixture
def sample_numeric_series():
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=100.0, scale=15.0, size=500))


def test_calc_descriptive_stats(sample_numeric_series):
    res = calc_descriptive_stats(sample_numeric_series)
    
    assert np.isclose(res["mean"], sample_numeric_series.mean())
    assert np.isclose(res["median"], sample_numeric_series.median())
    assert np.isclose(res["variance"], sample_numeric_series.var(ddof=1))
    assert np.isclose(res["std"], sample_numeric_series.std(ddof=1))
    assert np.isclose(res["cv"], (sample_numeric_series.std(ddof=1) / sample_numeric_series.mean()) * 100.0)
    assert np.isclose(res["skewness"], stats.skew(sample_numeric_series), atol=1e-3)
    assert np.isclose(res["kurtosis"], stats.kurtosis(sample_numeric_series), atol=1e-3)
    assert res["count"] == 500
    assert res["outliers_count"] >= 0


def test_calc_class_frequency_table():
    classes = pd.Series(["Benign"] * 60 + ["DDoS"] * 30 + ["Spoofing"] * 10)
    df = calc_class_frequency_table(classes)
    
    assert len(df) == 3
    assert np.isclose(df["Relative Frequency (%)"].sum(), 100.0)
    assert np.isclose(df["Cumulative Frequency (%)"].iloc[-1], 100.0)


def test_classify_features():
    cols = ["Protocol Type", "Rate", "syn_flag_number", "ack_count", "category"]
    df = classify_features(cols)
    assert len(df) == 5
    proto_row = df[df["Feature Name"] == "Protocol Type"].iloc[0]
    assert proto_row["Variable Type"] == "Qualitative"
    flag_row = df[df["Feature Name"] == "syn_flag_number"].iloc[0]
    assert "Bernoulli" in flag_row["Sub-classification"]


def test_bayes_theorem():
    # Base rate fallacy scenario: Prior = 0.01, Sensitivity = 0.99, Specificity = 0.95
    res = bayes_theorem(prior_attack=0.01, sensitivity=0.99, specificity=0.95)
    
    # P(Alert) = 0.99*0.01 + 0.05*0.99 = 0.0099 + 0.0495 = 0.0594
    # P(Attack|Alert) = 0.0099 / 0.0594 ≈ 0.1667 (only ~16.7%!)
    expected_posterior = (0.99 * 0.01) / (0.99 * 0.01 + 0.05 * 0.99)
    assert np.isclose(res["posterior_attack_given_alert"], expected_posterior)
    assert res["posterior_attack_given_alert"] < 0.20


def test_fit_exponential():
    np.random.seed(42)
    # Scale = 10 -> mean = 10, lambda = 0.1
    data = np.random.exponential(scale=10.0, size=1000)
    res = fit_exponential(data)
    
    assert np.isclose(res["scale"], 10.0, atol=1.0)
    assert np.isclose(res["lambda_mle"], 0.1, atol=0.02)
    assert res["p_value"] > 0.01  # Should fit exponential data well


def test_linear_transformation():
    np.random.seed(42)
    x = np.random.normal(50, 10, size=200)
    a, b = 2.5, 15.0
    res = linear_transformation_stats(x, a, b)
    
    assert np.isclose(res["empirical_transformed_mean"], res["theoretical_transformed_mean"])
    assert np.isclose(res["empirical_transformed_var"], res["theoretical_transformed_var"])


def test_fit_normal_and_tests():
    np.random.seed(42)
    x = np.random.normal(0, 1, size=500)
    res = fit_normal_and_tests(x)
    
    assert "Normal" in res["verdict"]
    assert res["shapiro_p"] > 0.05
    assert res["dagostino_p"] > 0.05


def test_quality_control_limits():
    data = np.array([10.0, 12.0, 9.0, 11.0, 10.5])
    res = quality_control_limits(data, sigma_multiplier=3.0)
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    assert np.isclose(res["cl"], mean)
    assert np.isclose(res["ucl"], mean + 3 * std)
    assert np.isclose(res["lcl"], max(0.0, mean - 3 * std))


def test_clt_simulation():
    # Heavily skewed exponential data
    np.random.seed(42)
    data = np.random.exponential(scale=5.0, size=2000)
    res = clt_simulation(data, sample_size=30, num_samples=1000, seed=42)
    
    assert np.isclose(res["mean_of_means"], res["pop_mean"], atol=0.3)
    assert np.isclose(res["empirical_se"], res["theoretical_se"], atol=0.1)


def test_simulate_variance_bias():
    np.random.seed(42)
    data = np.random.normal(0, 5, size=1000)
    res = simulate_variance_bias(data, sample_size=5, num_trials=2000, seed=42)
    
    # Biased variance must underestimate population variance on average
    assert res["mean_biased_var"] < res["true_pop_var"]
    # Unbiased variance should be very close to true population variance
    assert abs(res["mean_unbiased_var"] - res["true_pop_var"]) < abs(res["mean_biased_var"] - res["true_pop_var"])


def test_confidence_interval_mean():
    np.random.seed(42)
    data = np.random.normal(10, 2, size=100)
    
    # t-interval
    res_t = confidence_interval_mean(data, confidence=0.95)
    scipy_ci = stats.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    assert np.isclose(res_t["ci_lower"], scipy_ci[0])
    assert np.isclose(res_t["ci_upper"], scipy_ci[1])
    
    # z-interval
    res_z = confidence_interval_mean(data, confidence=0.95, sigma_known=2.0)
    scipy_ci_z = stats.norm.interval(0.95, loc=np.mean(data), scale=2.0/np.sqrt(len(data)))
    assert np.isclose(res_z["ci_lower"], scipy_ci_z[0])
    assert np.isclose(res_z["ci_upper"], scipy_ci_z[1])


def test_confidence_interval_proportion():
    res = confidence_interval_proportion(successes=82, n=100, confidence=0.95)
    assert 0.0 <= res["ci_lower"] <= res["p_hat"] <= res["ci_upper"] <= 1.0


def test_one_sample_test():
    np.random.seed(42)
    data = np.random.normal(10.0, 2.0, size=50)
    res = one_sample_test(data, mu_0=10.0, alpha=0.05, test_type="t", alternative="two-sided")
    
    scipy_t = stats.ttest_1samp(data, popmean=10.0)
    assert np.isclose(res["stat"], scipy_t.statistic)
    assert np.isclose(res["p_value"], scipy_t.pvalue)
    assert not res["reject_null"]


def test_two_sample_welch():
    np.random.seed(42)
    g1 = np.random.normal(10.0, 2.0, size=60)
    g2 = np.random.normal(15.0, 4.0, size=80)
    
    res = two_sample_welch(g1, g2)
    scipy_res = stats.ttest_ind(g1, g2, equal_var=False)
    
    assert np.isclose(res["t_stat"], scipy_res.statistic)
    assert np.isclose(res["p_value"], scipy_res.pvalue)
    assert res["cohens_d"] < -1.0  # Large difference


def test_type_errors_power():
    res = type_errors_power(mu0=0.0, mu1=1.0, sigma=1.0, n=25, alpha=0.05, alternative="greater")
    assert np.isclose(res["alpha"], 0.05)
    assert 0.0 <= res["beta"] <= 1.0
    assert np.isclose(res["power"], 1.0 - res["beta"])
    assert res["power"] > 0.90  # Large effect size and n=25 gives high power


def test_chi_square_gof():
    observed = pd.Series(["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25)
    res = chi_square_gof(observed)
    
    assert res["chi2_stat"] == 0.0
    assert res["p_value"] == 1.0
    assert not res["reject_null"]


def test_chi_square_independence():
    df = pd.DataFrame({
        "Protocol": ["TCP"] * 50 + ["UDP"] * 50,
        "Category": ["Benign"] * 40 + ["DDoS"] * 10 + ["Benign"] * 10 + ["DDoS"] * 40
    })
    res = chi_square_independence(df, "Protocol", "Category")
    
    contingency = pd.crosstab(df["Protocol"], df["Category"])
    scipy_chi2, scipy_p, scipy_dof, _ = stats.chi2_contingency(contingency)
    
    assert np.isclose(res["chi2_stat"], scipy_chi2)
    assert np.isclose(res["p_value"], scipy_p)
    assert res["df"] == scipy_dof
    assert res["cramers_v"] > 0.5


def test_one_way_anova():
    df = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "val": list(np.random.normal(10, 2, 30)) + list(np.random.normal(20, 2, 30)) + list(np.random.normal(30, 2, 30))
    })
    res = one_way_anova(df, "group", "val")
    
    groups = [df[df["group"] == g]["val"] for g in ["A", "B", "C"]]
    scipy_f, scipy_p = stats.f_oneway(*groups)
    
    assert np.isclose(res["f_stat"], scipy_f)
    assert np.isclose(res["p_val"], scipy_p)
    assert res["eta_squared"] > 0.5


def test_linear_regression_two_lines():
    np.random.seed(42)
    x = np.random.normal(10, 2, 100)
    y = 2.5 * x + 5.0 + np.random.normal(0, 1, 100)
    
    res = linear_regression_two_lines(x, y)
    scipy_reg = stats.linregress(x, y)
    
    assert np.isclose(res["slope_yx"], scipy_reg.slope)
    assert np.isclose(res["intercept_yx"], scipy_reg.intercept)
    assert np.isclose(res["r"], scipy_reg.rvalue)
    # Verify the fundamental identity: b_yx * b_xy == r^2
    assert np.isclose(res["b_product"], res["r_squared"])


def test_predict_with_intervals():
    np.random.seed(42)
    x = np.random.normal(10, 2, 100)
    y = 2.0 * x + 3.0 + np.random.normal(0, 0.5, 100)
    
    res = predict_with_intervals(x_new=10.0, x_data=x, y_data=y)
    
    assert res["ci_lower"] <= res["y_hat"] <= res["ci_upper"]
    assert res["pi_lower"] <= res["y_hat"] <= res["pi_upper"]
    # Prediction interval must always be wider than Confidence interval
    assert res["pi_margin"] > res["ci_margin"]


def test_spearman_rank_worked():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([12.0, 24.0, 33.0, 45.0, 52.0])
    
    res = spearman_rank_worked(x, y)
    assert np.isclose(res["rho_manual"], 1.0)
    assert np.isclose(res["rho_scipy"], 1.0)


def test_classification_error_analysis():
    classes = ["Benign", "DDoS", "Spoofing"]
    y_true = np.array(["Benign"] * 50 + ["DDoS"] * 40 + ["Spoofing"] * 10)
    y_pred = np.array(["Benign"] * 48 + ["DDoS"] * 2 + ["DDoS"] * 38 + ["Benign"] * 2 + ["Spoofing"] * 8 + ["Benign"] * 2)
    
    res = classification_error_analysis(y_true, y_pred, classes)
    
    assert res["confusion_matrix"].shape == (3, 3)
    assert len(res["per_class_df"]) == 3
    spoof_row = res["per_class_df"][res["per_class_df"]["Class"] == "Spoofing"].iloc[0]
    assert spoof_row["Support"] == 10
    assert spoof_row["Recall"] == 0.80
