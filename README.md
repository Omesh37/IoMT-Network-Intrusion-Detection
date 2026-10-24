# IoMT Network Intrusion Detection & Statistics Testbed

ML-based intrusion detection and interactive statistical learning testbed for medical IoT (IoMT) network traffic, using the CICIoMT2024 dataset (Canadian Institute for Cybersecurity, UNB). Classifies network flows as Benign or one of 5 attack categories (DDoS, DoS, MQTT, Recon, Spoofing), served via a FastAPI endpoint and an interactive, multi-tab Streamlit statistics dashboard for university **Probability and Statistics** coursework.

---

## 📊 Statistics Dashboard

An interactive university-grade probability and statistics dashboard built on real medical IoT telemetry. Every statistical concept is calculated dynamically from the data with plain-English *"What this means for IoMT security"* interpretations.

### 🖼️ Dashboard Preview
```
+----------------------------------------------------------------------------------------------------+
| 🛡️ Securing Medical IoT: A Statistical and Machine Learning Approach                              |
| [🏠 Project Overview] [📊 Tab 1: Descriptive] [🎲 Tab 2: Probability] [📈 Tab 3: Normal/CLT]       |
| [🔬 Tab 4: Inference/ANOVA] [🔗 Tab 5: Correlation/Regression] [🤖 Tab 6: Model & Errors]          |
+----------------------------------------------------------------------------------------------------+
| Global Sidebar Controls: Data source, Stratified sample slider (100k rows), Class filters, log1p  |
| Visualizations: Interactive Plotly distributions, Q-Q plots, CLT simulations, ANOVA & OLS Fit      |
+----------------------------------------------------------------------------------------------------+
```
*(Screenshot placeholder: `assets/dashboard_preview.png`)*

### Quickstart & Installation
```bash
# 1. Clone repository and install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

### Running with Docker
- **Dashboard container**:
  ```bash
  docker build -f Dockerfile.dashboard -t iomt-dashboard .
  docker run -p 8501:8501 iomt-dashboard
  ```
- **API container** (FastAPI `/predict` endpoint):
  ```bash
  docker build -t iomt-api .
  docker run -p 8000:8000 iomt-api
  ```

---

## 📋 Course Syllabus Coverage Matrix

| Course Unit | Syllabus Topic | Dashboard Module | Features & Interactive Controls |
|---|---|---|---|
| **Unit 1** | Types of Data (Qualitative vs Quantitative, Scales) | `Tab 1: Descriptive Stats` | Classification table of all 45 flow features; notes on Protocol Type nominal encoding and Bernoulli flags. |
| **Unit 1** | Frequency Distribution (Count, Relative %, Cumulative %) | `Tab 1: Descriptive Stats` | Dynamic frequency table and interactive Plotly bar chart. |
| **Unit 1** | Histograms, Box Plots, Violin Plots | `Tab 1: Descriptive Stats` | Bin-count slider (10-120), per-class box plots, optional violin plot toggle. |
| **Unit 1** | Central Tendency & Dispersion (Mean, Median, Mode, Var, SD) | `Tab 1: Descriptive Stats` | Overall and per-class summary statistics with Bessel's correction ($s^2$). |
| **Unit 1** | Relative Dispersion (CV) & Shape (Skewness, Kurtosis, IQR) | `Tab 1: Descriptive Stats` | Automated ranking of highest CV feature; Tukey 1.5×IQR outlier fences. |
| **Unit 2** | Probability Axioms, Complement, and Addition Rule | `Tab 2: Probability & Distributions` | $P(A)$, $P(A') = 1 - P(A)$, $P(A \cup B)$ with interactive event selectors and Venn-style area visualizer. |
| **Unit 2** | Conditional Probability & Statistical Independence | `Tab 2: Probability & Distributions` | 2×2 Contingency table, $P(C \mid F)$, $P(F \cap C)$ vs $P(F)P(C)$ independence test. |
| **Unit 2** | Bayes' Theorem & Base-Rate Effect | `Tab 2: Probability & Distributions` | Prior slider, sensitivity, specificity, posterior curve, and medical base-rate fallacy explainer. |
| **Unit 2** | Discrete Distributions (Bernoulli, Binomial, Poisson) | `Tab 2: Probability & Distributions` | Bernoulli flag $p$, Binomial($n, p$) PMF/CDF sliders, Poisson($\lambda$) MLE on arrival counts. |
| **Unit 2** | Continuous Distributions (Exponential MLE & Uniform) | `Tab 2: Probability & Distributions` | MLE $\hat{\lambda} = 1/\bar{x}$ on IAT, KS goodness-of-fit test; interactive Uniform $U(a, b)$. |
| **Unit 3** | Normal Distribution Fitting & Diagnostic Tests | `Tab 3: Normal & Sampling` | Normal fit ($\mu, \sigma$), Q-Q plot, Shapiro-Wilk and D'Agostino normality tests with verbal verdicts. |
| **Unit 3** | Standard Normal ($Z$-Scores) & Anomaly Detection | `Tab 3: Normal & Sampling` | Tail area calculator; outlier filtering ($|Z| > \text{threshold}$) and attack composition audit. |
| **Unit 3** | Normal Approximation to Discrete Distributions | `Tab 3: Normal & Sampling` | Normal approx to Binomial and Poisson with continuity correction ($\pm 0.5$) and error metric. |
| **Unit 3** | Central Limit Theorem (CLT) Simulation | `Tab 3: Normal & Sampling` | Sliders for $n$ and $M$; theoretical $SE = \sigma/\sqrt{n}$ vs empirical SD of means; multi-sample size progression. |
| **Unit 3** | Hypothesis Testing Rejection Regions | `Tab 3: Normal & Sampling` | Visualized rejection regions (two-tailed, right, left) on standard normal curve with $\alpha$ slider. |
| **Unit 4** | Point Estimation & Proof of Bessel's Correction ($n-1$) | `Tab 4: Inference & ANOVA` | Monte Carlo simulation proving $S_n^2$ is downward-biased while $s^2$ ($n-1$) is unbiased. |
| **Unit 4** | Confidence Intervals for Mean ($Z$ vs $t$) & Repeated Sampling | `Tab 4: Inference & ANOVA` | Confidence slider (80%-99%), $\sigma$ known/unknown toggle; 100-sample coverage simulation. |
| **Unit 4** | Confidence Interval for a Proportion | `Tab 4: Inference & ANOVA` | 95% Wilson Score Interval for rare categories (e.g. Spoofing proportion and precision). |
| **Unit 4** | One-Sample Hypothesis Testing ($Z$-Test & $t$-Test) | `Tab 4: Inference & ANOVA` | Automated null/alternative generation, critical values, $p$-value, rejection-region plot. |
| **Unit 4** | Two-Sample Welch's $t$-Test & Effect Sizes | `Tab 4: Inference & ANOVA` | Benign vs Spoofing comparison, Welch-Satterthwaite $df$, Cohen's $d$ effect size interpretation. |
| **Unit 4** | Type I ($\alpha$) and Type II ($\beta$) Errors & Statistical Power | `Tab 4: Inference & ANOVA` | Interactive trade-off plot; explicit mapping of Type I to false alarms and Type II to missed attacks. |
| **Unit 4** | Chi-Square Goodness-of-Fit & Independence Tests | `Tab 4: Inference & ANOVA` | Uniform GOF test; Protocol vs Category independence test with Cramér's $V$ and Pearson residuals heatmap. |
| **Unit 4** | One-Way ANOVA with $F$-Test & Tukey HSD | `Tab 4: Inference & ANOVA` | Complete ANOVA table (SSB, SSW, MS, $F$, $p$), Levene's variance test, Kruskal-Wallis, Tukey HSD. |
| **Unit 4** | Big Data P-Value Saturation Slider | `Tab 4: Inference & ANOVA` | Subsampling slider to contrast tiny $p$-values with practical effect sizes on large network samples. |
| **Unit 5** | Simple Linear Regression & Correlation | `Tab 5: Correlation & Regression` | OLS fit, Pearson correlation $r$, $R^2$, slope, intercept, and a single clean scatter plot with regression line. |
| **Unit 6** | Trained Random Forest 5-Class Evaluation | `Tab 6: Model & Error Analysis` | Confusion matrix heatmap, precision, recall, F1, and Gini feature importances. |
| **Unit 6** | Macro-Average vs Weighted-Average Analysis | `Tab 6: Model & Error Analysis` | Demonstrates how weighted F1 (~0.99) conceals Spoofing weakness revealed by macro F1 (~0.95). |
| **Unit 6** | Statistical Error Rates (Type I FPR & Type II FNR) | `Tab 6: Model & Error Analysis` | Per-class FPR and FNR with 95% Wilson Score Confidence Intervals. |
| **Unit 6** | Interactive Live Flow Predictor & JSON Classifier | `Tab 6: Model & Error Analysis` | Slider and JSON input classifier (folds `demo.py`), optional link to FastAPI `/predict`. |

---

## 🧪 Unit Testing
The dashboard's statistical functions are thoroughly tested against **SciPy** and **Statsmodels** standard outputs:
```bash
pytest tests/test_stats_utils.py
```

---

## 📁 Repository Structure
```
├── app.py                         # Master Streamlit dashboard entry point with sidebar controls
├── home.py                        # Project overview and dataset summary
├── tabs/
│   ├── tab1_descriptive.py        # Unit 1: Descriptive Statistics & Measurement Scales
│   ├── tab2_probability.py        # Unit 2: Probability, Bayes' Theorem, & Distributions
│   ├── tab3_normal_sampling.py    # Unit 3: Normal Fit, Z-Scores, & CLT
│   ├── tab4_inference_tests.py    # Unit 4: Hypothesis Tests, Welch t-Test, Chi-Square, & ANOVA
│   ├── tab5_correlation_regression.py # Unit 5: Correlation & Simple Linear Regression
│   └── tab6_model_errors.py       # Unit 6: Model Evaluation, Error Rates (FPR/FNR), & Live Inference
├── utils/
│   ├── data_loader.py             # Caching, stratified sampling, and synthetic data fallback
│   ├── stats_utils.py             # Pure, type-hinted, tested statistical functions
│   └── plot_utils.py              # Consistent Plotly dark/light styling and color palettes
├── tests/
│   └── test_stats_utils.py        # Pytest test suite benchmarking against SciPy
├── main.py                        # FastAPI /predict serving endpoint (preserved)
├── demo.py                        # Minimal JSON predictor (preserved / folded into Tab 6)
├── rf_5class_full.joblib          # Trained Random Forest 5-class model
├── requirements.txt               # Updated dependencies
├── Dockerfile                     # Preserved API container
└── Dockerfile.dashboard           # Dedicated Streamlit dashboard container
```

---

## 🔬 Dataset & Scientific Citation
- **Source**: Dadkhah et al., *CICIoMT2024: A Dataset for Internet of Medical Things Cybersecurity Analysis*, Canadian Institute for Cybersecurity, University of New Brunswick.
- **Features**: 45 numeric flow features covering packet lengths, protocol indicators, flag counts, and time-window statistics.
- **Classes**: Benign plus 5 attack families: DDoS, DoS, MQTT, Recon, and Spoofing.
- **Note on Class Ratios**: Class proportions are dynamically computed directly from loaded data in memory, avoiding hardcoded or contradictory historical values.