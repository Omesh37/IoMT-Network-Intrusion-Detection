# Securing Medical IoT: A Statistical and Machine Learning Approach to Network Intrusion Detection

ML-based intrusion detection and interactive statistical learning testbed for medical IoT (IoMT) network traffic, using the **CICIoMT2024** dataset (Canadian Institute for Cybersecurity, UNB). Classifies network flows as Benign or one of 5 attack categories (DDoS, DoS, MQTT, Recon, Spoofing), served via a FastAPI REST API and an interactive, multi-tab Streamlit dashboard designed for university **Probability and Statistics** coursework.

---

## 🎯 Problem Statement
Internet of Medical Things (IoMT) devices (e.g., patient monitors, infusion pumps, diagnostic telemetry sensors) are increasingly targeted by cyber attacks. This project investigates and detects 18 real attack variants across 5 major categories deployed against a realistic clinical IoT network testbed, evaluating both statistical signatures and supervised machine learning performance.

---

## 🛠️ Data Engineering
- **Source**: CICIoMT2024 `WiFi_and_MQTT/csv` split — 51 raw capture files (~1.9 GB), one capture file per attack-scenario.
- **Label Reconstruction**: Reconstructed canonical labels directly from file metadata (raw packet records lack a label column).
- **Consolidation**: Collapsed chunked packet captures (e.g., `UDP1` through `UDP8`) into unified scenario classes.
- **Hierarchical Taxonomy**: Mapped 19 granular capture labels to 5 standardized attack categories + Benign using the paper's taxonomy.
- **Format**: Preprocessed into high-performance Parquet format with 45 flow telemetry features.

---

## 📈 Exploratory Data Analysis (EDA)

### Class Imbalance
The network traffic exhibits natural real-world class imbalance. Volumetric attacks (such as DDoS and DoS) significantly outnumber rare evasion attacks like Spoofing:
- Attack flows outnumber Benign flows by approximately 7:1 in total traffic volume.
- DDoS alone accounts for nearly 5× the traffic of Spoofing, MQTT, or Reconnaissance.

<p align="center">
  <img width="48%" alt="Class Balance Distribution" src="https://github.com/user-attachments/assets/04b22d35-dac4-438c-8a9d-8c8ff129941d" />
  <img width="48%" alt="Attack Category Breakdown" src="https://github.com/user-attachments/assets/27eeb62a-4884-4b4e-9fba-b53cbe19af5b" />
</p>

### Distribution Overlap (Benign vs. Spoofing)
Spoofing attacks exhibit Duration, Rate, and Inter-Arrival Time (IAT) distributions that closely overlap normal Benign medical communications, making statistical discrimination challenging:

<p align="center">
  <img width="70%" alt="Feature Overlap Analysis" src="https://github.com/user-attachments/assets/383e9b51-a9b5-466b-9fce-e82541e1748d" />
</p>

### Multicollinearity in Network Protocols
Certain protocol header fields co-occur deterministically in Ethernet packet encapsulation:
- **`ARP` & `IPv`**: $r = 1.000$
- **`LLC` & `ARP`**: $r = 1.000$
- **`IPv` & `LLC`**: $r = 1.000$

<p align="center">
  <img width="75%" alt="Feature Correlation Heatmap" src="https://github.com/user-attachments/assets/a29dc114-ccc5-495a-a3e3-3a316cf0212c" />
</p>

### Traffic Distribution & Binary Balance
<p align="center">
  <img width="60%" alt="Traffic Distribution" src="https://github.com/user-attachments/assets/f6e2b628-8522-4b55-a9c4-98e9311c2dfb" />
</p>
<p align="center">
  <img width="70%" alt="Binary vs Multi-class Balance" src="https://github.com/user-attachments/assets/ad114b29-edd2-45f2-84f6-31bce88696dd" />
</p>

---

## 🤖 Modeling & Evaluation
A staged machine learning approach was implemented using Random Forest ensembles:
- **Binary Model (Benign vs. Attack)**:
  - Attack Recall: **1.00**
  - Benign Recall: **0.95**
- **5-Class Model (DDoS, DoS, MQTT, Recon, Spoofing)**:
  - Near-perfect classification on volumetric flood attacks (DDoS/DoS/MQTT: $\approx 1.00$ F1).
  - Challenging detection on stealth attacks (Spoofing: **0.74 Precision, 0.82 Recall**).
  - Evaluated using both **Macro-Average F1 (0.95)** and **Weighted-Average F1 (0.99)** to ensure the minority Spoofing class is rigorously assessed rather than masked by high-frequency DoS floods.
  - Top informative features: `Max`, `Magnitue`, `Variance`, `rst_count`.

---

## 📊 Interactive Statistics Dashboard

A dedicated, multi-tab Streamlit dashboard designed for a university **Probability and Statistics** course. Every statistical concept is dynamically computed from live telemetry flows, accompanied by plain-English *"What this means for IoMT security"* interpretations.

### 🖼️ Dashboard Architecture
```
+----------------------------------------------------------------------------------------------------+
| 🛡️ Securing Medical IoT: A Statistical and Machine Learning Approach                              |
| [🏠 Project Overview] [📊 Tab 1: Descriptive] [🎲 Tab 2: Probability] [📈 Tab 3: Normal/CLT]       |
| [🔬 Tab 4: Inference/ANOVA] [🔗 Tab 5: Correlation/Regression] [🤖 Tab 6: Model & Errors]          |
+----------------------------------------------------------------------------------------------------+
| Global Sidebar Controls: Stratified sample slider (100k flows), Class filter, Seed, log1p toggle  |
| Visualizations: Interactive Plotly distributions, Q-Q plots, CLT simulations, ANOVA & OLS Fit      |
+----------------------------------------------------------------------------------------------------+
```

### 📋 Course Syllabus Coverage Matrix

| Course Unit | Syllabus Topic | Dashboard Module | Features & Interactive Controls |
|---|---|---|---|
| **Unit 1** | Types of Data (Qualitative vs Quantitative, Scales) | `Tab 1: Descriptive Stats` | Classification table of all 45 flow features; nominal protocol types and Bernoulli flags. |
| **Unit 1** | Frequency Distributions (Count, Relative %, Cumulative %) | `Tab 1: Descriptive Stats` | Dynamic frequency table and interactive Plotly bar chart. |
| **Unit 1** | Histograms, Box Plots, & Dispersion | `Tab 1: Descriptive Stats` | Bin slider (10-120), per-class box plots, mean, median, mode, variance, and standard deviation. |
| **Unit 1** | Relative Dispersion (CV) & Shape (Skewness, Kurtosis, IQR) | `Tab 1: Descriptive Stats` | Automated ranking of highest CV feature; Tukey 1.5×IQR outlier detection fences. |
| **Unit 2** | Probability Axioms, Complement, and Addition Rule | `Tab 2: Probability & Distributions` | $P(A)$, $P(A') = 1 - P(A)$, and $P(A \cup B)$ with interactive event selectors and Venn visualizer. |
| **Unit 2** | Conditional Probability & Statistical Independence | `Tab 2: Probability & Distributions` | 2×2 Contingency table, $P(C \mid F)$, $P(F \cap C)$ vs $P(F)P(C)$ independence test. |
| **Unit 2** | Bayes' Theorem & Clinical Base-Rate Fallacy | `Tab 2: Probability & Distributions` | Interactive prior slider, sensitivity, specificity, posterior curve, and medical false-alarm explainer. |
| **Unit 2** | Discrete Probability Distributions | `Tab 2: Probability & Distributions` | Bernoulli flag $p$, Binomial($n, p$) PMF/CDF, Poisson($\lambda$) MLE on packet arrivals. |
| **Unit 2** | Continuous Probability Distributions | `Tab 2: Probability & Distributions` | Exponential MLE ($\hat{\lambda} = 1/\bar{x}$) on IAT with KS goodness-of-fit; Continuous Uniform $U(a, b)$. |
| **Unit 3** | Normal Distribution Fitting & Normality Tests | `Tab 3: Normal & Sampling` | Normal fit ($\mu, \sigma$), Q-Q plots, Shapiro-Wilk and D'Agostino-Pearson tests. |
| **Unit 3** | Standard Normal ($Z$-Scores) & Anomaly Detection | `Tab 3: Normal & Sampling` | Interactive $Z$-score calculator; $|Z| > \text{threshold}$ anomaly filter with attack composition audit. |
| **Unit 3** | Normal Approximation to Discrete Distributions | `Tab 3: Normal & Sampling` | Normal approx to Binomial and Poisson with continuity correction ($\pm 0.5$) and error metric. |
| **Unit 3** | Central Limit Theorem (CLT) Simulation | `Tab 3: Normal & Sampling` | Sliders for $n$ and $M$; theoretical $SE = \sigma/\sqrt{n}$ vs empirical SD of means with normal curve overlay. |
| **Unit 3** | Hypothesis Testing Rejection Regions | `Tab 3: Normal & Sampling` | Visualized rejection regions (two-tailed, right, left) on the standard Gaussian curve with $\alpha$ slider. |
| **Unit 4** | Point Estimation & Bessel's Correction ($n-1$) | `Tab 4: Inference & ANOVA` | Monte Carlo simulation demonstrating sample variance bias ($n$) vs unbias ($n-1$). |
| **Unit 4** | Hypothesis Testing ($Z$-Test, Welch's $t$-Test) | `Tab 4: Inference & ANOVA` | One-sample and two-sample tests, Welch-Satterthwaite $df$, and Cohen's $d$ effect sizes. |
| **Unit 4** | Type I ($\alpha$) and Type II ($\beta$) Error Trade-off | `Tab 4: Inference & ANOVA` | Statistical power visualizer; mapping Type I to false alarms and Type II to missed intrusions. |
| **Unit 4** | Chi-Square Goodness-of-Fit & Independence | `Tab 4: Inference & ANOVA` | Uniform GOF test; Protocol vs Category contingency table with Cramér's $V$ and Pearson residuals. |
| **Unit 4** | One-Way ANOVA with $F$-Test & Tukey HSD | `Tab 4: Inference & ANOVA` | Complete ANOVA table (SSB, SSW, MS, $F$, $p$), Levene's variance test, Kruskal-Wallis, Tukey HSD. |
| **Unit 5** | Simple Linear Regression & Correlation | `Tab 5: Correlation & Regression` | OLS regression line ($\hat{y} = b_0 + b_1 x$), Pearson $r$, $R^2$, and single clean scatter plot with trendline. |
| **Unit 6** | Trained Random Forest 5-Class Evaluation | `Tab 6: Model & Error Analysis` | Confusion matrix heatmap, precision, recall, F1, and Gini feature importances. |
| **Unit 6** | Statistical Error Rates (Type I FPR & Type II FNR) | `Tab 6: Model & Error Analysis` | Per-class false positive and false negative rates with 95% Wilson Score Confidence Intervals. |
| **Unit 6** | Interactive Live Flow Predictor & JSON Classifier | `Tab 6: Model & Error Analysis` | Interactive feature slider / raw JSON classifier (folds `demo.py`), optional link to FastAPI `/predict`. |

---

## 🚀 Quickstart & Installation

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/Omesh37/IoMT-Network-Intrusion-Detection.git
cd IoMT-Network-Intrusion-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit Statistics Dashboard
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Running with Docker

#### 1. Streamlit Statistics Dashboard
```bash
docker build -f Dockerfile.dashboard -t iomt-dashboard .
docker run -p 8501:8501 iomt-dashboard
```
Access at `http://localhost:8501`.

#### 2. FastAPI Model Serving API
```bash
docker build -t iomt-api .
docker run -p 8000:8000 iomt-api
```
Access interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

---

## 🧪 Unit Testing
All underlying statistical formulas are verified against standard **SciPy** and **Statsmodels** implementations:
```bash
python -m pytest tests/
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
├── main.py                        # FastAPI /predict serving endpoint
├── demo.py                        # Minimal JSON predictor
├── rf_5class_full.joblib          # Trained Random Forest 5-class model
├── requirements.txt               # Project dependencies
├── Dockerfile                     # API containerization
├── Dockerfile.dashboard           # Dashboard containerization
└── README.md                      # Comprehensive project documentation
```

---

## 🔬 Scientific Dataset Citation
```bibtex
@article{dadkhah2024ciciomt2024,
  title={CICIoMT2024: A Dataset for Internet of Medical Things Cybersecurity Analysis},
  author={Dadkhah, Sajjad and others},
  journal={Canadian Institute for Cybersecurity, University of New Brunswick},
  year={2024}
}
```
