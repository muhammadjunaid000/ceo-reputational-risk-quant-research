# CEO Reputational Risk, Corporate Events & Quantitative Portfolio Risk Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-11%20passed-brightgreen.svg)](tests/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An institutional-grade quantitative finance research repository investigating executive reputational exposure, corporate event econometrics, GARCH(1,1) volatility dynamics, Euler portfolio risk attribution, drawdown cycle decomposition, and downside stress testing.

---

## Table of Contents
1. [Research Overview & Problem Statement](#1-research-overview--problem-statement)
2. [Research Questions & Hypotheses](#2-research-questions--hypotheses)
3. [Analytical & Mathematical Framework](#3-analytical--mathematical-framework)
4. [Dataset & Data Provenance](#4-dataset--data-provenance)
5. [Key Empirical Findings & Audit](#5-key-empirical-findings--audit)
6. [Interactive Visualizations](#6-interactive-visualizations)
7. [Repository Architecture](#7-repository-architecture)
8. [Installation & Execution Guide](#8-installation--execution-guide)
9. [Unit Testing & Reproducibility](#9-unit-testing--reproducibility)
10. [Methodological Limitations](#10-methodological-limitations)
11. [Research Extensions](#11-research-extensions)
12. [Disclaimer](#12-disclaimer)

---

## 1. Research Overview & Problem Statement

In modern technology enterprises, executive leadership and founder reputations are intimately bound to enterprise valuation, capital cost, and brand equity. However, quantitative portfolio management often treats firm risk as homogeneous asset volatility, ignoring how key-person exposure and headline news transmit non-linear downside risk into multi-asset portfolios.

This research framework establishes a rigorous, reproducible econometrics and portfolio engineering methodology to analyze:
- **Event-Driven Abnormal Returns:** Modeling asset return deviations against market and peer-control benchmarks.
- **Volatility Regimes:** Quantifying time-varying conditional variance, shock persistence, and downside semivariance.
- **Euler Risk Decomposition:** Disentangling nominal portfolio capital weights from actual percentage risk contributions.
- **Drawdown Cycle Dynamics:** Isolating true economic peak-to-recovery drawdown episodes from superficial threshold crossings.
- **Downside Stress Testing:** Evaluating portfolio resilience under deterministic shocks and correlated Monte Carlo simulations.

---

## 2. Research Questions & Hypotheses

1. **Risk Concentration vs. Capital Allocation:** Does an equity holding's component risk contribution deviate significantly from its nominal capital weight due to cross-asset covariance?
   - *Finding:* In our target portfolio, **NVDA** (30% weight) contributes **33.74%** of total volatility, while **TSLA** (15% weight) contributes **18.23%**, demonstrating that volatile and highly correlated tech names act as leveraged risk drivers.
2. **Volatility Persistence:** How persistent are idiosyncratic shocks in high-beta semiconductor equities?
   - *Finding:* GARCH(1,1) modeling reveals a volatility persistence ($\alpha + \beta$) of **0.736**, with a shock half-life of **2.3 trading sessions**.
3. **Identification & Causal Attribution:** Can a single-firm price shock around an executive announcement be interpreted as causal reputational damage?
   - *Methodological Rule:* No. Observed market value changes combine macro factors, sector trends, and company news. Without verified announcement timestamps and robust control series, causal inference is unidentifiable.

---

## 3. Analytical & Mathematical Framework

### 3.1 Portfolio Risk Decomposition (Euler's Theorem)
For a portfolio with weight vector $w \in \mathbb{R}^N$ and covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$, portfolio volatility is linearly homogeneous:
$$\sigma_p(w) = \sqrt{w^T \Sigma w}$$

By Euler's theorem:
$$\sigma_p = \sum_{i=1}^N w_i \frac{\partial \sigma_p}{\partial w_i}$$

Where:
- **Marginal Contribution to Risk (MCR):**
  $$MCR_i = \frac{(\Sigma w)_i}{\sigma_p}$$
- **Absolute Component Contribution to Risk ($CR_i$):**
  $$CR_i = w_i \cdot MCR_i$$
- **Percentage Contribution to Risk ($PCR_i$):**
  $$PCR_i = \frac{w_i (\Sigma w)_i}{\sigma_p^2}, \quad \sum_{i=1}^N PCR_i = 100.0\%$$

### 3.2 GARCH(1,1) Conditional Heteroskedasticity
Time-varying volatility is modeled via:
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t_\nu(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### 3.3 Downside Tail Risk Metrics
- **Historical Value-at-Risk (VaR):** $VaR_\alpha = -Q_\alpha(R)$
- **Historical Expected Shortfall (CVaR):** $ES_\alpha = -\mathbb{E}[R \mid R \le -VaR_\alpha]$
- **Cornish-Fisher Expansion (accounting for skewness $S$ and excess kurtosis $K$):**
  $$z_{CF} = z_\alpha + \frac{z_\alpha^2 - 1}{6}S + \frac{z_\alpha^3 - 3z_\alpha}{24}K - \frac{2z_\alpha^3 - 5z_\alpha}{36}S^2$$
  $$VaR_{CF} = -(\mu - z_{CF} \sigma)$$

### 3.4 Drawdown Cycle Segmentation
Drawdown is measured relative to running peak wealth:
$$W_t = \prod_{s=1}^t (1 + R_s), \quad M_t = \max_{s \le t} W_s, \quad DD_t = \frac{W_t - M_t}{M_t}$$
An episode is strictly defined from initial peak drop ($W_t < M_t$) to full recovery ($W_t \ge M_t$). Fluctuations around interim loss levels are part of the same drawdown episode, not independent events.

---

## 4. Dataset & Data Provenance

- **Universe of Assets:**
  - **Portfolio Holdings:** NVIDIA (`NVDA`), Meta Platforms (`META`), Amazon (`AMZN`), Tesla (`TSLA`), Microsoft (`MSFT`).
  - **Expanded Research Universe:** Alphabet (`GOOGL`), Advanced Micro Devices (`AMD`), Broadcom (`AVGO`).
  - **Market & Sector Benchmarks:** S&P 500 ETF (`SPY`), Invesco QQQ (`QQQ`), iShares Semiconductor ETF (`SOXX`).
- **Research Sample Period:** November 3, 2025 to November 13, 2026 (260 trading sessions).
- **Source:** Yahoo Finance (adjusted for splits and dividends) with automated local CSV caching in `data/raw/`.
- **Quality Assurance:** 0 missing values, 0 nonpositive prices, verified timezone alignment.

---

## 5. Key Empirical Findings & Audit

### Portfolio Performance & Risk Attribution Audit
The target portfolio allocation was audited against the 260-session empirical sample:

| Holding | Nominal Weight | Marginal Risk (MCR) | Component Risk Contribution | Risk-to-Weight Ratio |
|---|---|---|---|---|
| **NVDA** | **30.00%** | **0.2955** | **33.74%** | **1.12** |
| **META** | **20.00%** | **0.2758** | **21.00%** | **1.05** |
| **AMZN** | **20.00%** | **0.2222** | **16.92%** | **0.85** |
| **TSLA** | **15.00%** | **0.3192** | **18.23%** | **1.22** |
| **MSFT** | **15.00%** | **0.1773** | **10.12%** | **0.67** |
| **Total** | **100.00%** | — | **100.00%** | — |

- **Sample Total Return:** **-4.07%**
- **Annualized Realized Volatility:** **26.27%**
- **Maximum Drawdown:** **-18.72%** (117-session duration, fully recovered on 2026-04-27).
- **1-Day 95% Historical VaR:** **2.62%** | **95% Expected Shortfall:** **3.70%**
- **1-Day 99% Historical VaR:** **4.48%** | **99% Expected Shortfall:** **5.23%**

---

## 6. Interactive Visualizations

All Plotly charts are fully interactive and exported as standalone HTML documents:

### Core Visualizations:
- [Portfolio Risk-Reward Profile](visualizations/portfolio_risk_reward.html): Cross-asset annualized volatility vs return bubble map.
- [Portfolio Drawdown & Underwater Analysis](visualizations/portfolio_drawdown.html): Underwater equity curves highlighting peak-to-trough episodes.
- [Component Risk Contribution](visualizations/risk_contribution.html): Side-by-side comparison of capital weights vs. percentage risk contributions.
- [Correlation Matrix Heatmap](visualizations/correlation_heatmap.html): Cross-asset daily return correlation matrix.

### Econometric & Analytical Visualizations:
- [GARCH(1,1) Volatility Dynamics](visualizations/other_charts/garch_volatility.html): Conditional volatility series and persistence decay.
- [Markowitz Efficient Frontier](visualizations/other_charts/markowitz_efficient_frontier.html): Monte Carlo portfolios, Min Vol, Max Sharpe (Tangency), and portfolio positioning.
- [Rolling VaR & Expected Shortfall](visualizations/other_charts/rolling_var_es.html): 60-day rolling downside risk metrics.
- [Monte Carlo Loss Distribution](visualizations/other_charts/monte_carlo_loss_distribution.html): 30,000 simulated reputational shock paths.
- [Normalized Asset Wealth](visualizations/other_charts/normalized_performance.html): Comparative base-100 equity curves.
- [Asset Return Distributions](visualizations/other_charts/returns_distribution.html): Dual-panel histograms and 5th-percentile downside tails.
- [Scenario Stress Testing](visualizations/other_charts/scenario_stress_testing.html): Deterministic parallel vs idiosyncratic downside shocks.
- [Monthly Returns Matrix](visualizations/other_charts/monthly_performance_heatmap.html): Year $\times$ Month portfolio return heatmap.
- [Event Study AR/CAR Panel Template](visualizations/other_charts/event_study_ar_car_template.html): Event-time abnormal return framework.

---

## 7. Repository Architecture

```text
ceo-reputational-risk-quant-research/
│
├── README.md                                 # Institutional research overview & documentation
├── LICENSE                                   # MIT open-source license
├── .gitignore                                # Version control exclusions
├── requirements.txt                          # Pip dependency specifications
├── environment.yml                           # Conda environment definition
│
├── notebooks/
│   └── CEO_Reputational_Risk_Quant_Research.ipynb # Modular, executable research notebook
│
├── src/                                      # Production quantitative library
│   ├── __init__.py
│   ├── data_collection.py                    # Yahoo Finance acquisition, caching & validation
│   ├── data_processing.py                    # Returns, rolling beta, volume surprise & summary stats
│   ├── event_study.py                        # AR/CAR estimation, bootstrap & placebo diagnostics
│   ├── volatility_models.py                  # Realized volatility, semivariance & GARCH(1,1)
│   ├── portfolio_optimization.py             # Euler decomposition, weights & Markowitz frontier
│   ├── risk_metrics.py                       # VaR, Expected Shortfall & drawdown cycle segmentation
│   ├── stress_testing.py                     # Parallel, idiosyncratic & correlated Monte Carlo stress
│   └── visualization.py                      # Interactive Plotly publication theme & chart generators
│
├── data/
│   ├── raw/                                  # Cached OHLCV market files (close, volume, open, high, low)
│   └── processed/                            # Cleaned daily returns, risk tables & drawdown records
│
├── visualizations/                           # Interactive standalone Plotly HTML charts
│   ├── portfolio_risk_reward.html
│   ├── portfolio_drawdown.html
│   ├── risk_contribution.html
│   ├── correlation_heatmap.html
│   └── other_charts/                         # 13 supplementary interactive visualizations
│
├── reports/
│   └── research_summary.md                   # Comprehensive executive research summary
│
└── tests/                                    # Automated unit test suite
    ├── __init__.py
    ├── test_risk_metrics.py                  # Tests for VaR, ES, and Cornish-Fisher monotonicity
    ├── test_portfolio.py                     # Tests for Euler theorem (sum=100%), weights & metrics
    └── test_drawdown.py                      # Tests for cycle segmentation vs threshold crossings
```

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Conda or virtualenv

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/MuhammadJunaid/ceo-reputational-risk-quant-research.git
cd ceo-reputational-risk-quant-research

# Option A: Conda
conda env create -f environment.yml
conda activate ceo-quant-risk

# Option B: Pip Virtualenv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Test Suite
```bash
pytest tests/ -v
```

### 3. Launch Research Notebook
```bash
jupyter lab notebooks/CEO_Reputational_Risk_Quant_Research.ipynb
```

---

## 9. Unit Testing & Reproducibility

The repository includes 11 automated unit tests across 3 test modules verifying numerical accuracy:
- **`test_portfolio.py`:** Validates that Euler component risk contributions sum strictly to 100.0%, verifies that capital weights do not equal risk contributions, and validates weight normalization.
- **`test_risk_metrics.py`:** Validates historical VaR monotonicity ($VaR_{99} \ge VaR_{95}$), verifies expected shortfall inequality ($ES \ge VaR$), tests analytical Gaussian VaR formulas, and verifies Cornish-Fisher tail corrections.
- **`test_drawdown.py`:** Methodological test verifying that multi-oscillation price declines are correctly identified as **one** single economic drawdown episode rather than multiple threshold crossings.

---

## 10. Methodological Limitations

1. **Event Study Verification Gate:** Provisional event dates without primary source documentation and timestamp verification remain disabled to preserve econometric integrity.
2. **Causal Confounding:** Single-company price movements around CEO announcements cannot be causally attributed to reputational damage without controlling for concurrent earnings surprises, semiconductor supply chain events, and interest rate announcements.
3. **Distributional Assumptions:** Correlated Monte Carlo parameters ($p=2\%$ annual event rate, $\mu=-6\%$ shock) represent scenario stress tests, not calibrated forward-looking predictive probabilities.

---

## 11. Research Extensions

- **Intraday Tick-Level Event Studies:** Measuring quote-level price formation and order book imbalance in the first 60 minutes following news releases.
- **Implied Volatility Surfaces:** Extracting option-implied variance risk premia and skew around corporate governance announcements.
- **Cross-Sectional Panel Econometrics:** Expanding analysis to a multi-year panel of CEO turnover and corporate governance events.

---

## 12. Disclaimer

*This repository is created exclusively for quantitative research, financial econometrics education, and algorithmic modeling purposes. None of the materials, findings, or models contained herein constitute investment advice, financial recommendations, or an offer to buy or sell securities.*
