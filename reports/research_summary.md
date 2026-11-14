# Executive Quantitative Research Summary

**Project Title:** CEO Reputational Risk, Corporate Events & Quantitative Portfolio Risk Analysis  
**Research Window:** November 3, 2025 to November 13, 2026 (260 Trading Sessions)  
**Target Asset:** NVIDIA Corporation (`NVDA`) | **Market Benchmark:** S&P 500 ETF (`SPY`)  
**Core Portfolio Allocation:** NVDA (30%), META (20%), AMZN (20%), TSLA (15%), MSFT (15%)  

---

## 1. Executive Summary

This quantitative research framework examines corporate governance risk, executive reputational exposure, econometric volatility transmission, and quantitative downside risk management. Using historical daily market prices and trading volumes across mega-cap technology equities (`NVDA`, `META`, `AMZN`, `GOOGL`, `TSLA`, `AMD`, `AVGO`, `MSFT`) and benchmark exchange-traded funds (`SPY`, `QQQ`, `SOXX`), we evaluate how single-stock concentration and executive-level uncertainty transmit into portfolio-level risk.

### Verified Baseline Portfolio Findings
- **Total Return:** -4.07% over the 260-session sample period.
- **Annualized Realized Volatility:** 26.27% (assuming 252 trading sessions per annum).
- **Maximum Drawdown:** -18.72% (Peak: 2025-11-05, Trough: 2026-03-30, Recovered: 2026-04-27, 117 sessions).
- **Component Risk Contribution (Euler Decomposition):**
  - **NVDA (30.00% capital weight):** 33.74% risk contribution ($MCR = 0.2955$, Risk-to-Weight ratio = 1.12)
  - **META (20.00% capital weight):** 21.00% risk contribution ($MCR = 0.2758$, Risk-to-Weight ratio = 1.05)
  - **AMZN (20.00% capital weight):** 16.92% risk contribution ($MCR = 0.2222$, Risk-to-Weight ratio = 0.85)
  - **TSLA (15.00% capital weight):** 18.23% risk contribution ($MCR = 0.3192$, Risk-to-Weight ratio = 1.22)
  - **MSFT (15.00% capital weight):** 10.12% risk contribution ($MCR = 0.1773$, Risk-to-Weight ratio = 0.67)
  - **Sum of Contributions:** Strictly 100.00%.

---

## 2. Research Objectives & Hypotheses

1. **Identification Challenge:** Disentangle market-wide equity movements, sector-wide semiconductor dynamics, and firm-specific reputational announcements.
2. **Risk Attribution:** Quantify whether equal-weighted or capital-weighted holdings understate actual risk concentration when assets exhibit asymmetric volatility and cross-correlation.
3. **Downside Cycle Dynamics:** Disaggregate continuous macroeconomic drawdown cycles from superficial threshold crossings.
4. **Stress Testing:** Measure portfolio vulnerability under correlated joint shocks versus isolated idiosyncratic CEO headline events.

---

## 3. Dataset & Data Provenance

The primary quantitative dataset consists of daily adjusted closing prices, open, high, low, and volume series acquired via Yahoo Finance with automated local caching:
- **Observation Frequency:** Daily trading sessions (Monday through Friday, excluding NYSE holidays).
- **Sample Length:** 260 consecutive valid trading sessions.
- **Data Quality Audit:**
  - Zero missing values across the 11 tickers during the active window.
  - Zero non-positive or corrupted prices.
  - Strict timezone normalization and deduplication.

---

## 4. Quantitative & Econometric Methodology

### 4.1 Euler's Homogeneous Risk Decomposition
Portfolio volatility $\sigma_p$ is a linearly homogeneous function of degree 1 with respect to portfolio asset weight vector $w$:
$$\sigma_p(w) = \sqrt{w^T \Sigma w}$$

By Euler's homogeneous function theorem:
$$\sigma_p = \sum_{i=1}^N w_i \frac{\partial \sigma_p}{\partial w_i}$$

The **Marginal Contribution to Risk (MCR)** of asset $i$ is:
$$MCR_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

The **Absolute Component Risk Contribution ($CR_i$)** and **Percentage Contribution to Risk ($PCR_i$)** are:
$$CR_i = w_i \cdot MCR_i, \quad PCR_i = \frac{CR_i}{\sigma_p} = \frac{w_i (\Sigma w)_i}{\sigma_p^2}$$
$$\sum_{i=1}^N PCR_i = 1.0 \quad (100\%)$$

### 4.2 GARCH(1,1) Conditional Volatility
Time-varying conditional variance $\sigma_t^2$ is estimated using a GARCH(1,1) specification with Student's $t$ distributed errors:
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim t_\nu(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- **Persistence Parameter:** $\lambda = \alpha + \beta = 0.736$
- **Half-Life of Volatility Shocks:** $\tau_{1/2} = \frac{\ln(0.5)}{\ln(\alpha + \beta)} \approx 2.3 \text{ trading sessions}$.

### 4.3 Drawdown Cycle Segmentation
A common methodological flaw in portfolio analytics is treating repeated oscillations around a fixed loss threshold (e.g. -5% or -10%) as independent drawdown events. 

In this repository, an economic drawdown episode is strictly defined as a continuous cycle:
1. **Initiation:** The session where cumulative wealth drops below its prior all-time high ($W_t < M_t = \max_{s \le t} W_s$).
2. **Trough:** The session where underwater loss reaches its minimum value within the cycle.
3. **Recovery:** The session where cumulative wealth first equals or exceeds the previous peak value ($W_t \ge M_t$).

#### Audited Drawdown Cycles:
| Episode | Peak Date | Trough Date | Recovery Date | Max DD | Duration (Sessions) | Status |
|---|---|---|---|---|---|---|
| **Episode 1** | 2025-11-05 | 2026-03-30 | 2026-04-27 | **-18.72%** | 117 | Recovered |
| **Episode 2** | 2026-05-14 | 2026-07-29 | 2026-09-03 | **-15.12%** | 77 | Recovered |
| **Episode 3** | 2026-10-06 | 2026-11-05 | N/A | **-12.15%** | 28 | Unrecovered |
| **Episode 4** | 2026-04-27 | 2026-04-30 | 2026-05-08 | **-4.53%** | 9 | Recovered |
| **Episode 5** | 2026-09-03 | 2026-09-16 | 2026-09-21 | **-2.19%** | 11 | Recovered |

---

## 5. Downside Tail Risk Decomposition

Daily 1-day Value-at-Risk (VaR) and Expected Shortfall (ES / CVaR) were computed across parametric, non-parametric, and Cornish-Fisher formulations:

| Methodology | 95% 1-Day VaR | 95% 1-Day ES | 99% 1-Day VaR | 99% 1-Day ES |
|---|---|---|---|---|
| **Historical Empirical** | **2.62%** | **3.70%** | **4.48%** | **5.23%** |
| **Parametric Gaussian** | 2.72% | 3.42% | 3.85% | 4.41% |
| **Cornish-Fisher (Modified)** | 2.60% | — | 3.82% | — |

- **Sample Skewness:** -0.1865 (moderate left-tail asymmetry).
- **Excess Kurtosis:** +0.9899 (leptokurtic fat tails relative to normal distribution).

---

## 6. Stress Testing & Correlated Monte Carlo

To evaluate catastrophic reputational risk scenarios, 30,000 Monte Carlo paths were simulated by decomposing the empirical correlation matrix $\Sigma_{corr} = L L^T$ via Cholesky factorization:
- Assumed annual per-name reputational event probability: $p = 2.0\%$.
- Conditional event shock distribution: $\mathcal{N}(-6.0\%, (3.0\%)^2)$, truncated at -50%.
- **Simulated Mean Scenario Loss:** 0.22%
- **Simulated 95% Scenario VaR:** 1.06%
- **Simulated 95% Scenario Expected Shortfall:** 1.73%

---

## 7. Event Study Governance & Research Limitations

1. **Source Verification Requirement:** The event study module includes an explicit verification gate (`source_present & verified & include`). Provisional event records in the dataset currently lack verified source URLs and timestamp confirmations, and are preserved in standby mode to prevent unverified conclusions.
2. **Causal Confounding:** Single-company price movements around CEO announcements cannot be causally attributed solely to reputational damage without controlling for earnings surprises, semiconductor export controls, and macroeconomic interest rate shifts.
3. **Holdings Discipline:** Zero-weight assets (`GOOGL`, `AMD`, `AVGO`) in the expanded analysis universe must never be conflated with active portfolio holdings.

---

## 8. Quantitative Research Roadmap
- **High-Frequency Timestamping:** Transition from daily close data to intraday 1-minute order book and trades data.
- **Implied Volatility Surfaces:** Integrate options-implied skew and variance risk premia around corporate disclosures.
- **Cross-Sectional Panel Econometrics:** Expand from single-firm case studies to a multi-year panel of corporate governance events.

