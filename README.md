<div align="center">

# 🏔️ Alpin Re — Portfolio Lab

### Surplus optimization · Black–Litterman views · Catastrophe liquidity stress

*An asset–liability management (ALM) decision tool for a US property-catastrophe reinsurer*

![Python](https://img.shields.io/badge/Python-3.10%2B-14213D?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-3A5A8C?logo=streamlit&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-SLSQP%20%7C%20HiGHS-7A9CC6?logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-5C6B73?logo=plotly&logoColor=white)
![Course](https://img.shields.io/badge/HEC%20Lausanne-QARM%20II-C9A227)

**🔗 Live app:** `https://alpinre.streamlit.app`

</div>

---

## 📌 Overview

**Alpin Re America Inc.** is a fictional US property-catastrophe reinsurer. It covers other insurers against large losses, mainly from **US hurricanes**, and its balance sheet is in USD.

Its investment portfolio has **two jobs**:

| # | Job | What it means |
|---|-----|---------------|
| 1 | 💸 **Pay what it owes** | Claims must be paid on time, even right after a major hurricane, and assets must follow the value of reserves when rates move. |
| 2 | 📈 **Grow the surplus** | Earn a return on the capital that belongs to shareholders, with controlled risk. |

Both are solved in **one optimization of the surplus**: *assets minus reserves*.

---

## 🧭 What the app does

| Tab | Content |
|-----|---------|
| **1 · Client & rules** | Balance sheet, reserve payment pattern, investment universe, and Alpin Re's investment-policy limits. |
| **2 · Views** | Editable Black–Litterman views (absolute and relative) on top of a policy-implied prior. |
| **3 · Recommended portfolio** | Optimal weights, trades in USD, rule compliance (current vs recommended), risk contributions, surplus efficient frontier. |
| **4 · Catastrophe stress** | Hurricane + market-shock scenarios with a day-by-day liquidity waterfall, breaking points, and effect on the surplus. |
| **5 · Additional analysis** | Sensitivity to risk aversion, duration tolerance, turnover and Black–Litterman τ; VaR / expected-shortfall table. |
| **6 · Assumptions & limits** | What the model assumes and what it leaves out. |

---

## 🧮 Quant methodology

### 1. Surplus as the object to optimize

With assets $A$, reserves $L = kA$ and surplus $S = A - L$, the surplus return per dollar of assets is

$$
r_S = w^\top r - k\,h^\top r = (w - k\,h)^\top r ,
$$

where $w$ are the portfolio weights and $h$ is the Treasury mix whose duration matches the reserves (built from the payout pattern). The return on the surplus itself is scaled by $A/S$.

### 2. Reserve duration

From the payout pattern $p_t$ and the discount rate $y$, with mid-year timing $t = 1\!-\!\tfrac12,\,2\!-\!\tfrac12,\dots$

$$
D_L = \frac{1}{1+y}\cdot\frac{\sum_t t\,p_t\,(1+y)^{-t}}{\sum_t p_t\,(1+y)^{-t}}
$$

### 3. Black–Litterman

**Prior** — the excess returns that make Alpin Re's own *policy mix* optimal for its surplus objective:

$$
\Pi = \delta\,\Sigma\,(w_{policy} - k\,h)
$$

**Views** — $P\mu = Q + \varepsilon$, with $\Omega_{ii} = \tau_i\,(P\Sigma P^\top)_{ii}$ (a small view-$\tau$ means high confidence).

**Posterior**

$$
\mu_{BL} = \Pi + \tau\Sigma P^\top\big(P\,\tau\Sigma\,P^\top + \Omega\big)^{-1}\,(Q - P\Pi)
$$

Default views are the one-year yields to maturity of the four bond funds (dated, objective, and editable).

### 4. Portfolio optimization

$$
\max_{w}\;\; \mu^\top (w - kh) \;-\; \frac{\lambda}{2}\,(w - kh)^\top \Sigma\,(w - kh)
$$

Solved with SciPy (`SLSQP`), subject to:

- fully invested, long-only, per-fund caps
- **liquidity**: cash + short-term Treasuries $\geq$ coverage $\times$ claims due in 12 months
- **duration gap**: $|D_A - D_L| \leq$ tolerance
- **equities** $\leq$ limit, **credit** (corporate + mortgage) $\leq$ limit
- **turnover**: $\tfrac12\sum_i |w_i - w_i^{current}| \leq$ limit

Feasibility is checked with a linear program (HiGHS). The *lowest-risk* portfolio and the efficient frontier are built with the same machinery.

### 5. Risk measures

- **Risk contributions** (surplus variance): $RC_i = \dfrac{a_i\,(\Sigma a)_i}{a^\top \Sigma a}$ with $a = w - kh$
- **1-year expected shortfall** (normal approximation): $ES_{99\%} = -\mu_S + 2.665\,\sigma_S$
- **Historical 1-day VaR / ES** from daily returns
- **Interest-rate sensitivity** of the surplus: $\Delta S \approx -(D_A\,A - D_L\,L)\,\Delta y$

### 6. Catastrophe liquidity waterfall

1. Markets are shocked (equities in %, bonds through duration × rate and spread moves).
2. Claims fall due at **days 3, 30, 90 and 365**.
3. **Cash pays first**, then assets are sold in tiers (safest first), with T+1 settlement, forced-sale haircuts and policy sale limits.
4. Outputs: unpaid claims by date, the **breaking point** (largest hurricane paid on time, found by bisection), and the bridge of the surplus before and after.

---

## 🗂️ Project structure

```
AlpinRe-app/
├── app.py                  # Streamlit app (model + interface)
├── make_snapshot.py        # Downloads ETF prices into data/prices_snapshot.csv
├── requirements.txt        # Python dependencies
├── LogoAlpinRe.png         # Logo
└── data/
    └── prices_snapshot.csv # Offline price history (fallback if Yahoo fails)
```

---

## 🚀 Run locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/alpin-re-portfolio.git
cd alpin-re-portfolio

# 2. Install
pip install -r requirements.txt

# 3. (Optional) refresh the offline price snapshot
python make_snapshot.py

# 4. Launch
streamlit run app.py
```

> 💡 **Data source.** The app downloads live prices from Yahoo Finance. If the download fails (rate limits, no internet), it automatically falls back to `data/prices_snapshot.csv`.

---

## 🏦 Investment universe

| Ticker | Asset | Class | Max weight |
|--------|-------|-------|-----------:|
| `USD_CASH` | USD cash (T-bills) | Cash | 100 % |
| `VGSH` | Short-term Treasuries (1–3y) | Government bonds | 50 % |
| `VGIT` | Intermediate Treasuries (3–10y) | Government bonds | 40 % |
| `VMBS` | Agency mortgage-backed bonds | Mortgage-backed | 25 % |
| `VCIT` | Investment-grade corporate bonds | Corporate bonds | 25 % |
| `VTI` | US equities | Equity | 20 % |
| `VEA` | Developed ex-US equities | Equity | 10 % |

Durations and yields to maturity come from Vanguard fund data (31 Oct 2026). Weights, limits and durations are editable in the app.

---

## ⚠️ Assumptions & limits

- Reserves react **only to interest rates**: claims inflation and reserve uncertainty are not modelled.
- The liquidity test ignores new premiums, retrocession and credit lines, so results are **conservative**.
- Risk is estimated on the chosen price history (2y / 5y / 10y); another window gives other numbers.
- Surplus at risk uses a **normal approximation**; real tails are fatter.
- Regulatory and rating-agency capital models (Solvency / RBC / BCAR) are **not** reproduced.

---

## 🎓 Context

Course project for **Quantitative Asset and Risk Management II (QARM II)**, HEC Lausanne — built on the Black–Litterman model, risk contributions and portfolio optimization seen in class.


<div align="center">

*This is an educational project. Nothing here is investment advice.*

</div>
