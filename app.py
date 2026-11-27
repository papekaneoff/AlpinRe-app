import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize
from scipy.stats import norm

st.set_page_config(page_title="Alpin Re | Portfolio Lab", layout="wide")

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "LogoAlpinRe.png"
SNAPSHOT_PATH = APP_DIR / "data" / "prices_snapshot.csv"

# ---------------------------------------------------------------------------
# Investment universe (what a US P&C reinsurer actually holds, simplified)
# Durations and yields to maturity: Vanguard fund data as at 31 Oct 2026.
# ---------------------------------------------------------------------------
ETF_INFO = {
    "USD_CASH": {"name": "USD cash (T-bills)", "class": "Cash", "max": 1.00, "duration": 0.0, "ytm": None},
    "VGSH": {"name": "Short-term Treasuries (1–3y)", "class": "Government bonds", "max": 0.50, "duration": 1.8, "ytm": 4.6},
    "VGIT": {"name": "Intermediate Treasuries (3–10y)", "class": "Government bonds", "max": 0.40, "duration": 4.8, "ytm": 4.7},
    "VMBS": {"name": "Agency mortgage-backed bonds", "class": "Mortgage-backed bonds", "max": 0.25, "duration": 5.5, "ytm": 5.6},
    "VCIT": {"name": "Investment-grade corporate bonds", "class": "Corporate bonds", "max": 0.25, "duration": 5.9, "ytm": 5.8},
    "VTI": {"name": "US equities", "class": "Equity", "max": 0.20, "duration": 0.0, "ytm": None},
    "VEA": {"name": "Developed ex-US equities", "class": "Equity", "max": 0.10, "duration": 0.0, "ytm": None},
}
ASSET_ORDER = list(ETF_INFO)
MARKET_TICKERS = [ticker for ticker in ETF_INFO if ticker != "USD_CASH"]
CREDIT_CLASSES = ("Mortgage-backed bonds", "Corporate bonds")

# Alpin Re's strategic policy mix (Black–Litterman starting point).
POLICY_WEIGHTS = {"USD_CASH": 0.15, "VGSH": 0.20, "VGIT": 0.20, "VMBS": 0.12, "VCIT": 0.18, "VTI": 0.10, "VEA": 0.05}
# Alpin Re's holdings today: too few liquid assets, too long, too much in one corporate bond fund.
CURRENT_HOLDINGS = {"USD_CASH": 0.06, "VGSH": 0.06, "VGIT": 0.28, "VMBS": 0.12, "VCIT": 0.28, "VTI": 0.14, "VEA": 0.06}

# Share of today's net reserves paid in each future year. Short-tail reinsurers pay roughly 20-30% of
# their reserves per year (e.g. Arch 2024: ~29%; RenaissanceRe 2025: incurred ~25% of reserves).
DEFAULT_PAYOUT_PATTERN = [25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0]

# Liquidation assumptions. US ETFs settle T+1 since 28 May 2024.
LIQUIDATION_DEFAULTS = {
    "Government bonds": {"tier": 1, "settlement": 1, "haircut": 0.25, "sale_limit": 100.0},
    "Mortgage-backed bonds": {"tier": 2, "settlement": 1, "haircut": 0.75, "sale_limit": 75.0},
    "Corporate bonds": {"tier": 2, "settlement": 1, "haircut": 1.0, "sale_limit": 75.0},
    "Equity": {"tier": 3, "settlement": 1, "haircut": 0.5, "sale_limit": 50.0},
}

STRESS_SCENARIO_DEFAULTS = [
    {"Scenario": "Moderate hurricane", "NatCat claims (% of invested assets)": 5.0, "Equity shock (%)": -10.0,
     "Rate shock (bp)": -25.0, "Credit-spread shock (bp)": 50.0, "MBS-spread shock (bp)": 20.0,
     "Extra haircut (pp)": 0.0, "Sale-capacity multiplier (%)": 100.0, "Extra settlement days": 0},
    {"Scenario": "Severe hurricane", "NatCat claims (% of invested assets)": 10.0, "Equity shock (%)": -25.0,
     "Rate shock (bp)": -50.0, "Credit-spread shock (bp)": 150.0, "MBS-spread shock (bp)": 50.0,
     "Extra haircut (pp)": 1.0, "Sale-capacity multiplier (%)": 75.0, "Extra settlement days": 1},
    {"Scenario": "Hurricane + financial crisis", "NatCat claims (% of invested assets)": 15.0, "Equity shock (%)": -35.0,
     "Rate shock (bp)": 100.0, "Credit-spread shock (bp)": 250.0, "MBS-spread shock (bp)": 100.0,
     "Extra haircut (pp)": 3.0, "Sale-capacity multiplier (%)": 60.0, "Extra settlement days": 2},
]

ES_LEVEL = 0.99
ES_FACTOR = float(norm.pdf(norm.ppf(ES_LEVEL)) / (1 - ES_LEVEL))  # ≈ 2.665 for a normal distribution
COLORWAY = ["#14213D", "#3A5A8C", "#7A9CC6", "#B8C7DC", "#5C6B73", "#9DA9B0", "#C9A227", "#2E8B57"]
PLOT_LAYOUT = {"colorway": COLORWAY, "font": {"size": 13}}


def fmt_musd(value):
    return f"USD {value / 1e6:,.1f}m"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def normalize_daily_index(frame):
    frame = frame.copy()
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame


def download_live_prices(tickers, period):
    """One batched Yahoo download (no per-ticker .info calls, which are rate limited)."""
    raw = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True, timeout=20)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return normalize_daily_index(close)


def load_snapshot(period):
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"No price snapshot found at {SNAPSHOT_PATH.name}.")
    snapshot = normalize_daily_index(pd.read_csv(SNAPSHOT_PATH, index_col=0, parse_dates=True)).sort_index()
    start = snapshot.index.max() - pd.DateOffset(years=int(period.rstrip("y")))
    return snapshot.loc[snapshot.index >= start]


def clean_prices(close, tickers):
    series, errors = {}, {}
    for ticker in tickers:
        if ticker not in close or close[ticker].dropna().shape[0] < 60:
            errors[ticker] = "No or insufficient price history"
            continue
        series[ticker] = close[ticker].dropna()
    if len(series) < len(tickers):
        raise ValueError("Missing tickers: " + ", ".join(errors))
    prices = pd.concat(series, axis=1).sort_index().ffill(limit=3).dropna()
    if len(prices) < 60:
        raise ValueError("Fewer than 60 common observations across the ETF prices.")
    return prices


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers, period, source):
    """Returns prices, the source actually used and a note."""
    if source == "Live (Yahoo Finance)":
        try:
            return clean_prices(download_live_prices(tickers, period), tickers), "Live (Yahoo Finance)", ""
        except Exception as exc:
            note = f"Live download failed ({str(exc)[:90]}); using the stored snapshot."
            return clean_prices(load_snapshot(period), tickers), "Snapshot (fallback)", note
    return clean_prices(load_snapshot(period), tickers), "Snapshot", ""


# ---------------------------------------------------------------------------
# Black–Litterman
# ---------------------------------------------------------------------------
def black_litterman_returns(cov, implied_excess, risk_free_rate, P, Q, view_taus, prior_tau):
    """Posterior expected returns. implied_excess is the prior (returns that make the
    client's policy mix optimal for its own objective)."""
    asset_count = len(implied_excess)
    P = np.atleast_2d(np.asarray(P, dtype=float)).reshape(-1, asset_count)
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    view_taus = np.atleast_1d(np.asarray(view_taus, dtype=float))
    if prior_tau <= 0 or np.any(view_taus <= 0):
        raise ValueError("Prior tau and view taus must be positive.")
    if len(Q):
        omega = np.diag(view_taus * np.diag(P @ cov @ P.T))
        prior_cov = prior_tau * cov
        view_excess = Q - risk_free_rate * P.sum(axis=1)
        posterior_excess = implied_excess + prior_cov @ P.T @ np.linalg.solve(
            P @ prior_cov @ P.T + omega, view_excess - P @ implied_excess
        )
    else:
        posterior_excess = implied_excess.copy()
    return posterior_excess + risk_free_rate, implied_excess + risk_free_rate


# ---------------------------------------------------------------------------
# Surplus optimisation
# ---------------------------------------------------------------------------
# Surplus return (per dollar of assets): r_S = w'r - k * h'r, with k = reserves / assets and
# h the Treasury mix that behaves like the reserves. So the surplus is just the portfolio a = w - k h.
def liability_proxy(liability_duration, durations, cash_index, short_index, mid_index):
    """Mix of cash / short Treasuries / intermediate Treasuries with the liabilities' duration."""
    h = np.zeros(len(durations))
    d_short, d_mid = durations[short_index], durations[mid_index]
    if liability_duration <= d_short:
        share = liability_duration / d_short if d_short > 0 else 0.0
        h[short_index], h[cash_index] = share, 1 - share
    elif liability_duration <= d_mid:
        share = (d_mid - liability_duration) / (d_mid - d_short)
        h[short_index], h[mid_index] = share, 1 - share
    else:
        h[mid_index] = 1.0
    return h


def mandate_inequalities(mandate, asset_count):
    """Linear rules in the space [weights, |trade| auxiliaries]."""
    zeroes = np.zeros(asset_count)
    rows, limits = [], []
    for mask, limit in mandate["groups"]:
        rows.append(np.r_[mask, zeroes])
        limits.append(limit)
    identity = np.eye(asset_count)
    current = mandate["current"]
    for index in range(asset_count):
        rows.append(np.r_[identity[index], -identity[index]])
        limits.append(current[index])
        rows.append(np.r_[-identity[index], -identity[index]])
        limits.append(-current[index])
    rows.append(np.r_[zeroes, np.ones(asset_count)])
    limits.append(2 * mandate["max_turnover"])
    return np.asarray(rows), np.asarray(limits)


def solve_linear(objective, lower, upper, mandate):
    asset_count = len(lower)
    rows, limits = mandate_inequalities(mandate, asset_count)
    return linprog(
        c=objective, A_ub=rows, b_ub=limits,
        A_eq=np.array([np.r_[np.ones(asset_count), np.zeros(asset_count)]]), b_eq=np.array([1.0]),
        bounds=list(zip(lower, upper)) + [(0, None)] * asset_count, method="highs",
    )


def surplus_risk_and_return(weights, mu, cov, hedge):
    """Expected surplus return and surplus volatility, per dollar of assets."""
    active = np.asarray(weights) - hedge
    return float(active @ mu), float(np.sqrt(max(active @ cov @ active, 0.0)))


def expected_shortfall(weights, mu, cov, hedge):
    expected, volatility = surplus_risk_and_return(weights, mu, cov, hedge)
    return -expected + ES_FACTOR * volatility


def solve_surplus(mu, cov, hedge, lower, upper, mandate, initial, risk_aversion=4.0,
                  mode="utility", risk_budget=None, target_return=None):
    """mode: 'utility' (max mu'w - lambda/2 Var_S), 'min_variance' or 'min_es'.
    hedge = k * h, risk_budget in % of assets (expected shortfall limit)."""
    asset_count = len(mu)
    zeroes = np.zeros(asset_count)
    rows, limits = mandate_inequalities(mandate, asset_count)
    equality_rows, equality_targets = [np.r_[np.ones(asset_count), zeroes]], [1.0]
    if target_return is not None:
        equality_rows.append(np.r_[mu, zeroes])
        equality_targets.append(target_return)
    constraints = [
        LinearConstraint(np.asarray(equality_rows), equality_targets, equality_targets),
        LinearConstraint(rows, -np.inf, limits),
    ]

    def surplus_vol(values):
        active = values[:asset_count] - hedge
        return np.sqrt(max(active @ cov @ active, 1e-14))

    def es_value(values):
        return -(values[:asset_count] - hedge) @ mu + ES_FACTOR * surplus_vol(values)

    def es_gradient(values):
        active = values[:asset_count] - hedge
        return np.r_[-mu + ES_FACTOR * (cov @ active) / surplus_vol(values), zeroes]

    if risk_budget is not None:
        constraints.append(NonlinearConstraint(es_value, -np.inf, risk_budget, jac=es_gradient))

    def objective(values):
        active = values[:asset_count] - hedge
        if mode == "min_variance":
            return active @ cov @ active
        if mode == "min_es":
            return es_value(values)
        return -mu @ values[:asset_count] + 0.5 * risk_aversion * (active @ cov @ active)

    def gradient(values):
        active = values[:asset_count] - hedge
        if mode == "min_variance":
            return np.r_[2 * cov @ active, zeroes]
        if mode == "min_es":
            return es_gradient(values)
        return np.r_[-mu + risk_aversion * (cov @ active), zeroes]

    initial = np.asarray(initial, dtype=float)
    result = minimize(
        objective, x0=np.r_[initial, np.abs(initial - mandate["current"])], jac=gradient, method="SLSQP",
        bounds=Bounds(np.r_[lower, zeroes], np.r_[upper, np.full(asset_count, np.inf)]),
        constraints=constraints, options={"maxiter": 1000, "ftol": 1e-11},
    )
    result.x = result.x[:asset_count]
    return result


def relative_risk_contributions(weights, cov, hedge):
    active = np.asarray(weights) - hedge
    variance = float(active @ cov @ active)
    if variance <= 0:
        return np.zeros_like(active)
    return active * (cov @ active) / variance


def liability_duration(payout_shares, discount_rate):
    payout_shares = np.asarray(payout_shares, dtype=float)
    times = np.arange(1, len(payout_shares) + 1) - 0.5
    present_values = payout_shares * (1.0 + discount_rate) ** (-times)
    macaulay = float((times * present_values).sum() / present_values.sum())
    return macaulay / (1.0 + discount_rate)


def historical_var_es(daily_returns, level):
    losses = -np.asarray(daily_returns, dtype=float)
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    return var, float(tail.mean()) if len(tail) else var


# ---------------------------------------------------------------------------
# Catastrophe liquidity stress
# ---------------------------------------------------------------------------
def catastrophe_liquidity_waterfall(portfolio_values, cash_index, claims_by_period, period_end_days,
                                    settlement_days, haircuts, sale_fractions, tiers):
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    sale_limits = portfolio_values * sale_fractions
    sold = np.zeros_like(portfolio_values)
    cash_remaining = float(portfolio_values[cash_index])
    cash_used_total = unpaid = cumulative_claims = net_proceeds_total = haircut_loss = 0.0
    periods = []
    for claims, horizon in zip(claims_by_period, period_end_days):
        cumulative_claims += claims
        to_fund = unpaid + claims
        cash_used = min(to_fund, cash_remaining)
        cash_remaining -= cash_used
        cash_used_total += cash_used
        remaining = to_fund - cash_used
        eligible = (settlement_days <= horizon) & (np.arange(len(portfolio_values)) != cash_index)
        capacity_left = np.maximum(sale_limits - sold, 0.0)
        for tier in sorted(np.unique(tiers[eligible])):
            tier_capacity = np.where(eligible & (tiers == tier), capacity_left * (1.0 - haircuts), 0.0)
            total = float(tier_capacity.sum())
            raise_now = min(remaining, total)
            if total <= 0 or raise_now <= 0:
                continue
            net = tier_capacity * (raise_now / total)  # pro rata within a tier
            gross = np.divide(net, 1.0 - haircuts, out=np.zeros_like(net), where=(1.0 - haircuts) > 0)
            sold += gross
            net_proceeds_total += float(net.sum())
            haircut_loss += float(gross.sum() - net.sum())
            remaining -= raise_now
            if remaining <= 1e-8:
                break
        unpaid = max(remaining, 0.0)
        periods.append({
            "Horizon": f"Day {int(horizon)}",
            "Claims due (USD)": float(claims),
            "Cash used cumulative (USD)": cash_used_total,
            "Net sale proceeds cumulative (USD)": net_proceeds_total,
            "Unpaid claims (USD)": unpaid,
        })
    post = portfolio_values - sold
    post[cash_index] -= cash_used_total
    return {"sold": sold, "haircut_loss": haircut_loss, "post_event_values": post, "periods": periods}


def shocked_returns(tickers, durations, scenario):
    rate = float(scenario["Rate shock (bp)"]) / 10_000
    changes = np.zeros(len(tickers))
    for index, ticker in enumerate(tickers):
        asset_class = ETF_INFO[ticker]["class"]
        if asset_class == "Equity":
            changes[index] = float(scenario["Equity shock (%)"]) / 100
        elif asset_class == "Government bonds":
            changes[index] = -durations[index] * rate
        elif asset_class == "Mortgage-backed bonds":
            changes[index] = -durations[index] * (rate + float(scenario["MBS-spread shock (bp)"]) / 10_000)
        elif asset_class == "Corporate bonds":
            changes[index] = -durations[index] * (rate + float(scenario["Credit-spread shock (bp)"]) / 10_000)
    return np.maximum(changes, -1.0)


# ---------------------------------------------------------------------------
# Header, alerts, summary area
# ---------------------------------------------------------------------------
logo_html = ""
if LOGO_PATH.exists():
    logo_html = f'<img src="data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode()}" style="height:68px;">'
st.markdown(
    f"""
    <div style="background:#14213D;padding:18px 28px;border-radius:6px;display:flex;align-items:center;gap:28px;margin-bottom:8px;">
      {logo_html}
      <div>
        <div style="color:#FFFFFF;font-size:1.85rem;font-weight:600;line-height:1.2;">Alpin Re — Investment Portfolio Strategy</div>
        <div style="color:#C9D1E0;font-size:1rem;margin-top:4px;">Surplus optimization with Black–Litterman views and a catastrophe liquidity test</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
alerts = st.container()
kpi_area = st.container()


def fail(message, details=None, hint=None):
    with alerts:
        st.error(message)
        if details is not None:
            st.dataframe(details, hide_index=True, width="stretch")
        if hint:
            st.info(hint)
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: Alpin Re's inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Alpin Re inputs")
    data_source = st.radio("Market data source", ["Live (Yahoo Finance)", "Snapshot (offline)"],
                           help="Live data falls back to the stored snapshot if Yahoo fails.")
    if st.button("Refresh market data", width="stretch"):
        load_prices.clear()
    period = st.selectbox("Price history for risk estimates", ["2y", "5y", "10y"], index=1)
    st.divider()
    st.caption("BALANCE SHEET")
    invested_assets_m = st.number_input("Invested assets (USD m)", 100, 50_000, 2_000, 100)
    reserves_pct = st.slider("Net loss reserves (% of invested assets)", 10, 90, 70,
                             help="Surplus = invested assets − reserves (simplified balance sheet).")
    st.divider()
    st.caption("INVESTMENT POLICY")
    liquidity_coverage = st.slider(
        "Liquidity floor (× claims due in 12 months)", 1.0, 2.0, 1.0, step=0.05,
        help="Cash + short-term Treasuries must cover this multiple of the claims due in the next 12 months "
             "(year-1 payments of the reserves).",
    )
    duration_tolerance = st.slider(
        "Duration gap tolerance (± years)", 0.25, 3.0, 1.0, step=0.25,
        help="The average duration of the assets must stay within this distance of the reserve duration.",
    )
    max_equity = st.slider("Maximum equities (%)", 0, 100, 20) / 100
    max_credit = st.slider("Maximum credit: corporate + mortgage bonds (%)", 0, 100, 40) / 100
    max_turnover = st.slider("Maximum one-way turnover per rebalance (%)", 0, 100, 20) / 100
    risk_aversion = st.slider("Portfolio risk aversion (λ)", 0.5, 30.0, 8.0, step=0.5,
                              help="Alpin Re's trade-off between surplus return and surplus risk, used by the "
                                   "optimizer. Higher = more cautious.")
    st.divider()
    st.caption("BLACK–LITTERMAN")
    prior_delta = st.slider("Prior risk aversion (δ)", 0.5, 30.0, 8.0, step=0.5,
                            help="Calibration of the starting point: the risk aversion under which Alpin Re's "
                                 "policy mix is optimal. It sets the level of the policy-implied returns. "
                                 "With no views and δ = λ, the optimizer returns the policy mix (if no rule binds).")
    prior_tau = st.slider("Prior uncertainty (tau)", 0.005, 0.20, 0.05, step=0.005)
    risk_free_rate = st.number_input("USD risk-free rate (% p.a.)", -5.0, 15.0, 4.0, 0.25) / 100
    with st.expander("Claims payment timing"):
        st.caption("Share of claims paid by each date; the rest falls in days 91–365.")
        expected_profile_inputs = [
            st.number_input("Expected claims by day 3 (%)", 0, 100, 5, 5),
            st.number_input("Expected claims days 4–30 (%)", 0, 100, 20, 5),
            st.number_input("Expected claims days 31–90 (%)", 0, 100, 30, 5),
        ]
        cat_profile_inputs = [
            st.number_input("Hurricane claims by day 3 (%)", 0, 100, 15, 5),
            st.number_input("Hurricane claims days 4–30 (%)", 0, 100, 30, 5),
            st.number_input("Hurricane claims days 31–90 (%)", 0, 100, 30, 5),
        ]

capital = float(invested_assets_m) * 1e6
k = reserves_pct / 100
reserves = capital * k
surplus = capital - reserves
to_surplus = capital / surplus  # converts "% of assets" into "% of surplus"
if sum(expected_profile_inputs) > 100 or sum(cat_profile_inputs) > 100:
    fail("Claims paid through day 90 cannot exceed 100%.")
expected_claims_profile = np.array(expected_profile_inputs + [100 - sum(expected_profile_inputs)], dtype=float) / 100
cat_claims_profile = np.array(cat_profile_inputs + [100 - sum(cat_profile_inputs)], dtype=float) / 100

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading ETF prices..."):
        prices, source_used, source_note = load_prices(tuple(MARKET_TICKERS), period, data_source)
except Exception as exc:
    fail(f"Market data unavailable: {exc}",
         hint="Run make_snapshot.py once with internet access and commit data/prices_snapshot.csv.")
if source_note:
    with alerts:
        st.warning(source_note)

prices = prices.copy()
prices["USD_CASH"] = 100.0 * np.power(1.0 + risk_free_rate, np.arange(len(prices)) / 252)
available = ASSET_ORDER
prices = prices[available]
asset_count = len(available)
cash_index = available.index("USD_CASH")
data_as_of = prices.index.max().date()
returns_usd = prices.pct_change(fill_method=None).dropna()
cov = returns_usd.cov().to_numpy() * 252
cov = (cov + cov.T) / 2
eigenvalues, eigenvectors = np.linalg.eigh(cov)
cov = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T

tab_client, tab_views, tab_portfolio, tab_stress, tab_more, tab_assumptions = st.tabs([
    "1 · Client & rules",
    "2 · Views",
    "3 · Recommended portfolio",
    "4 · Catastrophe stress",
    "5 · Additional analysis",
    "6 · Assumptions & limits",
])

# ---------------------------------------------------------------------------
# Tab 1 — client and rules
# ---------------------------------------------------------------------------
with tab_client:
    st.subheader("Alpin Re")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            """
            **Alpin Re America Inc.** is a US property-catastrophe reinsurer: it covers other
            insurers against large losses, mainly from **US hurricanes**. Its balance sheet is in USD.

            **Its money has two jobs**
            1. **Pay what it owes**: claims must be paid on time, even right after a major hurricane,
               and the value of the assets must follow the value of the reserves when rates move.
            2. **Grow its own capital (the surplus)** with controlled risk —
               most of the surplus is already exposed to hurricanes.

            We solve both in **one surplus optimization**: assets minus reserves.
            """
        )
    with right:
        st.markdown("**Reserve payment pattern** (share of today's reserves paid each year)")
        payout_pattern = st.data_editor(
            pd.DataFrame({"Year": [f"Year {year}" for year in range(1, len(DEFAULT_PAYOUT_PATTERN) + 1)],
                          "Share paid (%)": DEFAULT_PAYOUT_PATTERN}),
            hide_index=True, width="stretch", disabled=["Year"], key="payout_pattern_v2",
            column_config={"Share paid (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0)},
        )
    payout_values = payout_pattern["Share paid (%)"].to_numpy(dtype=float)
    if not np.isfinite(payout_values).all() or not np.isclose(payout_values.sum(), 100.0, atol=1e-6):
        fail(f"The reserve payment pattern must sum to 100% (currently {payout_values.sum():.1f}%).")
    liability_mod_duration = liability_duration(payout_values / 100, risk_free_rate)
    # Claims due in the next 12 months = year-1 payments of today's reserves (one source of truth).
    expected_claims_12m = reserves * payout_values[0] / 100
    liquid_floor = liquidity_coverage * expected_claims_12m / capital
    if liquid_floor > 1:
        fail("The liquidity floor exceeds 100% of invested assets.")
    duration_low = max(liability_mod_duration - duration_tolerance, 0.0)
    duration_high = liability_mod_duration + duration_tolerance

    balance = st.columns(5)
    balance[0].metric("Invested assets", fmt_musd(capital))
    balance[1].metric("Net loss reserves", fmt_musd(reserves))
    balance[2].metric("Surplus", fmt_musd(surplus))
    balance[3].metric("Reserve duration", f"{liability_mod_duration:.2f} years")
    balance[4].metric("Claims due in 12 months", fmt_musd(expected_claims_12m),
                      f"{payout_values[0]:.0f}% of reserves", delta_color="off")

    st.subheader("Investment universe and Alpin Re's limits")
    st.caption("Policy mix = Alpin Re's strategic allocation (Black–Litterman starting point). "
               "Current = what Alpin Re holds today. Durations from Vanguard fund data (31 Oct 2026).")
    universe = st.data_editor(
        pd.DataFrame({
            "Ticker": available,
            "Asset": [ETF_INFO[ticker]["name"] for ticker in available],
            "Class": [ETF_INFO[ticker]["class"] for ticker in available],
            "Policy mix (%)": [POLICY_WEIGHTS[ticker] * 100 for ticker in available],
            "Current (%)": [CURRENT_HOLDINGS[ticker] * 100 for ticker in available],
            "Min (%)": [0.0] * asset_count,
            "Max (%)": [ETF_INFO[ticker]["max"] * 100 for ticker in available],
            "Duration (years)": [ETF_INFO[ticker]["duration"] for ticker in available],
        }),
        hide_index=True, width="stretch", disabled=["Ticker", "Asset", "Class"], key="universe_v3",
        column_config={
            **{column: st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
               for column in ["Policy mix (%)", "Current (%)", "Min (%)", "Max (%)"]},
            "Duration (years)": st.column_config.NumberColumn(min_value=0.0, max_value=30.0, step=0.1, format="%.1f"),
        },
    )
    rules_table = pd.DataFrame([
        ("Liquid assets (cash + short Treasuries)", f"≥ {liquid_floor:.1%} of assets",
         "Pay a full year of claims without selling riskier assets"),
        ("Duration gap", f"asset duration {duration_low:.1f}–{duration_high:.1f} years "
                         f"(reserves {liability_mod_duration:.1f})", "Rates moving assets and reserves differently"),
        ("Equities", f"≤ {max_equity:.0%} of assets", "Market and hurricane losses can hit in the same year"),
        ("Credit (corporate + mortgage bonds)", f"≤ {max_credit:.0%} of assets", "Credit losses on top of hurricane losses"),
        ("Single asset", "Caps per fund (table above)", "No concentration in one fund"),
        ("Turnover", f"≤ {max_turnover:.0%} one-way per rebalance", "Limit trading costs, move gradually"),
    ], columns=["Rule", "Limit", "Risk it protects against"])
    st.dataframe(rules_table, hide_index=True, width="stretch")

policy = universe["Policy mix (%)"].to_numpy(dtype=float) / 100
current = universe["Current (%)"].to_numpy(dtype=float) / 100
asset_lower = universe["Min (%)"].to_numpy(dtype=float) / 100
upper = universe["Max (%)"].to_numpy(dtype=float) / 100
durations = universe["Duration (years)"].to_numpy(dtype=float)
lower = asset_lower.copy()
classes = universe["Class"].tolist()
equity_mask = np.array([c == "Equity" for c in classes], dtype=float)
credit_mask = np.array([c in CREDIT_CLASSES for c in classes], dtype=float)
liquid_mask = np.array([t in ("USD_CASH", "VGSH") for t in available], dtype=float)

for label, vector in (("Policy mix", policy), ("Current holdings", current)):
    if not np.isfinite(vector).all() or np.any(vector < 0) or not np.isclose(vector.sum(), 1.0, atol=1e-4):
        fail(f"{label} must be non-negative and sum to 100% (entered: {vector.sum():.2%}).")
if np.any(lower > upper):
    fail("A minimum weight (or the cash floor) exceeds its maximum.")
if lower.sum() > 1 + 1e-8 or upper.sum() < 1 - 1e-8:
    fail("The minimum and maximum weights cannot form a fully invested portfolio.")

liability_mix = liability_proxy(liability_mod_duration, durations, cash_index,
                                available.index("VGSH"), available.index("VGIT"))
hedge = k * liability_mix  # the "short liability" position inside the surplus

def build_mandate(liquid_minimum, tolerance=None, turnover=None):
    """All of Alpin Re's linear rules, written as mask · w <= limit."""
    tolerance = duration_tolerance if tolerance is None else tolerance
    turnover = max_turnover if turnover is None else turnover
    high, low = liability_mod_duration + tolerance, max(liability_mod_duration - tolerance, 0.0)
    return {
        "current": current,
        "groups": [
            (equity_mask, max_equity),
            (credit_mask, max_credit),
            (-liquid_mask, -liquid_minimum),   # cash + short Treasuries >= floor
            (durations, high),                 # asset duration <= reserve duration + tolerance
            (-durations, -low),                # asset duration >= reserve duration - tolerance
        ],
        "max_turnover": turnover,
    }


mandate = build_mandate(liquid_floor)

# ---------------------------------------------------------------------------
# Tab 2 — views
# ---------------------------------------------------------------------------
# Prior: the returns that make Alpin Re's policy mix optimal for its own surplus objective.
implied_excess = prior_delta * cov @ (policy - hedge)
view_columns = ["View type", "Asset", "Compared with", "Expected return / spread (%)", "View tau",
                "Source", "Date (YYYY-MM-DD)", "Rationale"]
default_views = []
for ticker, tau in (("VGSH", 0.02), ("VGIT", 0.02), ("VMBS", 0.10), ("VCIT", 0.05)):
    default_views.append({
        "View type": "Absolute", "Asset": ticker, "Compared with": None,
        "Expected return / spread (%)": ETF_INFO[ticker]["ytm"], "View tau": tau,
        "Source": "Vanguard fund data (yield to maturity)", "Date (YYYY-MM-DD)": "2026-10-31",
        "Rationale": ("Yield to maturity ≈ one-year return if yields do not change."
                      + (" Mortgage yields overstate returns (prepayments): lower confidence." if ticker == "VMBS" else "")),
    })

with tab_views:
    st.subheader("Alpin Re's views (Black–Litterman)")
    st.caption(
        "Fully customizable: add, change or delete views. Default examples: for each bond fund, "
        "the expected one-year return is its current yield to maturity — an objective, dated "
        "estimate. No default view on equities: they keep the policy-implied return."
    )
    views = st.data_editor(
        pd.DataFrame(default_views, columns=view_columns), hide_index=True, width="stretch",
        num_rows="dynamic", key="views_v3",
        column_config={
            "View type": st.column_config.SelectboxColumn(options=["Absolute", "Relative"], required=True),
            "Asset": st.column_config.SelectboxColumn(options=available, required=True),
            "Compared with": st.column_config.SelectboxColumn(options=available),
            "Expected return / spread (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=0.1, required=True),
            "View tau": st.column_config.NumberColumn(min_value=0.001, max_value=10.0, step=0.01, default=0.05, required=True),
        },
    )
    with st.expander("How the views enter the model"):
        st.markdown(
            r"""
            - **Starting point:** $\Pi = \delta\,\Sigma\,(w_{policy} - k\,h)$ — the excess returns that make
              Alpin Re's policy mix the best choice for its surplus (assets minus reserves).
            - **Views:** $P\mu = Q + \varepsilon$, with uncertainty $\Omega_{ii} = \tau_i (P\Sigma P^\top)_{ii}$.
              A small view tau means high confidence.
            - **Posterior:** $\mu_{BL} = \Pi + \tau\Sigma P^\top (P\tau\Sigma P^\top + \Omega)^{-1}(Q - P\Pi)$.
            """
        )

valid_views = views.dropna(subset=["View type", "Asset", "Expected return / spread (%)", "View tau"])
if (valid_views["View type"].eq("Relative") & valid_views["Compared with"].isna()).any():
    fail("Choose a comparison asset for each relative view.")
if (valid_views["View type"].eq("Absolute") & valid_views["Compared with"].notna()).any():
    with alerts:
        st.warning("'Compared with' is ignored for absolute views.")
view_rows = []
for _, view in valid_views.iterrows():
    row = np.zeros(asset_count)
    row[available.index(view["Asset"])] = 1.0
    if view["View type"] == "Relative":
        row[available.index(view["Compared with"])] -= 1.0
    view_rows.append(row)
P = np.array(view_rows, dtype=float).reshape(-1, asset_count)
Q = valid_views["Expected return / spread (%)"].to_numpy(dtype=float) / 100
view_taus = valid_views["View tau"].to_numpy(dtype=float)
if any(np.allclose(row, 0) for row in P):
    fail("A relative view must compare two different assets.")
try:
    mu, prior_returns = black_litterman_returns(cov, implied_excess, risk_free_rate, P, Q, view_taus, prior_tau)
except (ValueError, np.linalg.LinAlgError) as exc:
    fail(f"Could not compute Black–Litterman returns: {exc}")

with tab_views:
    returns_table = pd.DataFrame({
        "Ticker": available,
        "Policy-implied return (%)": prior_returns * 100,
        "After views (%)": mu * 100,
        "Volatility (%)": np.sqrt(np.diag(cov)) * 100,
    })
    st.dataframe(returns_table.style.format({c: "{:.2f}" for c in returns_table.columns if c != "Ticker"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
feasibility = solve_linear(np.zeros(2 * asset_count), lower, upper, mandate)
if not feasibility.success:
    fail("Alpin Re's limits cannot all be met from the current holdings.",
         hint="Relax turnover, the liquidity floor, the duration tolerance or a cap, then rerun.")
lowest_risk = solve_surplus(mu, cov, hedge, lower, upper, mandate, feasibility.x[:asset_count], mode="min_variance")
result = solve_surplus(mu, cov, hedge, lower, upper, mandate, lowest_risk.x, risk_aversion=risk_aversion)
if not result.success or not np.isfinite(result.x).all():
    fail(f"The optimizer did not converge ({result.message}).")
weights = result.x.copy()


def portfolio_stats(portfolio, expected=None):
    expected = mu if expected is None else expected
    surplus_return, surplus_vol = surplus_risk_and_return(portfolio, expected, cov, hedge)
    asset_duration = float(portfolio @ durations)
    return {
        "surplus_return": surplus_return * to_surplus,
        "surplus_vol": surplus_vol * to_surplus,
        "es": expected_shortfall(portfolio, expected, cov, hedge) * to_surplus,
        "asset_return": float(portfolio @ expected),
        "asset_duration": asset_duration,
        "surplus_duration": (asset_duration - k * liability_mod_duration) * to_surplus,
        "equity": float(equity_mask @ portfolio),
        "credit": float(credit_mask @ portfolio),
        "cash": float(portfolio[cash_index]),
        "liquid": float(liquid_mask @ portfolio),
        "turnover": float(0.5 * np.abs(portfolio - current).sum()),
    }


def compliance_checks(portfolio):
    stats = portfolio_stats(portfolio)
    return [
        ("Liquid assets", stats["liquid"] >= liquid_floor - 1e-6, f"{stats['liquid']:.1%} (min {liquid_floor:.1%})"),
        ("Duration gap", duration_low - 1e-6 <= stats["asset_duration"] <= duration_high + 1e-6,
         f"{stats['asset_duration']:.2f} y (allowed {duration_low:.2f}–{duration_high:.2f})"),
        ("Equities", stats["equity"] <= max_equity + 1e-6, f"{stats['equity']:.1%} (max {max_equity:.0%})"),
        ("Credit", stats["credit"] <= max_credit + 1e-6, f"{stats['credit']:.1%} (max {max_credit:.0%})"),
        ("Single-asset caps", bool(np.all(portfolio <= upper + 1e-6) and np.all(portfolio >= asset_lower - 1e-6)),
         f"largest excess {max((portfolio - upper).max(), 0):.1%}"),
        ("Turnover", stats["turnover"] <= max_turnover + 1e-6, f"{stats['turnover']:.1%} (max {max_turnover:.0%})"),
    ]


checks = compliance_checks(weights)
if not all(passed for _, passed, _ in checks):
    fail("The optimizer returned a portfolio that breaks a rule.",
         details=pd.DataFrame([{"Rule": n, "Status": "Compliant" if p else "Breach", "Value": d} for n, p, d in checks]))

recommended_stats = portfolio_stats(weights)
current_stats = portfolio_stats(current)
policy_stats = portfolio_stats(policy)
lowest_stats = portfolio_stats(lowest_risk.x)

# ---------------------------------------------------------------------------
# Tab 3 — recommended portfolio
# ---------------------------------------------------------------------------
def status_style(value):
    return "color: #9B1C1C; font-weight: 600; background-color: #F6E3E3;" if value == "Breach" else "color: #5C6B73;"


with tab_portfolio:
    st.subheader("Recommended portfolio")
    st.caption("One optimization of the surplus (assets − reserves): part of the portfolio behaves like "
               "the reserves, the rest earns a return with controlled risk.")
    cols = st.columns(4)
    cols[0].metric("Expected return on surplus", f"{recommended_stats['surplus_return']:.1%}",
                   f"{(recommended_stats['surplus_return'] - current_stats['surplus_return']) * 100:+.1f} pts vs current")
    cols[1].metric("Surplus volatility", f"{recommended_stats['surplus_vol']:.1%}",
                   f"{(recommended_stats['surplus_vol'] - current_stats['surplus_vol']) * 100:+.1f} pts vs current",
                   delta_color="inverse")
    cols[2].metric("Asset duration", f"{recommended_stats['asset_duration']:.2f} y",
                   f"reserves {liability_mod_duration:.2f} y · current {current_stats['asset_duration']:.2f} y",
                   delta_color="off")
    cols[3].metric("Surplus at risk (1-year ES 99%)", f"{recommended_stats['es']:.1%} of surplus",
                   f"current {current_stats['es']:.1%}", delta_color="off")

    comparison_figure = go.Figure()
    for label, portfolio, color in (("Current", current, "#B8C7DC"), ("Lowest risk within the rules", lowest_risk.x, "#7A9CC6"),
                                    ("Recommended", weights, "#14213D")):
        comparison_figure.add_trace(go.Bar(name=label, x=[ETF_INFO[t]["name"] for t in available],
                                           y=portfolio * 100, marker_color=color))
    comparison_figure.update_layout(title="Weights (%)", barmode="group", height=380,
                                    margin={"t": 48, "b": 12, "l": 12, "r": 12},
                                    legend={"orientation": "h", "y": 1.1}, **PLOT_LAYOUT)
    st.plotly_chart(comparison_figure, width="stretch")
    st.caption(
        f"'Lowest risk' is the portfolio that tracks the reserves as closely as Alpin Re's rules allow. "
        f"The recommended portfolio earns {(recommended_stats['surplus_return'] - lowest_stats['surplus_return']) * 100:.1f} "
        f"points more on the surplus by using {recommended_stats['es']:.1%} of the surplus as investment risk."
    )

    trades = pd.DataFrame({
        "Asset": [ETF_INFO[t]["name"] for t in available],
        "Current (%)": current * 100,
        "Recommended (%)": weights * 100,
        "Trade (USD)": (weights - current) * capital,
        "Share of surplus risk (%)": relative_risk_contributions(weights, cov, hedge) * 100,
    })
    st.dataframe(trades.style.format({
        "Current (%)": "{:.1f}", "Recommended (%)": "{:.1f}", "Share of surplus risk (%)": "{:.1f}",
        "Trade (USD)": lambda v: f"{'+' if v >= 0 else '−'}USD {abs(v) / 1e6:,.1f}m",
    }), hide_index=True, width="stretch")

    st.markdown("#### Alpin Re's rules: current vs recommended")
    compliance = pd.DataFrame([
        {"Rule": name, "Current": d_now, "Current status": "Compliant" if p_now else "Breach",
         "Recommended": d_new, "Recommended status": "Compliant" if p_new else "Breach"}
        for (name, p_now, d_now), (_, p_new, d_new) in zip(compliance_checks(current), checks)
    ])
    st.dataframe(compliance.style.map(status_style, subset=["Current status", "Recommended status"]),
                 hide_index=True, width="stretch")

    st.markdown("#### Where the recommendation sits: surplus efficient frontier")
    show_long_only_frontier = st.checkbox("Show long-only frontier (no policy rules)", value=False)
    with st.expander("How the portfolio optimization works"):
        st.markdown("The optimizer chooses asset weights to maximize expected return on the surplus "
                    "while penalizing surplus volatility:")
        st.latex(
            r"\max_{w}\quad \mu^\top(w-kh)"
            r"-\frac{\lambda}{2}(w-kh)^\top\Sigma(w-kh)"
        )
        st.markdown(
            "Here, **μ** is the expected asset-return vector, **Σ** is the return covariance matrix, "
            "**k** is reserves as a share of assets, and **h** is the Treasury mix used to represent the "
            "reserves. The term **w − kh** is the portfolio exposure relative to that reserve hedge; "
            "**λ** controls the trade-off between expected surplus return and risk. "
            "The solution is fully invested and must satisfy asset bounds and Alpin Re's policy rules: "
            "liquidity, duration, equity and credit limits, and maximum one-way turnover."
        )
    with st.spinner("Computing frontiers..."):
        long_only = {"current": current, "groups": [], "max_turnover": 1.0}
        zeros_lower, ones_upper = np.zeros(asset_count), np.ones(asset_count)

        def frontier(frontier_lower, frontier_upper, frontier_mandate, budget):
            top = solve_linear(np.r_[-mu, np.zeros(asset_count)], frontier_lower, frontier_upper, frontier_mandate)
            start = solve_linear(np.zeros(2 * asset_count), frontier_lower, frontier_upper, frontier_mandate)
            if not (top.success and start.success):
                return None
            base = solve_surplus(mu, cov, hedge, frontier_lower, frontier_upper, frontier_mandate,
                                 start.x[:asset_count], mode="min_variance")
            low, high = float(mu @ base.x), float(-top.fun)
            points, previous = [], base.x
            for target in np.linspace(low, high, 26):
                point = solve_surplus(mu, cov, hedge, frontier_lower, frontier_upper, frontier_mandate, previous,
                                      mode="min_variance", target_return=target)
                if point.success and abs(mu @ point.x - target) < 1e-5:
                    stats = portfolio_stats(point.x)
                    points.append((stats["surplus_vol"], stats["surplus_return"]))
                    previous = point.x
            return np.array(points) if points else None

        curves = [("Alpin Re's rules", frontier(lower, upper, mandate, None),
                   {"color": "#14213D", "width": 3})]
        if show_long_only_frontier:
            curves.insert(0, ("Long-only, no rules", frontier(zeros_lower, ones_upper, long_only, None),
                              {"color": "#9DA9B0", "dash": "dot"}))
    frontier_figure = go.Figure()
    for name, curve, style in curves:
        if curve is not None:
            frontier_figure.add_trace(go.Scatter(x=curve[:, 0] * 100, y=curve[:, 1] * 100, mode="lines", name=name, line=style))
    for label, stats, symbol, color in (("Current", current_stats, "x", "#B03A2E"),
                                        ("Lowest risk", lowest_stats, "circle", "#7A9CC6"),
                                        ("Recommended", recommended_stats, "diamond", "#C9A227")):
        frontier_figure.add_trace(go.Scatter(x=[stats["surplus_vol"] * 100], y=[stats["surplus_return"] * 100],
                                             mode="markers", name=label, marker={"symbol": symbol, "size": 13, "color": color}))
    frontier_figure.update_layout(xaxis_title="Surplus volatility (% of surplus)",
                                  yaxis_title="Expected return on surplus (%)", height=440,
                                  margin={"t": 24, "b": 16, "l": 16, "r": 16},
                                  legend={"orientation": "h", "y": 1.08}, **PLOT_LAYOUT)
    st.plotly_chart(frontier_figure, width="stretch")
    if show_long_only_frontier:
        st.caption("The gap between the curves is the cost of Alpin Re's rules.")

# ---------------------------------------------------------------------------
# Tab 4 — catastrophe stress
# ---------------------------------------------------------------------------
period_end_days = np.array([3, 30, 90, 365], dtype=float)
expected_claims_by_period = expected_claims_12m * expected_claims_profile

with tab_stress:
    st.subheader("Can Alpin Re pay a major hurricane on time, after a market shock?")
    st.caption("Markets are shocked first (equities in %, bonds through duration × rate and spread moves). "
               "Claims then fall due at days 3, 30, 90 and 365. Cash pays first, then assets are sold, "
               "safest first. Premiums, retrocession and credit lines are left out (conservative).")
    with st.expander("Edit scenarios"):
        stress_scenarios = st.data_editor(pd.DataFrame(STRESS_SCENARIO_DEFAULTS), hide_index=True, width="stretch",
                                          key="scenarios_v4", num_rows="fixed")
    with st.expander("Edit liquidation order and costs"):
        st.caption("Haircut = forced-sale cost on top of the price shock. Policy sale limit = internal rule "
                   "(do not sell a whole sleeve at the bottom of the market). US ETFs settle T+1.")
        liquidity_assumptions = st.data_editor(
            pd.DataFrame([{
                "Ticker": t, "Class": ETF_INFO[t]["class"],
                "Tier (1 = first)": LIQUIDATION_DEFAULTS[ETF_INFO[t]["class"]]["tier"],
                "Settlement days": LIQUIDATION_DEFAULTS[ETF_INFO[t]["class"]]["settlement"],
                "Haircut (%)": LIQUIDATION_DEFAULTS[ETF_INFO[t]["class"]]["haircut"],
                "Policy sale limit (%)": LIQUIDATION_DEFAULTS[ETF_INFO[t]["class"]]["sale_limit"],
            } for t in available if t != "USD_CASH"]),
            hide_index=True, width="stretch", disabled=["Ticker", "Class"], key="liquidation_v4",
        )

if stress_scenarios["Scenario"].astype(str).str.strip().eq("").any() or stress_scenarios["Scenario"].duplicated().any():
    fail("Give each stress scenario a unique, non-empty name.")
numeric_scenarios = stress_scenarios.drop(columns="Scenario").to_numpy(dtype=float)
if not np.isfinite(numeric_scenarios).all():
    fail("Stress scenario inputs must be numbers.")

settlement_days = np.zeros(asset_count)
haircuts = np.zeros(asset_count)
sale_fractions = np.zeros(asset_count)
tiers = np.ones(asset_count)
for index, ticker in enumerate(available):
    if ticker == "USD_CASH":
        continue
    row = liquidity_assumptions.loc[liquidity_assumptions["Ticker"] == ticker].iloc[0]
    tiers[index] = max(1.0, float(row["Tier (1 = first)"]))
    settlement_days[index] = max(0.0, float(row["Settlement days"]))
    haircuts[index] = min(max(float(row["Haircut (%)"]) / 100, 0.0), 1.0)
    sale_fractions[index] = min(max(float(row["Policy sale limit (%)"]) / 100, 0.0), 1.0)


def run_scenario(portfolio, scenario, natcat_share=None):
    natcat_share = float(scenario["NatCat claims (% of invested assets)"]) / 100 if natcat_share is None else natcat_share
    cat_claims = capital * natcat_share
    cat_by_period = cat_claims * cat_claims_profile
    base_values = portfolio * capital
    stressed = base_values * (1.0 + shocked_returns(available, durations, scenario))
    scenario_haircuts = np.minimum(haircuts + float(scenario["Extra haircut (pp)"]) / 100, 1.0)
    scenario_sales = np.clip(sale_fractions * float(scenario["Sale-capacity multiplier (%)"]) / 100, 0.0, 1.0)
    scenario_settlement = settlement_days + max(int(scenario["Extra settlement days"]), 0)
    scenario_settlement[cash_index] = 0
    waterfall = catastrophe_liquidity_waterfall(stressed, cash_index, expected_claims_by_period + cat_by_period,
                                                period_end_days, scenario_settlement, scenario_haircuts,
                                                scenario_sales, tiers)
    asset_change = float(stressed.sum() - base_values.sum())
    liability_effect = liability_mod_duration * float(scenario["Rate shock (bp)"]) / 10_000 * reserves
    return {
        "waterfall": waterfall, "stressed": stressed, "cat_claims": cat_claims, "cat_by_period": cat_by_period,
        "asset_change": asset_change, "liability_effect": liability_effect,
        "surplus_after": surplus + asset_change + liability_effect - cat_claims - waterfall["haircut_loss"],
    }


def first_gap(outcome):
    return next((p["Horizon"] for p in outcome["waterfall"]["periods"] if p["Unpaid claims (USD)"] > 1e-6 * capital), None)


def breaking_point(portfolio, scenario):
    """Largest hurricane loss (share of assets) paid on time at every date (bisection)."""
    if first_gap(run_scenario(portfolio, scenario, 0.0)) is not None:
        return 0.0
    if first_gap(run_scenario(portfolio, scenario, 1.0)) is None:
        return None
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if first_gap(run_scenario(portfolio, scenario, middle)) is None else (low, middle)
    return low


scenario_rows, reverse_rows, surplus_rows, outcomes = [], [], [], {}
for _, scenario in stress_scenarios.iterrows():
    name = str(scenario["Scenario"])
    outcome = run_scenario(weights, scenario)
    outcomes[name] = outcome
    unpaid = [p["Unpaid claims (USD)"] for p in outcome["waterfall"]["periods"]]
    scenario_rows.append({"Scenario": name, "Hurricane claims (USD)": outcome["cat_claims"],
                          "Market change on assets (USD)": outcome["asset_change"],
                          "Largest unpaid amount (USD)": max(unpaid), "First gap": first_gap(outcome) or "None"})
    share = float(scenario["NatCat claims (% of invested assets)"]) / 100
    bp_new, bp_now = breaking_point(weights, scenario), breaking_point(current, scenario)
    reverse_rows.append({"Scenario": name, "Scenario loss (% of assets)": share * 100,
                         "Breaking point — recommended (% of assets)": np.nan if bp_new is None else bp_new * 100,
                         "Breaking point — current (% of assets)": np.nan if bp_now is None else bp_now * 100})
    surplus_rows.append({"Scenario": name, "Assets: market shock (USD)": outcome["asset_change"],
                         "Reserves: rate effect (USD)": outcome["liability_effect"],
                         "Hurricane loss (USD)": -outcome["cat_claims"],
                         "Forced-sale costs (USD)": -outcome["waterfall"]["haircut_loss"],
                         "Surplus after (USD)": outcome["surplus_after"],
                         "Surplus after / before (%)": outcome["surplus_after"] / surplus * 100})
scenario_table, reverse_table, surplus_table = map(pd.DataFrame, (scenario_rows, reverse_rows, surplus_rows))
worst_breaking_point = reverse_table["Breaking point — recommended (% of assets)"].min()

with tab_stress:
    paid = int((scenario_table["Largest unpaid amount (USD)"] <= 1e-6 * capital).sum())
    top = st.columns(3)
    top[0].metric("Scenarios paid on time", f"{paid} / {len(scenario_table)}")
    top[1].metric("Hurricane absorbable (worst scenario)",
                  "> 100% of assets" if np.isnan(worst_breaking_point) else f"{worst_breaking_point:.0f}% of assets")
    top[2].metric("Lowest surplus after stress", f"{surplus_table['Surplus after / before (%)'].min():.0f}% of today")
    st.markdown("#### 1 · Are claims paid on time?")
    st.dataframe(scenario_table.style.format({c: fmt_musd for c in scenario_table.columns if c.endswith("(USD)")}),
                 hide_index=True, width="stretch")
    st.markdown("#### 2 · How big a hurricane before the first payment fails?")
    st.dataframe(reverse_table.style.format({c: "{:.1f}" for c in reverse_table.columns if c != "Scenario"}, na_rep="> 100"),
                 hide_index=True, width="stretch")
    st.caption("The hurricane loss is increased, under each scenario's market conditions, until a claim "
               "cannot be paid on its due date.")
    st.markdown("#### 3 · What happens to the surplus?")
    st.dataframe(surplus_table.style.format({**{c: fmt_musd for c in surplus_table.columns if c.endswith("(USD)")},
                                             "Surplus after / before (%)": "{:.0f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Starting surplus {fmt_musd(surplus)}; each column is an effect on the surplus. When rates rise, "
               "the value of the reserves falls and offsets part of the bond losses.")
    selected = st.selectbox("Explore a scenario", list(outcomes))
    detail = pd.DataFrame(outcomes[selected]["waterfall"]["periods"])
    detail.insert(2, "Of which hurricane (USD)", outcomes[selected]["cat_by_period"])
    st.dataframe(detail.style.format({c: fmt_musd for c in detail.columns if c.endswith("(USD)")}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5 — additional analysis
# ---------------------------------------------------------------------------
with tab_more:
    st.subheader("Sensitivity analysis")
    questions = {
        "Risk aversion": "How much more return does Alpin Re get if it accepts more risk?",
        "Duration gap tolerance": "What does it cost to match the reserves more tightly?",
        "Turnover limit": "How much is lost by moving gradually instead of all at once?",
        "Black–Litterman prior tau": "Does the result depend on an arbitrary model parameter?",
    }
    mode = st.selectbox("Vary this input", list(questions))
    st.caption(questions[mode])
    if mode == "Risk aversion":
        values, label = sorted({risk_aversion * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)}), "Risk aversion"
    elif mode == "Duration gap tolerance":
        values, label = sorted({0.5, 0.75, 1.0, 1.5, 2.0, 3.0, duration_tolerance}), "Duration tolerance (± years)"
    elif mode == "Turnover limit":
        values, label = sorted({0.10, 0.20, 0.30, 0.40, 0.50, 1.0, max_turnover}), "Turnover limit"
    else:
        values, label = sorted({prior_tau * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)}), "Prior tau"
    rows, failures = [], []
    for value in values:
        case_lower, case_aversion, case_mu, case_mandate = lower, risk_aversion, mu, mandate
        if mode == "Risk aversion":
            case_aversion = value
        elif mode == "Duration gap tolerance":
            case_mandate = build_mandate(liquid_floor, tolerance=value)
        elif mode == "Turnover limit":
            case_mandate = build_mandate(liquid_floor, turnover=value)
        else:
            case_mu, _ = black_litterman_returns(cov, implied_excess, risk_free_rate, P, Q, view_taus, value)
        start = solve_linear(np.zeros(2 * asset_count), case_lower, upper, case_mandate)
        if not start.success:
            failures.append(f"{value:.2f}: limits infeasible")
            continue
        case = solve_surplus(case_mu, cov, hedge, case_lower, upper, case_mandate, start.x[:asset_count],
                             risk_aversion=case_aversion)
        if not case.success:
            failures.append(f"{value:.2f}: optimizer failed")
            continue
        stats = portfolio_stats(case.x, case_mu)
        rows.append({label: value, "Return on surplus (%)": stats["surplus_return"] * 100,
                     "Surplus volatility (%)": stats["surplus_vol"] * 100, "Surplus at risk, ES 99% (% of surplus)": stats["es"] * 100,
                     "Cash (%)": stats["cash"] * 100, "Equities (%)": stats["equity"] * 100,
                     "Credit (%)": stats["credit"] * 100, "Asset duration (y)": stats["asset_duration"]})
    if failures:
        st.warning("Cases without a solution: " + "; ".join(failures))
    if rows:
        table = pd.DataFrame(rows)
        figure = go.Figure(go.Scatter(x=table[label], y=table["Return on surplus (%)"], mode="lines+markers",
                                      line={"color": "#14213D", "width": 3}))
        figure.update_layout(title="Expected return on surplus", xaxis_title=label, yaxis_title="%", height=340,
                             margin={"t": 48, "b": 12, "l": 12, "r": 12}, **PLOT_LAYOUT)
        st.plotly_chart(figure, width="stretch")
        st.dataframe(table.style.format("{:.2f}"), hide_index=True, width="stretch")

    st.subheader("Risk measures of the surplus")
    reserve_daily = returns_usd.to_numpy() @ liability_mix
    risk_rows = []
    for name, portfolio in (("Recommended", weights), ("Current", current), ("Policy mix", policy), ("Lowest risk", lowest_risk.x)):
        daily_surplus = (returns_usd.to_numpy() @ portfolio - k * reserve_daily) * to_surplus
        var95, es95 = historical_var_es(daily_surplus, 0.95)
        var99, es99 = historical_var_es(daily_surplus, 0.99)
        risk_rows.append({"Portfolio": name, "1-day VaR 95% (% of surplus)": var95 * 100,
                          "1-day ES 99% (% of surplus)": es99 * 100,
                          "1-year ES 99% (% of surplus)": portfolio_stats(portfolio)["es"] * 100})
    risk_table = pd.DataFrame(risk_rows)
    st.dataframe(risk_table.style.format({c: "{:.2f}" for c in risk_table.columns if c != "Portfolio"}),
                 hide_index=True, width="stretch")
    st.caption("1-day: historical simulation. 1-year: normal approximation used in the risk-budget rule.")


# ---------------------------------------------------------------------------
# Tab 6 — assumptions and limits
# ---------------------------------------------------------------------------
with tab_assumptions:
    st.subheader("Assumptions")
    st.dataframe(pd.DataFrame([
        ("Data", f"{source_used}: {period} of daily adjusted ETF prices up to {data_as_of}. Cash earns the risk-free rate."),
        ("Reserves", f"Value moves with interest rates like a Treasury mix with the same duration "
                     f"({liability_mod_duration:.2f} years, from the payment pattern): "
                     f"{liability_mix[available.index('VGSH')]:.0%} short and {liability_mix[available.index('VGIT')]:.0%} "
                     "intermediate Treasuries."),
        ("Objective", "Maximize expected surplus return − λ/2 × surplus variance, with surplus return = "
                      "portfolio return − (reserves / assets) × reserve return."),
        ("Surplus at risk", "Shown, not constrained: 1-year expected shortfall at 99% of the surplus under a normal "
                            "approximation, −expected surplus return + 2.665 × surplus volatility."),
        ("Black–Litterman", "Prior = returns that make the policy mix optimal for the surplus objective; default "
                            "views = bond yields to maturity (Vanguard, 31 Oct 2026); covariance not adjusted by the views."),
        ("Liquidity rules", "Claims due in 12 months = year-1 payments of the reserves; cash + short-term Treasuries "
                            "(sold and settled in one day) must cover them."),
        ("Duration gap", "The average asset duration stays within ± the tolerance of the reserve duration: a "
                         "standard investment-policy limit, kept as a guardrail on top of the surplus optimization."),
        ("Liquidity stress", "Deterministic scenarios, T+1 settlement, forced-sale haircuts and policy sale limits."),
    ], columns=["Topic", "Assumption"]), hide_index=True, width="stretch")
    st.subheader("Limits")
    st.markdown(
        """
        - Reserves only react to interest rates: claims inflation and reserve uncertainty are not modelled.
        - The liquidity test ignores new premiums, retrocession and credit lines: results are conservative.
        - Risk is estimated on the chosen price history; another period gives other numbers.
        - Surplus at risk uses a normal approximation; real tails are fatter.
        - Regulatory and rating-agency capital models are not reproduced.
        """
    )

# ---------------------------------------------------------------------------
# Summary row (top of the page)
# ---------------------------------------------------------------------------
with kpi_area:
    st.caption(f"Data: {source_used} · prices to {data_as_of} · invested assets {fmt_musd(capital)} · surplus {fmt_musd(surplus)}")
    summary = st.columns(5)
    summary[0].metric("Expected return on surplus", f"{recommended_stats['surplus_return']:.1%}")
    summary[1].metric("Surplus volatility", f"{recommended_stats['surplus_vol']:.1%}")
    summary[2].metric("Asset duration", f"{recommended_stats['asset_duration']:.2f} y",
                      f"reserves {liability_mod_duration:.2f} y", delta_color="off",
                      help="Average duration of the assets vs the duration of the reserves. The rule keeps the gap "
                           f"within ± {duration_tolerance:.2f} year.")
    rate_hit = -(recommended_stats['asset_duration'] * capital - liability_mod_duration * reserves) / 100
    summary[3].metric("Surplus change if rates +1 pt", f"{rate_hit / 1e6:+,.0f}m",
                      f"{rate_hit / surplus:+.1%} of surplus", delta_color="off",
                      help="Assets lose value, the reserves owed fall too; the difference hits the surplus.")
    summary[4].metric("Rules respected", f"{sum(p for _, p, _ in checks)} / {len(checks)}")
