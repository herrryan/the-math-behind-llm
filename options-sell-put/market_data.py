#!/usr/bin/env python3
"""Market and Option Surface Simulator for Sell Put Strategy.

Simulates realistic underlying asset dynamics with:
1. Geometric Brownian Motion + Merton Jump-Diffusion (captures fat-tail market crashes).
2. Volatility clustering and stochastic volatility (Heston-like regimes).
3. Option chain pricing via Black-Scholes with implied volatility smile/skew.
4. Volatility Risk Premium (VRP = IV - RV) and IV Rank calculation.
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_put(
    s: float, k: float, t: float, r: float, sigma: float
) -> Tuple[float, float, float, float]:
    """Calculate Black-Scholes Put price and Greeks.

    Returns:
        (put_price, delta, theta_per_day, vega_per_pct)
    """
    if t <= 1e-5:
        intrinsic = max(0.0, k - s)
        return intrinsic, (-1.0 if s < k else 0.0), 0.0, 0.0

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    put_price = k * math.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1.0  # Put delta is in [-1, 0]

    # Theta annual -> per calendar day (divide by 365)
    theta_annual = -(s * norm_pdf(d1) * sigma) / (2.0 * sqrt_t) + r * k * math.exp(-r * t) * norm_cdf(-d2)
    theta_day = theta_annual / 365.0

    # Vega per 1% move in IV
    vega_pct = (s * sqrt_t * norm_pdf(d1)) / 100.0

    return max(0.01, put_price), delta, theta_day, vega_pct


def simulate_underlying_paths(
    n_days: int = 1200,
    s0: float = 100.0,
    mu: float = 0.08,
    base_sigma: float = 0.18,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulate daily OHLCV paths with Merton Jump Diffusion & Regime Shifts."""
    np.random.seed(seed)
    dt = 1.0 / 252.0

    prices = np.zeros(n_days)
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    rv_30d = np.zeros(n_days)
    regimes = np.zeros(n_days, dtype=int)  # 0: Low Vol Bull, 1: High Vol Crash

    curr_s = s0
    curr_sigma = base_sigma
    regime = 0

    # Jump diffusion parameters
    jump_intensity = 0.04  # 4% chance of jump per month
    jump_mean = -0.06      # Crashes are negative jumps on average
    jump_std = 0.04

    for t in range(n_days):
        # Regime transition probability
        if regime == 0 and np.random.rand() < 0.01:
            regime = 1  # Enter crisis
        elif regime == 1 and np.random.rand() < 0.05:
            regime = 0  # Recovery

        regimes[t] = regime
        vol_factor = 2.2 if regime == 1 else 1.0
        eff_sigma = curr_sigma * vol_factor

        # Brownian increment
        z = np.random.standard_normal()
        drift = (mu - 0.5 * eff_sigma * eff_sigma) * dt
        diffusion = eff_sigma * math.sqrt(dt) * z

        # Jump component
        jump = 0.0
        if np.random.rand() < (jump_intensity * dt):
            jump = np.random.normal(jump_mean, jump_std)

        log_return = drift + diffusion + jump
        next_s = curr_s * math.exp(log_return)

        # Generate realistic intraday High and Low
        intra_vol = eff_sigma * math.sqrt(dt)
        highs[t] = max(curr_s, next_s) * math.exp(abs(np.random.normal(0, intra_vol * 0.7)))
        lows[t] = min(curr_s, next_s) * math.exp(-abs(np.random.normal(0, intra_vol * 0.7)))
        prices[t] = next_s
        curr_s = next_s

    # Compute 30-day rolling Realized Volatility (annualized)
    log_rets = np.zeros(n_days)
    log_rets[1:] = np.log(prices[1:] / prices[:-1])
    for t in range(29, n_days):
        rv_30d[t] = np.std(log_rets[t - 29 : t + 1]) * math.sqrt(252)
    rv_30d[:29] = base_sigma

    return {
        "prices": prices,
        "highs": highs,
        "lows": lows,
        "log_returns": log_rets,
        "rv_30d": rv_30d,
        "regimes": regimes,
    }


def generate_sell_put_dataset(
    market_data: Dict[str, np.ndarray],
    target_dte: int = 35,
    target_delta: float = -0.20,
    seq_len: int = 60,
    r: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Generate training dataset with Temporal Sequences, Option Surface Tabulars, and Labels.

    Target Definition:
        y = 1: Safe Sell Put (Option expires OTM or max drawdown <= 1.5x premium)
        y = 0: Severe Tail Risk Breach (Assigned at loss or drawdown > 1.5x premium)
    """
    prices = market_data["prices"]
    rvs = market_data["rv_30d"]
    log_rets = market_data["log_returns"]
    n_days = len(prices)

    temporal_samples = []
    tabular_samples = []
    labels = []
    trade_meta = []

    # Historical IV series for IV Rank calculation
    iv_history = []

    for t in range(seq_len, n_days - target_dte - 1):
        s_t = prices[t]
        rv_t = rvs[t]

        # Implied Volatility dynamics: VRP premium + Skew
        # When RV is high, IV expands even more (volatility risk premium)
        vrp_premium = 0.03 + 0.02 * np.random.rand()
        iv_atm = max(0.10, rv_t + vrp_premium)
        iv_history.append(iv_atm)

        # 1-year IV Rank (IVR): (IV - Min_IV) / (Max_IV - Min_IV)
        lookback_iv = iv_history[-252:] if len(iv_history) >= 252 else iv_history
        min_iv, max_iv = min(lookback_iv), max(lookback_iv)
        iv_rank = (iv_atm - min_iv) / (max_iv - min_iv + 1e-6)

        # Put Skew (25D Put IV is higher than ATM IV due to crash fear)
        skew_spread = 0.04 + 0.05 * (1.0 if market_data["regimes"][t] == 1 else 0.0)
        iv_put = iv_atm + skew_spread

        # Find Strike K matching target delta (-0.20 delta put)
        t_years = target_dte / 365.0
        # Analytic approx for delta strike: K = S * exp(d1 * sigma * sqrt(T))
        # For delta = -0.20, norm_cdf(d1) = 0.80 -> d1 approx 0.84
        d1_target = 0.8416
        k_exact = s_t * math.exp(-d1_target * iv_put * math.sqrt(t_years) + (r + 0.5 * iv_put**2) * t_years)
        k = round(k_exact, 1)

        premium, delta, theta, vega = black_scholes_put(s_t, k, t_years, r, iv_put)
        strike_safety_buffer = (s_t - k) / s_t  # percentage OTM

        # Track outcome over holding period [t, t + target_dte]
        forward_low = np.min(market_data["lows"][t + 1 : t + target_dte + 1])
        forward_end = prices[t + target_dte]

        # Maximum unrealized loss during trade
        max_intrinsic_drawdown = max(0.0, k - forward_low)
        end_loss = max(0.0, k - forward_end)

        # Label criteria:
        # Success (y = 1): Kept premium, max loss during tenure <= 1.5x premium
        # Failure (y = 0): Deep drawdown breach > 1.5x premium (tail risk)
        is_safe = 1 if (end_loss == 0.0 and max_intrinsic_drawdown <= 1.5 * premium) else 0

        # Construct 60-day Temporal Sequence:
        # Features: [Normalized Return, Normalized RV, EMA Divergence, Normalized High-Low Range]
        sub_rets = log_rets[t - seq_len : t]
        sub_rvs = rvs[t - seq_len : t]
        sub_prices = prices[t - seq_len : t]
        ema_20 = np.mean(sub_prices[-20:]) / sub_prices[-1] - 1.0
        ema_60 = np.mean(sub_prices) / sub_prices[-1] - 1.0

        seq_feat = np.column_stack([
            sub_rets,
            sub_rvs,
            (market_data["highs"][t - seq_len : t] - market_data["lows"][t - seq_len : t]) / sub_prices,
            (sub_prices - np.mean(sub_prices)) / (np.std(sub_prices) + 1e-6),
        ])  # Shape: [seq_len, 4]

        # Construct Option Tabular Vector (8 features):
        tabular_feat = np.array([
            iv_atm,
            iv_rank,
            iv_put - iv_atm,          # Skew
            iv_atm - rv_t,             # Volatility Risk Premium (VRP)
            strike_safety_buffer,      # Buffer (S - K) / S
            abs(delta),                # Put Delta
            premium / s_t,             # Premium yield
            theta / premium,           # Daily theta decay yield
        ], dtype=np.float32)

        temporal_samples.append(seq_feat)
        tabular_samples.append(tabular_feat)
        labels.append(is_safe)
        trade_meta.append({
            "day": t,
            "s_t": s_t,
            "k": k,
            "premium": premium,
            "delta": delta,
            "iv_rank": iv_rank,
            "forward_end": forward_end,
            "is_safe": is_safe,
            "pnl": premium - end_loss,
        })

    return (
        np.array(temporal_samples, dtype=np.float32),
        np.array(tabular_samples, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        trade_meta,
    )


if __name__ == "__main__":
    print("Simulating market environment with Merton Jump Diffusion...")
    mkt = simulate_underlying_paths(n_days=1000)
    print(f"Generated {len(mkt['prices'])} trading days.")
    print(f"Price range: [{mkt['prices'].min():.2f}, {mkt['prices'].max():.2f}]")
    print(f"Crisis regime days count: {np.sum(mkt['regimes'] == 1)} days.")

    x_seq, x_tab, y, meta = generate_sell_put_dataset(mkt)
    print(f"Dataset generated:")
    print(f" - Temporal sequence shape: {x_seq.shape}")
    print(f" - Option tabular shape:    {x_tab.shape}")
    print(f" - Labels shape:            {y.shape} (Positive rate: {np.mean(y)*100:.1f}%)")
