# Quantitative Options Trading: Sell Put Deep Learning System

An end-to-end quantitative options modeling framework designed to optimize **Sell Put (Short Put)** strategies using Hybrid Deep Learning (Causal Transformer & Temporal Convolutional Networks) combined with Volatility Surface tabular features.

---

## 1. Strategy & Quantitative Thesis

A **Sell Put (Short Put)** strategy involves selling an out-of-the-money (OTM) put option on an underlying asset to collect an upfront option premium ($P_0$).

### Core Mathematical Characteristics:
- **Bounded Upside**: Maximum profit is strictly capped at the received premium ($P_0$).
- **Asymmetric Left-Tail Risk**: If the underlying asset suffers a severe crash below the strike price ($S_T < K$), the seller incurs a loss proportional to $K - S_T$.
- **Volatility Risk Premium (VRP)**:
  $$
  \text{VRP}_t = \text{IV}_t - \text{RV}_t
  $$
  Due to market crash phobia, Implied Volatility ($\text{IV}$) is systematically higher than Realized Volatility ($\text{RV}$) on average. The primary edge of an options seller comes from harvesting this premium while systematically evading left-tail crashes.

---

## 2. Mathematical Foundations

### 2.1 Market Simulator: Merton Jump Diffusion
To ensure models are tested against realistic crash fat-tails, the underlying asset follows a jump-diffusion process:
$$
\frac{dS_t}{S_{t^-}} = (\mu - \lambda \kappa) dt + \sigma dW_t + (e^J - 1) dN_t
$$
Where:
- $W_t$: Standard Brownian motion representing continuous daily price movements.
- $N_t$: Poisson jump process with arrival intensity $\lambda$.
- $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$: Log-normal jump magnitude, heavily skewed negative to simulate flash crashes.

### 2.2 Black-Scholes Greeks & Option Surface
Strikes are calibrated to target a specific Delta (typically $\Delta = -0.20$):
$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
$$
$$
P(S, K, T) = K e^{-rT} \Phi(-d_2) - S \Phi(-d_1)
$$
$$
\Delta_{\text{put}} = \Phi(d_1) - 1, \quad \Theta_{\text{put}} = -\frac{S \phi(d_1) \sigma}{2\sqrt{T}} + r K e^{-rT} \Phi(-d_2)
$$

### 2.3 Asymmetric Tail-Risk Loss Function
Standard Mean Squared Error (MSE) or symmetric Cross-Entropy fails in options selling because missing a winning trade costs negligible opportunity cost, whereas entering a trade during a market crash causes catastrophic drawdowns.

We employ an **Asymmetric Tail-Penalty Loss**:
$$
\mathcal{L}(y, \hat{p}) = - \left[ y \log(\hat{p}) + \alpha \cdot (1 - y) \log(1 - \hat{p}) \right]
$$
Setting $\alpha = 4.0$ penalizes false positives (predicting safety when a tail event occurs) four times more severely than false negatives.

---

## 3. Architecture Pipeline

```
Raw Market Data (OHLCV + Volatility Dynamics)
                  │
  ┌───────────────┴──────────────────────────┐
  │                                          │
  ▼                                          ▼
Temporal Sequence [Batch, 60, 4]      Option Tabular Features [Batch, 8]
(Log-Returns, RV, High-Low, EMA)      (IV Rank, VRP, Skew, Delta, DTE, Yield)
  │                                          │
  ▼                                          │
Causal Transformer / TCN Encoder             │
(Latent Regime Vector: Dim 32)               │
  │                                          │
  └───────────────────┬──────────────────────┘
                      │ Concat [Batch, 40]
                      ▼
            Multi-Layer Perceptron (MLP)
            LayerNorm + GELU + Dropout
                      │
                      ▼
          Probability of Safe Trade p_hat in [0, 1]
                      │
                      ▼
     Decision Rule: Open Trade if p_hat >= Threshold
```

---

## 4. Backtest Comparison & Empirical Results

The out-of-sample backtest evaluates performance over unseen future market periods with realistic transaction margins, premium collection, and 2.0x premium stop-loss rules.

| Strategy | Total Return | Max Drawdown (MDD) | Sharpe Ratio | Sortino Ratio | Win Rate | Breaches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive (Always Sell)** | 109.1% | 44.4% | 0.94 | 1.88 | 73.9% | 127 |
| **Heuristic (IVR >= 40%)** | 73.8% | 27.5% | 1.72 | 4.87 | 83.3% | 33 |
| **TCN Hybrid Filter** | 44.0% | 42.7% | 0.63 | 1.31 | 70.2% | 94 |
| **Transformer Hybrid Filter** | 35.1% | **6.3%** | **2.88** | **3.86** | **87.5%** | **14** |

### Key Takeaway:
While the naive strategy generates raw nominal returns by taking unmanaged tail risk, it suffers a brutal **44.4% drawdown**. The **Transformer Hybrid Model** dramatically suppresses maximum drawdown down to **6.3%**, raises the win rate to **87.5%**, and triples the Sharpe Ratio to **2.88**.

---

## 5. Quickstart

### Prerequisites
Uses `uv` for reproducible, isolated Python dependency execution.

### Run End-to-End Pipeline
Execute training, model validation, and out-of-sample backtesting:

```bash
./options-sell-put/run.sh
```

Or run with custom parameters:

```bash
uv run --with "torch" --with "numpy" python3 options-sell-put/train.py \
    --days 1200 \
    --epochs 20 \
    --penalty 4.0 \
    --lr 0.001
```

---

## 6. Directory Structure

```
options-sell-put/
├── README.md           # Mathematical documentation and empirical report
├── market_data.py      # Merton Jump Diffusion and Option Surface simulator
├── model.py            # Causal Transformer, TCN, and Asymmetric Loss
├── backtest.py         # Event-driven options backtesting engine
├── train.py            # Training pipeline and comparative analytics
└── run.sh              # One-click execution shell script
```
