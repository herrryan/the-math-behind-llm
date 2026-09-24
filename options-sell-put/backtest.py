#!/usr/bin/env python3
"""Event-driven Backtesting Engine for Sell Put Strategies.

Compares:
1. Benchmark: Underlying Buy & Hold.
2. Naive Sell Put: Unconditional selling (every cycle).
3. Rule-Based Heuristic: Filtered by IV Rank >= 40 + Stop Loss.
4. ML / Deep Learning Filtered: Sell only when model confidence >= threshold.

Calculates key risk-adjusted metrics:
- Total Return (%)
- Annualized Return (%)
- Maximum Drawdown (MDD %)
- Sharpe Ratio
- Sortino Ratio (downside risk only)
- Win Rate (%)
- Number of Crash / Tail-Loss Breaches
"""

import math
from typing import Dict, List, Optional
import numpy as np


class SellPutBacktester:
    """Simulates realistic options trading with margin, premium collection, and risk management."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        margin_pct: float = 0.20,      # 20% margin requirement
        allocation_per_trade: float = 0.15,  # Risk at most 15% capital per trade
        stop_loss_mult: float = 2.0,    # Cut loss if put price reaches 2x initial premium
        profit_target_pct: float = 0.50, # Take profit if 50% of premium captured
    ):
        self.initial_capital = initial_capital
        self.margin_pct = margin_pct
        self.allocation_per_trade = allocation_per_trade
        self.stop_loss_mult = stop_loss_mult
        self.profit_target_pct = profit_target_pct

    def run_backtest(
        self,
        trade_meta: List[Dict],
        prices: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        strategy_name: str = "ML-Filtered",
        conf_threshold: float = 0.60,
    ) -> Dict:
        """Run backtest on trade metadata."""
        capital = self.initial_capital
        equity_curve = [capital]
        trades_executed = 0
        winning_trades = 0
        tail_breaches = 0
        total_pnl = 0.0

        daily_returns = []

        # Iterate through available trade setup days
        for i, trade in enumerate(trade_meta):
            s_t = trade["s_t"]
            k = trade["k"]
            premium = trade["premium"]
            raw_pnl = trade["pnl"]
            is_safe = trade["is_safe"]
            iv_rank = trade["iv_rank"]

            # Filter decisions based on strategy
            should_trade = False
            if "naive" in strategy_name.lower():
                should_trade = True
            elif "heuristic" in strategy_name.lower():
                # Rule of thumb: only sell when IV Rank is elevated (>= 40%)
                should_trade = (iv_rank >= 0.40)
            elif predictions is not None:
                # Sell only when model is confident safety is >= threshold
                should_trade = (predictions[i] >= conf_threshold)

            if not should_trade:
                equity_curve.append(capital)
                continue

            # Position sizing (number of contracts)
            # Cash required = K * 100 * margin_pct
            margin_per_contract = k * 100 * self.margin_pct
            max_risk_dollars = capital * self.allocation_per_trade
            n_contracts = max(1, int(max_risk_dollars / (margin_per_contract + 1e-6)))

            # Trade execution PnL calculation
            # Account for stop loss
            trade_profit_per_share = raw_pnl
            if trade_profit_per_share < -self.stop_loss_mult * premium:
                # Stop loss triggered: capped at multiplier * premium
                trade_profit_per_share = -self.stop_loss_mult * premium

            trade_dollar_pnl = trade_profit_per_share * 100 * n_contracts
            capital += trade_dollar_pnl

            # Prevent total bankruptcy in simulations
            capital = max(1000.0, capital)

            trades_executed += 1
            if trade_dollar_pnl > 0:
                winning_trades += 1
            if not is_safe:
                tail_breaches += 1

            ret = trade_dollar_pnl / equity_curve[-1]
            daily_returns.append(ret)
            equity_curve.append(capital)

        # Performance analytics
        equity_arr = np.array(equity_curve)
        total_return_pct = (capital - self.initial_capital) / self.initial_capital * 100.0

        # Max Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_drawdown_pct = abs(np.min(drawdowns)) * 100.0

        # Sharpe & Sortino (assuming ~12 trade cycles per year)
        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-7:
            annual_factor = math.sqrt(12)  # Monthly cycle
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = (mean_ret / std_ret) * annual_factor

            # Downside deviation for Sortino
            downside = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-6
            sortino = (mean_ret / (downside_std + 1e-6)) * annual_factor
        else:
            sharpe = 0.0
            sortino = 0.0

        win_rate = (winning_trades / trades_executed * 100.0) if trades_executed > 0 else 0.0

        return {
            "strategy": strategy_name,
            "final_capital": capital,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate_pct": win_rate,
            "trades_executed": trades_executed,
            "tail_breaches": tail_breaches,
        }


def format_backtest_report(results: List[Dict]) -> str:
    """Generate clean comparison table."""
    header = (
        f"{'Strategy':<16} | {'Return':>8} | {'MDD':>7} | {'Sharpe':>7} | "
        f"{'Sortino':>7} | {'Win Rate':>9} | {'Trades':>6} | {'Breaches':>8}"
    )
    divider = "-" * len(header)
    lines = [divider, header, divider]

    for r in results:
        lines.append(
            f"{r['strategy']:<16} | {r['total_return_pct']:>7.1f}% | {r['max_drawdown_pct']:>6.1f}% | "
            f"{r['sharpe_ratio']:>7.2f} | {r['sortino_ratio']:>7.2f} | {r['win_rate_pct']:>8.1f}% | "
            f"{r['trades_executed']:>6} | {r['tail_breaches']:>8}"
        )
    lines.append(divider)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing SellPutBacktester...")
    from market_data import simulate_underlying_paths, generate_sell_put_dataset

    mkt = simulate_underlying_paths(n_days=800)
    _, _, _, meta = generate_sell_put_dataset(mkt)

    tester = SellPutBacktester()
    naive_res = tester.run_backtest(meta, mkt["prices"], strategy_name="Naive")
    heur_res = tester.run_backtest(meta, mkt["prices"], strategy_name="Heuristic")

    print(format_backtest_report([naive_res, heur_res]))
