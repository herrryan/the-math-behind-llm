#!/usr/bin/env python3
"""Training and Evaluation Pipeline for Sell Put Deep Learning Models.

Workflow:
1. Simulates realistic historical market data with Merton Jump Diffusion.
2. Extracts temporal sequences and cross-sectional option surface features.
3. Splits into Chronological Train (70%) and Out-of-Sample Test (30%) sets.
4. Trains the model with AsymmetricTailLoss (penalizing tail risk breaches).
5. Runs event-driven backtesting on out-of-sample data across 4 strategies:
   - Naive (Unconditional Sell Put)
   - Heuristic (IV Rank >= 40%)
   - TCN Hybrid Filtered
   - Transformer Hybrid Filtered
"""

import argparse
import time
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from market_data import simulate_underlying_paths, generate_sell_put_dataset
from model import SellPutDecisionModel, AsymmetricTailLoss
from backtest import SellPutBacktester, format_backtest_report


def prepare_dataloaders(
    x_seq: np.ndarray,
    x_tab: np.ndarray,
    y: np.ndarray,
    split_ratio: float = 0.70,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, Tuple, Tuple]:
    """Chronological Train/Test split to avoid future lookahead bias."""
    n_samples = len(y)
    split_idx = int(n_samples * split_ratio)

    # Train Split
    x_seq_tr = torch.from_numpy(x_seq[:split_idx])
    x_tab_tr = torch.from_numpy(x_tab[:split_idx])
    y_tr = torch.from_numpy(y[:split_idx])

    # Out-of-Sample Test Split
    x_seq_te = torch.from_numpy(x_seq[split_idx:])
    x_tab_te = torch.from_numpy(x_tab[split_idx:])
    y_te = torch.from_numpy(y[split_idx:])

    train_ds = TensorDataset(x_seq_tr, x_tab_tr, y_tr)
    test_ds = TensorDataset(x_seq_te, x_tab_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (x_seq_tr, x_tab_tr, y_tr), (x_seq_te, x_tab_te, y_te)


def train_model(
    model: SellPutDecisionModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    penalty_ratio: float = 4.0,
    device: str = "cpu",
) -> SellPutDecisionModel:
    """Train model with Asymmetric Tail Loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = AsymmetricTailLoss(penalty_ratio=penalty_ratio)

    print(f">> Training on device: {device.upper()} | Epochs: {epochs} | Asymmetric Penalty: {penalty_ratio}x")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for b_seq, b_tab, b_y in train_loader:
            b_seq, b_tab, b_y = b_seq.to(device), b_tab.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_seq, b_tab)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            # Eval on test set
            model.eval()
            te_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for b_seq, b_tab, b_y in test_loader:
                    b_seq, b_tab, b_y = b_seq.to(device), b_tab.to(device), b_y.to(device)
                    logits = model(b_seq, b_tab)
                    te_loss += criterion(logits, b_y).item()
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    correct += (preds == b_y).sum().item()
                    total += len(b_y)

            avg_tr_loss = total_loss / max(1, batches)
            avg_te_loss = te_loss / max(1, len(test_loader))
            accuracy = correct / max(1, total) * 100.0
            print(f"Epoch {epoch:2d}/{epochs} - Train Loss: {avg_tr_loss:.4f} | Test Loss: {avg_te_loss:.4f} | Test Acc: {accuracy:.1f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Sell Put Quantitative Model")
    parser.add_argument("--days", type=int, default=1200, help="Simulated market trading days")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--penalty", type=float, default=4.0, help="Asymmetric false-positive penalty")
    args = parser.parse_args()

    # Hardware acceleration
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("=" * 70)
    print(" Sell Put Quantitative Options Model Pipeline")
    print("=" * 70)

    # 1. Market Simulation
    t0 = time.time()
    print(">> Simulating underlying price paths with Merton Jump Diffusion...")
    mkt = simulate_underlying_paths(n_days=args.days, seed=42)
    x_seq, x_tab, y, meta = generate_sell_put_dataset(mkt)
    print(f">> Dataset: {len(y)} trade decisions generated in {time.time() - t0:.2f}s")
    print(f">> Safe rate (y=1): {np.mean(y)*100:.1f}% | Crash rate (y=0): {(1.0 - np.mean(y))*100:.1f}%")

    # 2. Chronological Split
    split_ratio = 0.70
    split_idx = int(len(y) * split_ratio)
    train_loader, test_loader, _, test_tensors = prepare_dataloaders(
        x_seq, x_tab, y, split_ratio=split_ratio, batch_size=32
    )
    test_meta = meta[split_idx:]
    test_prices = mkt["prices"][split_idx:]

    # 3. Train Model 1: TCN Backbone
    print("\n" + "=" * 50)
    print(" Training Model A: Temporal Convolutional Network (TCN)")
    print("=" * 50)
    model_tcn = SellPutDecisionModel(
        seq_features=x_seq.shape[-1],
        tabular_features=x_tab.shape[-1],
        temporal_backbone="tcn",
        latent_dim=32,
    )
    model_tcn = train_model(
        model_tcn, train_loader, test_loader, epochs=args.epochs, lr=args.lr, penalty_ratio=args.penalty, device=device
    )

    # 4. Train Model 2: Transformer Backbone
    print("\n" + "=" * 50)
    print(" Training Model B: Causal Transformer Encoder")
    print("=" * 50)
    model_tf = SellPutDecisionModel(
        seq_features=x_seq.shape[-1],
        tabular_features=x_tab.shape[-1],
        temporal_backbone="transformer",
        latent_dim=32,
    )
    model_tf = train_model(
        model_tf, train_loader, test_loader, epochs=args.epochs, lr=args.lr, penalty_ratio=args.penalty, device=device
    )

    # 5. Out-of-Sample Predictions
    x_seq_te, x_tab_te, _ = test_tensors
    x_seq_te, x_tab_te = x_seq_te.to(device), x_tab_te.to(device)

    preds_tcn = model_tcn.predict_probability(x_seq_te, x_tab_te).cpu().numpy()
    preds_tf = model_tf.predict_probability(x_seq_te, x_tab_te).cpu().numpy()

    # 6. Event-Driven Backtesting on Out-of-Sample Test Period
    print("\n" + "=" * 70)
    print(f" OUT-OF-SAMPLE BACKTEST COMPARISON ({len(test_meta)} Trading Days)")
    print("=" * 70)

    tester = SellPutBacktester(
        initial_capital=100_000.0,
        margin_pct=0.20,
        allocation_per_trade=0.15,
        stop_loss_mult=2.0,
    )

    naive_res = tester.run_backtest(test_meta, test_prices, strategy_name="Naive (Always)")
    heur_res = tester.run_backtest(test_meta, test_prices, strategy_name="Heuristic (IVR)")
    tcn_res = tester.run_backtest(test_meta, test_prices, predictions=preds_tcn, strategy_name="TCN Filtered", conf_threshold=0.55)
    tf_res = tester.run_backtest(test_meta, test_prices, predictions=preds_tf, strategy_name="Transformer Filter", conf_threshold=0.55)

    print(format_backtest_report([naive_res, heur_res, tcn_res, tf_res]))

    # Save weights
    torch.save(model_tcn.state_dict(), "options-sell-put/checkpoint_tcn.pt")
    torch.save(model_tf.state_dict(), "options-sell-put/checkpoint_tf.pt")
    print("\n>> Trained model weights saved to options-sell-put/checkpoint_tcn.pt & checkpoint_tf.pt")


if __name__ == "__main__":
    main()
