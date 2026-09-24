#!/usr/bin/env python3
"""Deep Learning and Hybrid Architectures for Sell Put Decision Modeling.

Includes:
1. CausalConv1d & Temporal Convolutional Network (TCN).
2. Lightweight Causal Transformer Encoder (for comparison).
3. Hybrid Fusion Classifier (Temporal Sequence + Option Greeks/Tabular).
4. Asymmetric Tail Loss (heavily penalizing false positives where tail risk breaches).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Causal Convolutional Blocks (TCN)
# =====================================================================

class CausalConv1d(nn.Module):
    """Causal 1D Convolution with dilation, preventing future lookahead bias."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Slice off right padding to enforce strict causality
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNResidualBlock(nn.Module):
    """Residual block with two dilated causal convolutions, layer norms, and dropout."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return F.gelu(h + res)


class TemporalTCNEncoder(nn.Module):
    """Encodes historical time series into a compact latent regime vector."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        # Dilations [1, 2, 4, 8] cover receptive field of > 60 days
        self.block1 = TCNResidualBlock(hidden_dim, kernel_size=3, dilation=1)
        self.block2 = TCNResidualBlock(hidden_dim, kernel_size=3, dilation=2)
        self.block3 = TCNResidualBlock(hidden_dim, kernel_size=3, dilation=4)
        self.block4 = TCNResidualBlock(hidden_dim, kernel_size=3, dilation=8)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [Batch, SeqLen, InFeatures]
        Returns: [Batch, OutDim]
        """
        # Transpose to [Batch, InFeatures, SeqLen]
        h = x_seq.transpose(1, 2)
        h = self.input_proj(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        # Take the final temporal step (current time t)
        last_step = h[:, :, -1]
        return self.out_proj(last_step)


# =====================================================================
# 2. Lightweight Causal Transformer Encoder (Alternative)
# =====================================================================

class TemporalTransformerEncoder(nn.Module):
    """Causal Transformer Encoder for time series feature representation."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 32, n_heads: int = 4, n_layers: int = 2, out_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_seq.shape
        # Create upper-triangular causal attention mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x_seq.device), diagonal=1)

        h = self.input_proj(x_seq)
        h = self.transformer(h, mask=causal_mask)
        # Return representation of the final time step
        return self.out_proj(h[:, -1, :])


# =====================================================================
# 3. Hybrid Decision Model (Temporal + Option Tabular)
# =====================================================================

class SellPutDecisionModel(nn.Module):
    """Full decision model combining temporal dynamics with option surface features."""

    def __init__(
        self,
        seq_features: int = 4,
        tabular_features: int = 8,
        temporal_backbone: str = "tcn",
        latent_dim: int = 32,
    ):
        super().__init__()
        self.temporal_backbone_name = temporal_backbone

        if temporal_backbone == "tcn":
            self.temporal_encoder = TemporalTCNEncoder(
                in_features=seq_features, hidden_dim=32, out_dim=latent_dim
            )
        elif temporal_backbone == "transformer":
            self.temporal_encoder = TemporalTransformerEncoder(
                in_features=seq_features, hidden_dim=32, out_dim=latent_dim
            )
        else:
            raise ValueError(f"Unknown temporal backbone: {temporal_backbone}")

        # Combine temporal latent (32) + option surface tabular features (8)
        combined_dim = latent_dim + tabular_features

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [Batch, SeqLen, SeqFeatures]
        x_tab: [Batch, TabularFeatures]
        Returns: Logit score [Batch, 1]
        """
        latent_seq = self.temporal_encoder(x_seq)
        combined = torch.cat([latent_seq, x_tab], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def predict_probability(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """Return sigmoid open-position confidence in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x_seq, x_tab)
            return torch.sigmoid(logits)


# =====================================================================
# 4. Asymmetric Tail-Risk Loss
# =====================================================================

class AsymmetricTailLoss(nn.Module):
    """Loss function penalizing false positives where underlying crashes.

    In Sell Put trading:
    - False Negative (predicted 0, actually safe): We just missed a safe premium (small regret).
    - False Positive (predicted 1, actually breached): We sold a put into a crash (disastrous loss).
    penalty_ratio > 1.0 enforces strong risk aversion against tail events.
    """

    def __init__(self, penalty_ratio: float = 4.0):
        super().__init__()
        self.penalty_ratio = penalty_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        eps = 1e-7
        probs = torch.clamp(probs, eps, 1.0 - eps)

        # Standard binary cross entropy with asymmetric weighting on the negative class
        loss_pos = targets * torch.log(probs)
        loss_neg = self.penalty_ratio * (1.0 - targets) * torch.log(1.0 - probs)

        loss = -(loss_pos + loss_neg)
        return loss.mean()


if __name__ == "__main__":
    print("Testing SellPutDecisionModel architectures...")
    batch_size = 16
    seq_len = 60
    n_seq_feat = 4
    n_tab_feat = 8

    dummy_seq = torch.randn(batch_size, seq_len, n_seq_feat)
    dummy_tab = torch.randn(batch_size, n_tab_feat)
    dummy_target = torch.randint(0, 2, (batch_size,)).float()

    # Test TCN backbone
    model_tcn = SellPutDecisionModel(seq_features=n_seq_feat, tabular_features=n_tab_feat, temporal_backbone="tcn")
    out_tcn = model_tcn(dummy_seq, dummy_tab)
    print(f">> TCN backbone output shape: {out_tcn.shape}")

    # Test Transformer backbone
    model_tf = SellPutDecisionModel(seq_features=n_seq_feat, tabular_features=n_tab_feat, temporal_backbone="transformer")
    out_tf = model_tf(dummy_seq, dummy_tab)
    print(f">> Transformer backbone output shape: {out_tf.shape}")

    # Test Asymmetric Loss
    criterion = AsymmetricTailLoss(penalty_ratio=4.0)
    loss = criterion(out_tcn, dummy_target)
    print(f">> Asymmetric loss calculated: {loss.item():.4f}")
    print("Model architectures verified successfully.")
