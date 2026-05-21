"""
Autoencoder branch losses.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_autoencoder_losses(
    encoder,
    decoder,
    x: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = x.new_zeros(())

    z_day, _ = encoder(x)
    x_recon = decoder(z_day, seq_len=x.size(1))

    price_loss = (
        F.mse_loss(x_recon[:, :, cfg.PRICE_IDX], x[:, :, cfg.PRICE_IDX])
        if cfg.PRICE_IDX else zero
    )
    trade_loss = (
        F.mse_loss(x_recon[:, :, cfg.TRADE_IDX], x[:, :, cfg.TRADE_IDX])
        if cfg.TRADE_IDX else zero
    )
    fingerprint = z_day
    return trade_loss, price_loss, fingerprint
