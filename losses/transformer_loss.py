"""
Transformer-context branch losses.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_trade_reconstruction_loss(
    enc_out_recon: torch.Tensor,
    x_original: torch.Tensor,
    mask: torch.Tensor | None,
    trade_idx: list[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is None or mask.sum() == 0:
        return torch.tensor(0.0, device=x_original.device)

    x_trade_original = x_original[:, :, trade_idx] + eps
    x_trade_recon = enc_out_recon[:, :, trade_idx]
    loss = F.mse_loss(x_trade_recon[mask], x_trade_original[mask])

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=x_original.device)
    return loss


def causal_price_decoder_loss(
    pred_price: torch.Tensor,
    target_price: torch.Tensor,
    price_weights: list[float] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred_price.shape != target_price.shape:
        raise ValueError(
            f"pred_price shape {pred_price.shape} does not match target_price shape {target_price.shape}"
        )

    _, _, f_price = pred_price.shape
    if price_weights is None:
        price_weights = [1.0] * f_price
    if len(price_weights) != f_price:
        raise ValueError(
            f"price_weights length {len(price_weights)} must equal price dimension {f_price}"
        )

    weights = torch.tensor(price_weights, device=pred_price.device, dtype=pred_price.dtype)
    mse_per_dim = F.mse_loss(pred_price + eps, target_price + eps, reduction="none")
    weighted_mse = mse_per_dim * weights.unsqueeze(0).unsqueeze(0)
    loss = weighted_mse.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred_price.device, dtype=pred_price.dtype)
    return loss


def compute_transformer_context_losses(
    encoder,
    decoder,
    x: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc_out, enc_out_recon, mask = encoder(x, mask_trade_ratio=cfg.MASK_RATIO)

    x_price = x[:, :, cfg.PRICE_IDX]
    x_trade = x[:, :, cfg.TRADE_IDX]
    dec_out = decoder(x_price, x_trade, memory=enc_out)

    enc_loss = masked_trade_reconstruction_loss(
        enc_out_recon=enc_out_recon,
        x_original=x,
        mask=mask,
        trade_idx=cfg.TRADE_IDX,
    )
    dec_loss = causal_price_decoder_loss(
        pred_price=dec_out,
        target_price=x_price,
    )
    fingerprint = enc_out[:, -1, :]
    return enc_loss, dec_loss, fingerprint
