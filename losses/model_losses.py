"""
Unified task-loss entrypoints for different training model branches.
"""
from __future__ import annotations

import torch

from losses.autoencoder_loss import compute_autoencoder_losses
from losses.transformer_loss import compute_transformer_context_losses


def compute_model_losses(
    model_type: str,
    encoder,
    decoder,
    x: torch.Tensor,
    cfg: Any,
    reg_loss_fn=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if model_type == "transformer_context":
        enc_loss, dec_loss, fingerprint = compute_transformer_context_losses(
            encoder=encoder,
            decoder=decoder,
            x=x,
            cfg=cfg,
        )
    elif model_type == "autoencoder":
        enc_loss, dec_loss, fingerprint = compute_autoencoder_losses(
            encoder=encoder,
            decoder=decoder,
            x=x,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

    reg_metrics: dict[str, float] = {}
    if getattr(cfg, "USE_REG_LOSS", False) and reg_loss_fn is not None:
        total_loss, reg_metrics = reg_loss_fn.total_loss(
            emb=fingerprint,
            loss_forward=dec_loss,
            loss_backward=enc_loss,
        )
    else:
        total_loss = dec_loss + enc_loss

    metrics = {
        "enc_loss": float(enc_loss.item()),
        "dec_loss": float(dec_loss.item()),
        "total_loss": float(total_loss.item()),
        "loss_diversity": float(reg_metrics.get("loss_diversity", 0.0)),
        "loss_orthogonality": float(reg_metrics.get("loss_orthogonality", 0.0)),
        "loss_uniformity": float(reg_metrics.get("loss_uniformity", 0.0)),
    }
    return total_loss, metrics
