"""
models/autoencoder.py - day-level autoencoder for compression-style fingerprints.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding


class AutoEncoderEncoder(nn.Module):
    """
    Encode an intraday sequence into a single day-level bottleneck vector.
    """

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        latent_dim: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(f_in, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = self.pos_enc(x_proj)
        enc_out = self.transformer(x_proj)
        pooled = enc_out.mean(dim=1)
        z_day = self.to_latent(pooled)
        return z_day, enc_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_day, _ = self.forward(x)
        return z_day


class AutoEncoderDecoder(nn.Module):
    """
    Reconstruct the full intraday sequence from a single day-level bottleneck.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        f_out: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, f_out)

    def forward(self, z_day: torch.Tensor, seq_len: int) -> torch.Tensor:
        latent_tokens = self.latent_proj(z_day).unsqueeze(1).expand(-1, seq_len, -1)
        latent_tokens = self.input_norm(latent_tokens)
        latent_tokens = self.pos_enc(latent_tokens)
        dec_out = self.temporal_decoder(latent_tokens)
        return self.output_proj(dec_out)


class DayAutoEncoder(nn.Module):
    """
    Convenience wrapper for future inference / export workflows.
    """

    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        latent_dim: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoEncoderEncoder(
            f_in=f_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = AutoEncoderDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            f_out=f_in,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def encode_day(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_day, enc_out = self.encoder(x)
        x_recon = self.decoder(z_day, seq_len=x.size(1))
        return z_day, x_recon, enc_out
