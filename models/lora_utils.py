"""
Utilities for applying LoRA to the current encoder stack.

This file is intentionally standalone so we can:
1. keep the existing training path unchanged;
2. experiment with freeze-only / freeze+LoRA finetuning side by side;
3. attach LoRA only to the last transformer blocks of the encoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAAttachReport:
    target_layer_indices: List[int]
    target_module_names: List[str]
    trainable_param_count: int
    total_param_count: int


class LoRALinear(nn.Module):
    """
    A minimal LoRA wrapper around an existing nn.Linear.

    Forward:
        y = base_linear(x) + scale * B(A(dropout(x)))
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA.")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_linear.out_features, bias=False)
        target_device = base_linear.weight.device
        target_dtype = base_linear.weight.dtype
        self.lora_a.to(device=target_device, dtype=target_dtype)
        self.lora_b.to(device=target_device, dtype=target_dtype)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def lora_delta_weight(self) -> torch.Tensor:
        return (self.lora_b.weight @ self.lora_a.weight) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        """
        Expose an effective weight for modules that directly read `.weight`
        instead of calling this module's `forward()` (e.g. PyTorch fused
        fastpaths in TransformerEncoderLayer during eval).

        Note:
        - this is exact when LoRA dropout is disabled or in eval mode;
        - for training-time functional fastpaths, LoRA dropout cannot be fully
          folded into a static weight.
        """
        return self.base_linear.weight + self.lora_delta_weight()

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRAMultiheadAttention(nn.Module):
    """
    Wrap nn.MultiheadAttention and inject LoRA into the packed Q/V projections.

    The upstream encoder still uses the stock TransformerEncoderLayer, so we
    keep the base module shape intact and only replace `self_attn` during
    finetuning / inference reconstruction.
    """

    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        enable_q: bool = True,
        enable_v: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA.")
        if not enable_q and not enable_v:
            raise ValueError("At least one of enable_q / enable_v must be True.")
        if base_attn.kdim not in (None, base_attn.embed_dim):
            raise NotImplementedError("LoRAMultiheadAttention only supports kdim == embed_dim.")
        if base_attn.vdim not in (None, base_attn.embed_dim):
            raise NotImplementedError("LoRAMultiheadAttention only supports vdim == embed_dim.")
        if base_attn.bias_k is not None or base_attn.bias_v is not None:
            raise NotImplementedError("LoRAMultiheadAttention does not support bias_k / bias_v.")
        if base_attn.add_zero_attn:
            raise NotImplementedError("LoRAMultiheadAttention does not support add_zero_attn.")

        self.base_attn = base_attn
        self.embed_dim = base_attn.embed_dim
        self.num_heads = base_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = base_attn.batch_first
        self._qkv_same_embed_dim = True
        self.kdim = base_attn.kdim
        self.vdim = base_attn.vdim
        self.add_zero_attn = base_attn.add_zero_attn
        self.bias_k = base_attn.bias_k
        self.bias_v = base_attn.bias_v
        self.attn_dropout = float(base_attn.dropout)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.enable_q = enable_q
        self.enable_v = enable_v
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        # Keep `self_attn.out_proj` reachable after wrapping so mixed targets
        # like q/v + out_proj still work.
        original_out_proj = base_attn.out_proj
        self.base_attn.out_proj = nn.Identity()
        self.out_proj = original_out_proj

        for param in self.base_attn.parameters():
            param.requires_grad = False
        for param in self.out_proj.parameters():
            param.requires_grad = False

        target_device = self.base_attn.in_proj_weight.device
        target_dtype = self.base_attn.in_proj_weight.dtype

        if self.enable_q:
            self.q_lora_a = nn.Linear(self.embed_dim, rank, bias=False)
            self.q_lora_b = nn.Linear(rank, self.embed_dim, bias=False)
            self.q_lora_a.to(device=target_device, dtype=target_dtype)
            self.q_lora_b.to(device=target_device, dtype=target_dtype)
            nn.init.kaiming_uniform_(self.q_lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.q_lora_b.weight)

        if self.enable_v:
            self.v_lora_a = nn.Linear(self.embed_dim, rank, bias=False)
            self.v_lora_b = nn.Linear(rank, self.embed_dim, bias=False)
            self.v_lora_a.to(device=target_device, dtype=target_dtype)
            self.v_lora_b.to(device=target_device, dtype=target_dtype)
            nn.init.kaiming_uniform_(self.v_lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.v_lora_b.weight)

    def _q_delta_weight(self) -> torch.Tensor:
        if not self.enable_q:
            return torch.zeros_like(self.base_attn.in_proj_weight[: self.embed_dim])
        return (self.q_lora_b.weight @ self.q_lora_a.weight) * self.scaling

    def _v_delta_weight(self) -> torch.Tensor:
        if not self.enable_v:
            return torch.zeros_like(self.base_attn.in_proj_weight[: self.embed_dim])
        return (self.v_lora_b.weight @ self.v_lora_a.weight) * self.scaling

    @property
    def in_proj_weight(self) -> torch.Tensor:
        """
        Expose an effective packed QKV weight for Transformer fastpaths.

        This is exact in eval mode, which is the relevant case because PyTorch's
        encoder-layer fused path is only considered when gradients are disabled.
        """
        e = self.embed_dim
        weight = self.base_attn.in_proj_weight
        w_q = weight[:e] + self._q_delta_weight()
        w_k = weight[e: 2 * e]
        w_v = weight[2 * e:] + self._v_delta_weight()
        return torch.cat([w_q, w_k, w_v], dim=0)

    @property
    def in_proj_bias(self) -> torch.Tensor | None:
        return self.base_attn.in_proj_bias

    def merge_masks(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        query: torch.Tensor,
    ):
        return self.base_attn.merge_masks(attn_mask, key_padding_mask, query)

    def _to_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x
        return x.transpose(0, 1)

    def _from_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x
        return x.transpose(0, 1)

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.embed_dim
        weight = self.base_attn.in_proj_weight
        bias = self.base_attn.in_proj_bias

        w_q = weight[:e]
        w_k = weight[e: 2 * e]
        w_v = weight[2 * e:]

        if bias is None:
            b_q = b_k = b_v = None
        else:
            b_q = bias[:e]
            b_k = bias[e: 2 * e]
            b_v = bias[2 * e:]

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        if self.enable_q:
            q = q + self.q_lora_b(self.q_lora_a(self.dropout(query))) * self.scaling
        if self.enable_v:
            v = v + self.v_lora_b(self.v_lora_a(self.dropout(value))) * self.scaling

        return q, k, v

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def _apply_attention_mask(self, attn_scores: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if attn_mask.dim() == 2:
            mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            if attn_mask.shape[0] == attn_scores.shape[0] * attn_scores.shape[1]:
                mask = attn_mask.view(attn_scores.shape[0], attn_scores.shape[1], *attn_mask.shape[-2:])
            elif attn_mask.shape[0] == attn_scores.shape[0]:
                mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError("Unsupported 3D attn_mask shape for LoRAMultiheadAttention.")
        elif attn_mask.dim() == 4:
            mask = attn_mask
        else:
            raise ValueError("attn_mask must be 2D, 3D or 4D.")

        if mask.dtype == torch.bool:
            return attn_scores.masked_fill(mask.to(device=attn_scores.device), float("-inf"))
        return attn_scores + mask.to(device=attn_scores.device, dtype=attn_scores.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_bf = self._to_batch_first(query)
        key_bf = self._to_batch_first(key)
        value_bf = self._to_batch_first(value)

        q, k, v = self._project_qkv(query_bf, key_bf, value_bf)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores = self._apply_attention_mask(attn_scores, attn_mask)

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be 2D.")
            key_padding_mask = key_padding_mask.to(device=attn_scores.device)
            if key_padding_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            else:
                attn_scores = attn_scores + key_padding_mask[:, None, None, :].to(dtype=attn_scores.dtype)

        if is_causal:
            q_len = attn_scores.shape[-2]
            k_len = attn_scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self._from_batch_first(attn_output)

        if not need_weights:
            return attn_output, None

        if average_attn_weights:
            attn_weights_out = attn_weights.mean(dim=1)
        else:
            attn_weights_out = attn_weights
        return attn_output, attn_weights_out


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _replace_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def _resolve_target_module(root: nn.Module, dotted_path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child_name = parts[-1]
    child = getattr(parent, child_name)
    return parent, child_name, child


def collect_trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def attach_lora_to_encoder(
    encoder: nn.Module,
    num_last_layers: int = 2,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("self_attn.q_proj", "self_attn.v_proj"),
    freeze_fixed_proj: bool = True,
    freeze_input_norm: bool = True,
    freeze_proj_back: bool = True,
    freeze_transformer_norm: bool = True,
) -> LoRAAttachReport:
    """
    Attach LoRA to the last N encoder blocks.

    Recommended default for the current encoder:
    - freeze fixed_proj / input_norm / proj_back;
    - freeze all transformer blocks first;
    - attach LoRA only to the last 2 blocks on:
        * self_attn.q_proj
        * self_attn.v_proj
      optionally also:
        * self_attn.out_proj
        * linear1
        * linear2
    """

    if not hasattr(encoder, "transformer") or not hasattr(encoder.transformer, "layers"):
        raise TypeError("encoder must expose transformer.layers to attach LoRA.")

    layers = encoder.transformer.layers
    num_layers = len(layers)
    if num_last_layers <= 0 or num_last_layers > num_layers:
        raise ValueError(f"num_last_layers must be in [1, {num_layers}].")

    if freeze_fixed_proj and hasattr(encoder, "fixed_proj"):
        freeze_module(encoder.fixed_proj)
    if freeze_input_norm and hasattr(encoder, "input_norm"):
        freeze_module(encoder.input_norm)
    if freeze_proj_back and hasattr(encoder, "proj_back"):
        freeze_module(encoder.proj_back)
    if freeze_transformer_norm and hasattr(encoder.transformer, "norm") and encoder.transformer.norm is not None:
        freeze_module(encoder.transformer.norm)

    for layer in layers:
        freeze_module(layer)

    target_layer_indices = list(range(num_layers - num_last_layers, num_layers))
    attached_module_names: List[str] = []

    for layer_idx in target_layer_indices:
        layer = layers[layer_idx]
        qv_targets = {name for name in target_modules if name in {"self_attn.q_proj", "self_attn.v_proj"}}
        if qv_targets:
            layer.self_attn = LoRAMultiheadAttention(
                base_attn=layer.self_attn,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                enable_q="self_attn.q_proj" in qv_targets,
                enable_v="self_attn.v_proj" in qv_targets,
            )
            for module_name in sorted(qv_targets):
                attached_module_names.append(f"layers[{layer_idx}].{module_name}")

        remaining_targets = [name for name in target_modules if name not in qv_targets]
        for module_name in remaining_targets:
            parent, child_name, child = _resolve_target_module(layer, module_name)
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"Only nn.Linear is supported for LoRA right now, "
                    f"but {module_name} in layer {layer_idx} is {type(child).__name__}."
                )
            _replace_child_module(
                parent,
                child_name,
                LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            attached_module_names.append(f"layers[{layer_idx}].{module_name}")

    trainable_params = collect_trainable_parameters(encoder)
    all_params = list(encoder.parameters())
    return LoRAAttachReport(
        target_layer_indices=target_layer_indices,
        target_module_names=attached_module_names,
        trainable_param_count=count_parameters(trainable_params),
        total_param_count=count_parameters(all_params),
    )
