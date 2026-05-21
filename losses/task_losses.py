"""
losses/task_losses.py — 主任务损失函数
encoder_loss_fn 和 decoder_loss_fn 逻辑与原始代码完全一致，
TRADE_IDX 改为函数参数传入，方便灵活配置。
"""
import torch
import torch.nn.functional as F


def encoder_loss_fn(enc_out_recon, x_original, mask, trade_idx, eps=1e-8):
    """
    Encoder损失：在原始15维空间计算被掩码交易特征的重建损失
    Args:
        enc_out_recon: Encoder反投影后的15维输出 [B,T,15]
        x_original:   原始未掩码数据 [B,T,15]
        mask:         Encoder生成的交易特征掩码 [B,T,len(trade_idx)]（None则损失为0）
        trade_idx:    交易特征的列索引列表
        eps:          数值稳定性常数
    Returns:
        平均MSE损失
    """
    if mask is None or mask.sum() == 0:
        return torch.tensor(0.0, device=x_original.device)

    # 提取原始交易特征和重建的交易特征
    x_trade_original = x_original[:, :, trade_idx] + eps  # [B,T,len(trade_idx)]
    x_trade_recon    = enc_out_recon[:, :, trade_idx]      # [B,T,len(trade_idx)]

    # 仅计算被掩码位置的重建损失（mask=True的位置是被掩码的）
    loss = F.mse_loss(x_trade_recon[mask], x_trade_original[mask])

    # 兜底：防止NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=x_original.device)

    return loss


def decoder_loss_fn(
    pred_price: torch.Tensor,
    target_price: torch.Tensor,
    price_weights: list = [1.0, 1.0, 1.0,1.0],  # 价格维度权重（开盘/收盘/最高）
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Decoder损失函数：计算预测价格与真实价格的加权MSE损失
    适配移位后的因果预测逻辑，保证数值稳定性

    Args:
        pred_price:    Decoder输出的预测价格，shape=[B, T, f_price]（f_price=3）
        target_price:  真实价格标签，shape=[B, T, f_price]
        price_weights: 各价格维度的权重，长度需等于f_price（默认等权重）
        eps:           数值稳定性常数，避免MSE计算中出现0/0

    Returns:
        loss: 标量损失值（加权MSE）
    """
    # 1. 维度校验（防止输入维度不匹配）
    if pred_price.shape != target_price.shape:
        raise ValueError(f"预测价格维度{pred_price.shape}与真实价格维度{target_price.shape}不匹配！")

    B, T, f_price = pred_price.shape

    # 2. 权重适配（确保权重长度与价格维度一致）
    if len(price_weights) != f_price:
        raise ValueError(f"价格权重长度{len(price_weights)}需等于价格维度{f_price}！")
    price_weights = torch.tensor(price_weights, device=pred_price.device, dtype=pred_price.dtype)

    # 3. 计算每个价格维度的MSE损失（加eps保证数值稳定）
    mse_per_dim = F.mse_loss(
        pred_price + eps,    # 预测值加eps避免数值下溢
        target_price + eps,
        reduction='none'     # [B, T, f_price]
    )

    # 4. 应用维度权重，计算加权MSE
    weighted_mse = mse_per_dim * price_weights.unsqueeze(0).unsqueeze(0)  # 广播到所有维度

    # 5. 计算平均损失（按批次、时间步、维度平均）
    loss = weighted_mse.mean()

    # 兜底：防止NaN/Inf（极端情况防护）
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=pred_price.device, dtype=pred_price.dtype)

    return loss
