import torch
import torch.nn as nn
import torch.nn.functional as F
 
class RegularizationLoss(nn.Module):
    """
    包含多样性损失、正交性损失、均匀性损失的正则化损失类
    输入：emb - 形状为 [B, d] 的嵌入张量（B=批次大小，d=嵌入维度）
    """
    def __init__(
        self,
        lambda_d: float = 1.0,    # 多样性损失权重
        lambda_o: float = 1.0,    # 正交性损失权重
        lambda_u: float = 1.0,    # 均匀性损失权重
        lambda_f: float = 1.0,    # 前向主任务损失权重
        lambda_b: float = 1.0,    # 反向主任务损失权重
        sigma_thresh: float = 0.1,# 多样性损失触发阈值
        sim_thresh: float = 0.5,  # 均匀性损失触发阈值
        eps: float = 1e-8         # 数值稳定性常数
    ):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.lambda_u = lambda_u
        self.lambda_f = lambda_f
        self.lambda_b = lambda_b
        self.sigma_thresh = sigma_thresh
        self.sim_thresh = sim_thresh
        self.eps = eps

    def diversity_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """
        多样性损失：鼓励嵌入特征维度的标准差足够大，避免分布坍缩
        Args:
            emb: [B, d] 嵌入张量
        Returns:
            loss_div: 标量损失值
        """
        # 1. 按特征维度计算均值 [d]
        mean_e = emb.mean(dim=0)
        # 2. 按特征维度计算标准差 [d]
        std_e = torch.sqrt(((emb - mean_e) ** 2).mean(dim=0) + self.eps)
        # 3. 计算所有维度的平均标准差
        sigma_e = std_e.mean()
        # 4. 指示函数：仅当sigma_e < 阈值时触发惩罚
        cond = (sigma_e < self.sigma_thresh).float()
        # 5. 计算多样性损失
        loss_div = self.lambda_d * cond * (-torch.log(sigma_e + self.eps))
        return loss_div

    def orthogonality_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """
        正交性损失：鼓励嵌入的特征维度正交，减少维度冗余
        Args:
            emb: [B, d] 嵌入张量
        Returns:
            loss_orth: 标量损失值
        """
        B, d = emb.shape
        # 1. 对每一列（特征维度）做L2归一化 [B, d]
        norm_emb = emb / (emb.norm(dim=0, keepdim=True) + self.eps)
        # 2. 计算归一化后的协方差矩阵 [d, d]
        cov_matrix = (norm_emb.T @ norm_emb) / B
        # 3. 单位矩阵 [d, d]
        eye_matrix = torch.eye(d, device=emb.device)
        # 4. 计算Frobenius范数损失
        loss_orth = self.lambda_o * torch.norm(cov_matrix - eye_matrix, p='fro')
        return loss_orth

    def uniformity_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """
        均匀性损失：鼓励批次内样本间的余弦相似度均匀，避免过度相似
        Args:
            emb: [B, d] 嵌入张量
        Returns:
            loss_unif: 标量损失值
        """
        B, _ = emb.shape
        if B < 2:  # 批次大小为1时无样本对，损失为0
            return torch.tensor(0.0, device=emb.device)
        # 1. 对每个样本做L2归一化 [B, d]
        norm_emb = emb / (emb.norm(dim=1, keepdim=True) + self.eps)
        # 2. 计算样本间余弦相似度矩阵 [B, B]
        sim_matrix = norm_emb @ norm_emb.T
        # 3. 屏蔽对角线（i=j的自相似度）
        mask = ~torch.eye(B, dtype=torch.bool, device=emb.device)
        # 4. 计算所有i≠j样本对的平均相似度
        s_bar = sim_matrix[mask].mean()
        # 5. 指示函数：仅当平均相似度 > 阈值时触发惩罚
        cond = (s_bar > self.sim_thresh).float()
        # 6. 计算均匀性损失
        loss_unif = self.lambda_u * cond * s_bar
        return loss_unif

    def total_loss(
        self,
        emb: torch.Tensor,
        loss_forward: torch.Tensor,
        loss_backward: torch.Tensor
    ):
        """
        总损失：主任务损失 + 所有正则化损失
        Args:
            emb: [B, d] 嵌入张量
            loss_forward: 前向主任务损失（标量）
            loss_backward: 反向主任务损失（标量）
        Returns:
            total_loss: 总损失值（标量）
            loss_dict: 各部分损失的字典（方便监控）
        """
        # 计算各正则化损失
        loss_div = self.diversity_loss(emb)
        loss_orth = self.orthogonality_loss(emb)
        loss_unif = self.uniformity_loss(emb)
        
        # 计算总损失
        total_loss = (
            self.lambda_f * loss_forward
            + self.lambda_b * loss_backward
            + loss_div
            + loss_orth
            + loss_unif
        )
        
        # 封装损失字典（方便打印监控）
        loss_dict = {
            "loss_forward": loss_forward.item(),
            "loss_backward": loss_backward.item(),
            "loss_diversity": loss_div.item(),
            "loss_orthogonality": loss_orth.item(),
            "loss_uniformity": loss_unif.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict


# -----------------------------
# 使用示例（可直接运行测试）
# -----------------------------
if __name__ == "__main__":
    # 1. 初始化损失类（可根据需求调整超参数）
    loss_fn = RegularizationLoss(
        lambda_d=0.5,
        lambda_o=0.3,
        lambda_u=0.2,
        lambda_f=1.0,
        lambda_b=1.0
    )
    
    # 2. 模拟输入（批次大小B=32，嵌入维度d=64）
    batch_size = 32
    emb_dim = 64
    emb = torch.randn(batch_size, emb_dim).cuda()  # 模拟GPU张量
    loss_forward = torch.tensor(0.8).cuda()        # 模拟前向主任务损失
    loss_backward = torch.tensor(0.6).cuda()       # 模拟反向主任务损失
    
    # 3. 计算总损失
    total_loss, loss_dict = loss_fn.total_loss(emb, loss_forward, loss_backward)
    
    # 4. 打印结果
    print("各部分损失值：")
    for k, v in loss_dict.items():
        print(f"{k}: {v:.6f}")
    print(f"\n总损失：{total_loss.item():.6f}")