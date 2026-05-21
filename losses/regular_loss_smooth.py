import torch
import torch.nn as nn
import torch.nn.functional as F
 
class RegularizationLossSmooth(nn.Module):
    """
    包含多样性损失、正交性损失、均匀性损失的平滑版正则化损失类
    专门修复了原代数学上截断断崖和协方差计算对角线常数的 Bug。
    """
    def __init__(
        self,
        lambda_d: float = 1.0,    
        lambda_o: float = 1.0,    
        lambda_u: float = 1.0,    
        lambda_f: float = 1.0,    
        lambda_b: float = 1.0,    
        sigma_thresh: float = 0.1,
        sim_thresh: float = 0.5,  
        eps: float = 1e-8         
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
        """
        mean_e = emb.mean(dim=0)
        std_e = torch.sqrt(((emb - mean_e) ** 2).mean(dim=0) + self.eps)
        sigma_e = std_e.mean()
        # 用平滑铰链(Hinge Loss)代替硬截断(指示函数)，让梯度平稳连贯不突变
        loss_div = self.lambda_d * F.relu(self.sigma_thresh - sigma_e)
        return loss_div

    def orthogonality_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """
        正交性损失：鼓励嵌入的特征维度正交，减少维度冗余
        """
        B, d = emb.shape
        # 1. 首先减去均值中心化（计算相关系数的前提）
        emb_centered = emb - emb.mean(dim=0, keepdim=True)
        # 2. 对每一列做 L2 归一化。这代表每列（特征）向量的长度正好为 1。
        norm_emb = emb_centered / (emb_centered.norm(dim=0, keepdim=True) + self.eps)
        # 3. 计算相关系数矩阵 [d, d]。
        # 绝对不能除以 B！由于 norm_emb 已经按列 L2 归一化，norm_emb.T @ norm_emb 算出来的对角线必定是 1。
        corr_matrix = norm_emb.T @ norm_emb
        eye_matrix = torch.eye(d, device=emb.device)
        # 4. 使用 MSE 计算相差度，不使用 Frobenius 范数以避免 0 点梯度不可导的隐患。
        loss_orth = self.lambda_o * F.mse_loss(corr_matrix, eye_matrix)
        return loss_orth

    def uniformity_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """
        均匀性损失：鼓励批次内样本间分布均匀，避免过度相似甚至相向扎堆
        """
        B, _ = emb.shape
        if B < 2:  
            return torch.tensor(0.0, device=emb.device)
        # 1. 样本做 L2 归一化 [B, d]
        norm_emb = emb / (emb.norm(dim=1, keepdim=True) + self.eps)
        # 2. 样本间余弦相似度矩阵 [B, B]
        sim_matrix = norm_emb @ norm_emb.T
        # 3. 屏蔽自相似度对角线
        mask = ~torch.eye(B, dtype=torch.bool, device=emb.device)
        # 4. 防抵消漏洞：正相似度和负相似度互相抵消会骗过原版 loss，这里必须先做平方！
        s_sq_bar = (sim_matrix[mask] ** 2).mean()
        loss_unif = self.lambda_u * s_sq_bar
        return loss_unif

    def total_loss(
        self,
        emb: torch.Tensor,
        loss_forward: torch.Tensor,
        loss_backward: torch.Tensor
    ):
        loss_div = self.diversity_loss(emb)
        loss_orth = self.orthogonality_loss(emb)
        loss_unif = self.uniformity_loss(emb)
        
        total_loss = (
            self.lambda_f * loss_forward
            + self.lambda_b * loss_backward
            + loss_div
            + loss_orth
            + loss_unif
        )
        
        loss_dict = {
            "loss_forward": loss_forward.item(),
            "loss_backward": loss_backward.item(),
            "loss_diversity": loss_div.item(),
            "loss_orthogonality": loss_orth.item(),
            "loss_uniformity": loss_unif.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
