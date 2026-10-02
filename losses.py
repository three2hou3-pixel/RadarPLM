import torch
import torch.nn as nn

# ---------- 偏好感知损失函数 ----------
class PreferenceAwareLoss(nn.Module):
    def __init__(self, alpha=0.9):
        """
        用于微调骨干 PLM 大模型
        """
        super().__init__()
        self.alpha = alpha
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, plm_logits, ref_logits, labels):
        """
        参数：
        ----------
        plm_logits：主模型输出 (B, 55, 2)
        ref_logits：参考模型输出 (B, 55, 2)
        labels：token 级标签(B, 55)
        """

        # 输入形状检查
        assert plm_logits.shape == ref_logits.shape, "主模型和参考模型输出形状不匹配"
        assert plm_logits.shape[:2] == labels.shape, "logits 和标签形状不匹配"

        # 展平计算交叉熵
        B, K, C = plm_logits.shape
        plm_logits_flat = plm_logits.reshape(-1, C)             # (B * 55, 2)
        ref_logits_flat = ref_logits.reshape(-1, C)             # (B * 55, 2)
        labels_flat = labels.reshape(-1)

        # 计算每个 token 的独立损失
        loss_plm = self.base_loss_fn(plm_logits_flat, labels_flat)
        loss_ref = self.base_loss_fn(ref_logits_flat, labels_flat).detach()   # 让参考模型的损失变成 “只读” 模式

        # 计算每个 token 的学习价值：只有当 PLM 模型损失 > α * 参考模型损失时，才会被认为有训练价值
        learning_value = torch.relu(loss_plm - self.alpha * loss_ref)
        
        final_loss = torch.mean(learning_value * loss_plm)

        return final_loss, learning_value.reshape(B, K)
    


# ---------- 不确定性自适应联合损失函数 ----------
class UncertaintyJointLoss(nn.Module):
    def __init__(self):
        """
        用于最后的自编码器二分类头的训练
        """
        super(UncertaintyJointLoss, self).__init__()
        
        # 声明两个可学习的对数方差标量参数
        # 初始化为 0.0，意味着初始权重系数 exp(-0) 恰好为 1，给予两轨子损失公平的初始导数步长
        self.log_var_recon = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_ce = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # 底层子任务算子
        self.mse_loss = nn.MSELoss(reduction='mean')  # 对应公式(21)的 L2 范数平均化形式
        self.ce_loss = nn.CrossEntropyLoss()         # 对应公式(22)的交叉熵形式

    def forward(self, logits, F_LN_recon, targets, F_LN_true):
        """
        返回:
        ----------
        total_loss : torch.Tensor -> 用于 backward 的总标量
        metrics_dict : dict -> 包含所有子任务损失以及【核心权重】的指标监控
        """
        # 1. 基础子任务 Loss 计算
        loss_recon = self.mse_loss(F_LN_recon, F_LN_true)
        loss_ce = self.ce_loss(logits, targets)
        
        # 2. 数学变换还原
        # precision 就是最终乘在 Loss 前面的实际倒数权重系数
        precision_recon = torch.exp(-self.log_var_recon)
        precision_ce = torch.exp(-self.log_var_ce)
        
        # 3. 还原论文公式中的物理标准差 sigma 
        # sigma = sqrt(exp(log_var)) = exp(0.5 * log_var)
        sigma_recon = torch.exp(0.5 * self.log_var_recon).item()
        sigma_ce = torch.exp(0.5 * self.log_var_ce).item()
        
        # 4. 组装多任务总损失
        term_recon = 0.5 * precision_recon * loss_recon
        term_ce = precision_ce * loss_ce
        term_penalty = 0.5 * (self.log_var_recon + self.log_var_ce)
        
        total_loss = term_recon + term_ce + term_penalty
        
        # 5. 监控字典，为 Trainer 提供全方位的视觉数据支持
        metrics_dict = {
            'loss_recon_pure': loss_recon.item(),           # 未加权的纯净重构均方误差
            'loss_ce_pure': loss_ce.item(),                 # 未加权的纯净分类交叉熵
            'weight_recon': (0.5 * precision_recon).item(), # 最终在重构Loss上的实际乘数
            'weight_ce': precision_ce.item(),               # 最终在分类Loss上的实际乘数
            'sigma_recon': sigma_recon,                     # 重构任务的物理不确定性标准差
            'sigma_ce': sigma_ce                            # 分类任务的物理不确定性标准差
        }
        
        return total_loss, metrics_dict