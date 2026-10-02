import torch
import torch.nn as nn

# ---------- 参考模型 ----------

# ===============================================

# 1. 32 维参考模型嵌入层
class RadarEmbedding(nn.Module):
    def __init__(self, patch_size=48, embed_dim=32, max_tokens=55):
        """
        雷达特征块的自适应嵌入层（线性投影 + 可学习的位置编码）

        参数：
        ----------
        patch_size：输入特征块的长度 L，即 Token 长度，论文中为 48
        embed_dim：映射到的隐藏层维度 D，参考模型为 32，主干模型 GPT-2 为 768
        max_token：一个样本中的最大 Token 数量 K，论文配置为 55
        """
        super().__init__()
        # 线性投影层
        self.projection = nn.Linear(patch_size, embed_dim)
        # 可学习的位置编码：形状为 (1, K, D)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        # 正态分布对位置编码进行初始化，加速收敛
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        x_projected = self.projection(x)                # (B, K, L) -> (B, K, D)
        x_embedded = x_projected + self.position_embeddings             # (B, K, D)

        return x_embedded
    
# 2. 轻量级参考模型架构
class RadarReferenceModel(nn.Module):
    def __init__(self, patch_size=48, embed_dim=32, max_tokens=55, nhead=8, num_layers=2, dim_feedforward=128):
        """
        按推测出的轻量级参考模型配置：
        2 层，输入维度 32，8 头，FFN 隐层维度 128
        """
        super().__init__()

        # 1. 基础嵌入层（将 48 维投影到 32 维主干空间，并叠加位置编码）
        self.embedding = RadarEmbedding(patch_size=patch_size, embed_dim=embed_dim, max_tokens=max_tokens)

        # 2. 单层 Transformer 编码器架构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='relu',
            dropout=0.1
        )
        
        # 3. 组装多层编码器（2层）
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 4. 后端二分类头
        self.classifier = nn.Linear(embed_dim, 2)           # 每个 token 独立输出预测

    def forward(self, x):
        x_emb = self.embedding(x)
        x_enc = self.transformer_encoder(x_emb)
        logits = self.classifier(x_enc)                 # (B, 55, 2)

        return logits