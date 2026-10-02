import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
# ---------- PLM 骨干大模型 ----------

# ======================================

# 1. 768 维雷达特征嵌入层（含 LayerNorm 层）
class GPT2RadarEmbedding(nn.Module):
    def __init__(self, patch_size=48, embed_dim=768, max_tokens=55):
        super().__init__()
        self.projection = nn.Linear(patch_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.normal_(self.position_embeddings, std=0.02)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_emb = self.projection(x) + self.position_embeddings
        return self.dropout(self.layer_norm(x_emb))
    
# 2. 针对 GPT-2 特殊 c_attn 层的 LoRA 包裹器
class LoRAGPT2AttentionWrapper(nn.Module):
    def __init__(self, base_conv1d, embed_dim=768, r=8, lora_alpha=16):
        """
        将原始的 c_attn 封包。精准在 Q 和 V 分支上挂载 LoRA，绕过 K 分支
        """
        super().__init__()
        self.base_layer = base_conv1d               # 官方预训练的 c_attn 层
        self.r = r
        self.scaling = lora_alpha / r
        
        # 针对 Query 的 LoRA 矩阵
        self.lora_A_q = nn.Linear(embed_dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, embed_dim, bias=False)

        # 针对 Value 的 LoRA 矩阵
        self.lora_A_v = nn.Linear(embed_dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, embed_dim, bias=False)

        # 按论文要求初始化 LoRA 矩阵
        nn.init.normal_(self.lora_A_q.weight, std=1.0 / r)
        nn.init.normal_(self.lora_A_v.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        # 1. 让输入通过官方预训练的骨干矩阵，拿到基础三合一投影
        base_out = self.base_layer(x)                   # (B, 55, 768 * 3)

        # 2. 并行计算 Q 和 V 的增量 Delta W
        delta_q = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        delta_v = self.lora_B_v(self.lora_A_v(x)) * self.scaling

        # 3. 将矩阵切成三段，修改 Q 和 V 后再拼接
        q_base, k_base, v_base = torch.split(base_out, 768, dim=-1)
        q_final = q_base + delta_q
        k_final = k_base
        v_final = v_base + delta_v

        return torch.cat([q_final, k_final, v_final], dim=-1)
    
# 3. 裁剪版 RadarPLM 主干网络模型
class RadarPLMBackbone(nn.Module):
    def __init__(self, r=8, lora_alpha=16):
        super().__init__()
        
        # 实例化雷达大模型嵌入投影层
        self.radar_embedding = GPT2RadarEmbedding()
        
        # 载入 HuggingFace 官方开源 GPT-2 骨架
        print('正在载入 HuggingFace GPT-2 官方预训练权重...')
        config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2', config=config)
        
        # 裁剪模型，只保留前 4 个编码层(L=4)
        self.gpt2.h = nn.ModuleList([self.gpt2.h[i] for i in range(4)])
        print('-> 成功执行大模型物理裁剪！当前骨干网络层数：4 层')

        for i in range(4):
            official_c_attn = self.gpt2.h[i].attn.c_attn # type: ignore
            self.gpt2.h[i].attn.c_attn = LoRAGPT2AttentionWrapper( # type: ignore
                base_conv1d=official_c_attn,
                embed_dim=768,
                r=r,
                lora_alpha=lora_alpha
            )
        
        print('-> 成功在 4 个 Attention 层的 Q、V 分支注入 LoRA 自适应算子')

        for name, param in self.named_parameters():
            if 'lora_' in name or 'ln_' in name or "layer_norm" in name or 'radar_embedding' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        hidden_states = self.radar_embedding(x)
        gpt2_outputs = self.gpt2(inputs_embeds=hidden_states)

        F_LN = gpt2_outputs.last_hidden_state

        return F_LN
    
# 4. 带临时 Token 分类头的 Stage3 骨干网络训练外壳
class RadarPLMWithHeadForStage3(nn.Module):
    """
    stage3 专用微调外壳：骨干网络 + 临时双层 Token 分类头
    用于使用偏好感知损失微调 GPT-2 主干
    """
    def __init__(self, backbone, embed_dim=768, hidden_dim=32, num_classes=2):
        """
        参数：
        ----------
        backbone    : 裁剪版 GPT-2 骨干网络
        embed_dim   : 大模型特征维度，固定为 768
        hidden_dim  : 双层预测头的中间隐藏层维度，默认设为 32
        num_classes : 最终 Token 分类数，固定为 2
        """
        super().__init__()
        self.backbone = backbone
        
        # 对齐论文：构建双层输出线性层 (Output Linear Layer * 2) ───
        self.layer1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()                              # 注入非线性激活，防止两层线性矩阵直接塌陷合并
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        
        # 严格按照大模型微调标准对两层线性层执行高斯初始化
        for layer in [self.layer1, self.layer2]:
            nn.init.normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 1. 抽取大模型倒数第二层高维物理特征流 -> 形状: (B, 55, 768)
        F_LN = self.backbone(x)                 
        
        # 2. 数据流经双层预测头，逐个 Token 进行非线性决策映射
        out = self.layer1(F_LN)                 # (B, 55, 768) -> (B, 55, hidden_dim)
        out = self.act(out)
        logits = self.layer2(out)               # (B, 55, hidden_dim) -> (B, 55, 2)

        return logits, F_LN