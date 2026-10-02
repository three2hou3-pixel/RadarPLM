"""
训练自编码器决策头（Stage 4）
================================
完全冻结 Stage 3 训练好的 PLM 骨干网络，联合训练 1x1 自编码决策头与任务不确定性标量。

数据流：
    features/{train,val}_tokens_raw.npy + features/mean.npy, std.npy
    → RadarTokenDataset (归一化) → RadarPLMBackbone (冻结, 带 LoRA 权重)
    → RadarAutoencoderHead → logits + F_LN_recon → UncertaintyJointLoss

前置条件：
    1. python process_ipix.py   (提取特征)
    2. python train_reference.py (训练参考模型)
    3. python train_plm.py       (微调 PLM 骨干)

用法：
    python train_autoencoder.py

输出：
    - autoencoder_head.pth : 自编码器决策头 + 不确定性标量权重
"""

import numpy as np
import torch
import os
import sys

from dataset import RadarTokenDataset
from trainer import train_autoencoder_head
from models.plm_model import RadarPLMBackbone
from models.autoencoder_head import RadarAutoencoderHead


# ======================================
# 配置参数
# ======================================
FEATURES_DIR = 'features'

# 特征文件路径
TRAIN_TOKENS_PATH = os.path.join(FEATURES_DIR, 'train_tokens_raw.npy')
TRAIN_LABELS_PATH = os.path.join(FEATURES_DIR, 'train_labels.npy')
VAL_TOKENS_PATH = os.path.join(FEATURES_DIR, 'val_tokens_raw.npy')
VAL_LABELS_PATH = os.path.join(FEATURES_DIR, 'val_labels.npy')
MEAN_PATH = os.path.join(FEATURES_DIR, 'mean.npy')
STD_PATH = os.path.join(FEATURES_DIR, 'std.npy')

# Stage 3 训练好的 PLM 权重
PLM_MODEL_PATH = 'plm_model.pth'

# LoRA 超参数（需与 Stage 3 保持一致）
LORA_R = 8
LORA_ALPHA = 16

# 训练超参数（严格对齐论文 Table II）
EPOCHS = 300
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 400
LR = 1e-5
PATIENCE = 30


def check_files():
    """检查所有必需的文件是否存在"""
    required = [
        (TRAIN_TOKENS_PATH, '训练特征'),
        (TRAIN_LABELS_PATH, '训练标签'),
        (VAL_TOKENS_PATH, '验证特征'),
        (VAL_LABELS_PATH, '验证标签'),
        (MEAN_PATH, '归一化均值'),
        (STD_PATH, '归一化标准差'),
        (PLM_MODEL_PATH, 'PLM 模型权重 (Stage 3)'),
    ]
    missing = [f'  {desc}: {path}' for path, desc in required if not os.path.exists(path)]
    if missing:
        print('错误: 缺少必需文件，请先运行:')
        print('  1. python process_ipix.py')
        print('  2. python train_reference.py')
        print('  3. python train_plm.py')
        print('缺少的文件:')
        print('\n'.join(missing))
        sys.exit(1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # ============================================
    # 1. 加载 features/ 中已提取好的特征
    # ============================================
    check_files()

    print('加载特征...')
    train_tokens = np.load(TRAIN_TOKENS_PATH)   # (N, 55, 48)
    train_labels = np.load(TRAIN_LABELS_PATH)    # (N,)
    val_tokens = np.load(VAL_TOKENS_PATH)        # (N, 55, 48)
    val_labels = np.load(VAL_LABELS_PATH)         # (N,)
    mean = np.load(MEAN_PATH)                     # (1, 55, 48)
    std = np.load(STD_PATH)                       # (1, 55, 48)

    print(f'  训练集: {train_tokens.shape}, 目标: {train_labels.sum():.0f}')
    print(f'  验证集: {val_tokens.shape}, 目标: {val_labels.sum():.0f}')

    # ============================================
    # 2. 初始化 PLM 骨干网络，加载 Stage 3 LoRA 权重
    # ============================================
    print('\n初始化 RadarPLMBackbone (GPT-2 4层 + LoRA)...')
    backbone = RadarPLMBackbone(r=LORA_R, lora_alpha=LORA_ALPHA)

    # plm_model.pth 保存的是 RadarPLMWithHeadForStage3 的 state_dict，
    # 包含 backbone.* (LoRA/嵌入层等) 和 layer1.*, layer2.* (分类头)。
    # 我们只需要 backbone.* 部分，去掉前缀后加载到 RadarPLMBackbone。
    print(f'\n加载 PLM 权重 ({PLM_MODEL_PATH})...')
    plm_ckpt = torch.load(PLM_MODEL_PATH, map_location=device, weights_only=True)

    # 兼容两种保存格式：dict 或嵌套的 {'model_state_dict': ...}
    plm_sd = plm_ckpt.get('model_state_dict', plm_ckpt)

    # 过滤并去掉 backbone. 前缀
    bb_sd = {
        k[len('backbone.'):]: v
        for k, v in plm_sd.items()
        if k.startswith('backbone.')
    }
    backbone.load_state_dict(bb_sd, strict=False)
    print(f'  已加载 {len(bb_sd)} 个 backbone 权重参数')

    # ── 彻底封印：Stage 4 中 backbone 只读，不参与梯度 ──
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f'  ✅ 主干已封印 (冻结 {sum(p.numel() for p in backbone.parameters()):,} 个参数)')

    # ============================================
    # 3. 初始化自编码器决策头
    # ============================================
    print('\n初始化 RadarAutoencoderHead...')
    ae_head = RadarAutoencoderHead()

    ae_params = sum(p.numel() for p in ae_head.parameters())
    print(f'  自编码器参数量: {ae_params:,} (可训练: {sum(p.numel() for p in ae_head.parameters() if p.requires_grad):,})')

    # ============================================
    # 4. Stage 4 联合训练
    # ============================================
    print(f'\n开始 Stage 4 自编码器联合重训 (epoch={EPOCHS}, lr={LR})...')
    best_val_loss = train_autoencoder_head(
        backbone=backbone,
        ae_head=ae_head,
        train_tokens=train_tokens,
        train_labels=train_labels,
        val_tokens=val_tokens,
        val_labels=val_labels,
        mean=mean,
        std=std,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        lr=LR,
        save_path='autoencoder_head.pth',
        patience=PATIENCE
    )

    print(f'\n✅ Stage 4 自编码器决策头训练完成！')
    print(f'💾 最终权重已保存至: autoencoder_head.pth')
    print(f'📊 最优联合验证损失: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
