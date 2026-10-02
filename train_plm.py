"""
训练 PLM 大模型（偏好感知微调，Stage 3）
===========================================
在 train 集上微调 GPT-2 骨干网络，用参考模型作为"考官"，
只在 PLM 比参考模型表现差的位置激活梯度（PreferenceAwareLoss）。

数据流：
    features/{train,val}_tokens_raw.npy  +  features/mean.npy, std.npy
    → RadarTokenDataset (归一化) → RadarPLMBackbone → PreferenceAwareLoss
    参考模型 RadarReferenceModel ← reference_model.pth (冻结)

前置条件：
    1. 先跑 process_ipix.py 提取特征
    2. 先跑 train_reference.py 训练参考模型

用法：
    python train_plm.py

输出：
    - plm_model.pth : LoRA + LayerNorm + 嵌入层 + 分类头权重
"""

import numpy as np
import torch
import os
import sys

from dataset import RadarTokenDataset
from trainer import train_plm_model
from models.reference_model import RadarReferenceModel
from models.plm_model import RadarPLMBackbone, RadarPLMWithHeadForStage3


# ======================================
# 配置参数
# ======================================
FEATURES_DIR = 'features'

# 训练数据（PLM 用 train 集，参考模型在 val 集上练的）
TRAIN_TOKENS_PATH = os.path.join(FEATURES_DIR, 'train_tokens_raw.npy')
TRAIN_LABELS_PATH = os.path.join(FEATURES_DIR, 'train_labels.npy')
VAL_TOKENS_PATH = os.path.join(FEATURES_DIR, 'val_tokens_raw.npy')
VAL_LABELS_PATH = os.path.join(FEATURES_DIR, 'val_labels.npy')
MEAN_PATH = os.path.join(FEATURES_DIR, 'mean.npy')
STD_PATH = os.path.join(FEATURES_DIR, 'std.npy')
REF_MODEL_PATH = 'reference_model.pth'

# PLM 超参数
PATCH_SIZE = 48
EMBED_DIM = 32
MAX_TOKENS = 55
NHEAD = 8
NUM_LAYERS = 2
DIM_FF = 128

# LoRA 超参数
LORA_R = 8
LORA_ALPHA = 16

# 训练超参数
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4
ALPHA = 0.9          # PreferenceAwareLoss 阈值
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
        (REF_MODEL_PATH, '参考模型权重'),
    ]
    missing = [f'  {desc}: {path}' for path, desc in required if not os.path.exists(path)]
    if missing:
        print('错误: 缺少必需文件，请先运行:')
        print('  1. python process_ipix.py  (提取特征)')
        print('  2. python train_reference.py  (训练参考模型)')
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
    print(f'  归一化: mean {mean.shape}, std {std.shape}')

    # ============================================
    # 2. 加载参考模型（冻结，当考官）
    # ============================================
    print('\n加载参考模型...')
    ref_model = RadarReferenceModel(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        max_tokens=MAX_TOKENS,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF
    )
    ref_model.load_state_dict(torch.load(REF_MODEL_PATH, map_location=device, weights_only=True))
    ref_model.to(device)
    ref_model.eval()
    print(f'  参考模型已加载 ({REF_MODEL_PATH})')

    # ============================================
    # 3. 初始化 PLM 骨干网络 + 分类头
    # ============================================
    print('\n初始化 RadarPLMBackbone (GPT-2 4层 + LoRA)...')
    backbone = RadarPLMBackbone(r=LORA_R, lora_alpha=LORA_ALPHA)

    print('\n组装 RadarPLMWithHeadForStage3...')
    plm_model = RadarPLMWithHeadForStage3(backbone)

    total = sum(p.numel() for p in plm_model.parameters())
    trainable = sum(p.numel() for p in plm_model.parameters() if p.requires_grad)
    print(f'  PLM 总参数量: {total:,}')
    print(f'  可训练参数量: {trainable:,} ({100*trainable/total:.2f}%)')

    # ============================================
    # 4. 训练
    # ============================================
    print(f'\n开始 PLM 偏好感知微调 (α={ALPHA}, epoch={EPOCHS})...')
    train_plm_model(
        model=plm_model,
        ref_model=ref_model,
        train_tokens=train_tokens,
        train_labels=train_labels,
        val_tokens=val_tokens,
        val_labels=val_labels,
        mean=mean,
        std=std,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        alpha=ALPHA,
        save_path='plm_model.pth',
        patience=PATIENCE
    )

    print(f'\n✅ PLM 微调完成，已保存至 plm_model.pth')


if __name__ == '__main__':
    main()
