"""
训练参考模型（RadarReferenceModel）
=====================================
参考模型只在验证集上训练（80% 训练，20% 验证），用于后续 PLM 微调的偏好对比基线。

数据流：
    features/val_tokens_raw.npy + features/val_labels.npy
    → RadarTokenDataset (归一化) → RadarReferenceModel → CrossEntropyLoss

前置条件：
    先运行 process_ipix.py 提取特征到 features/ 目录

用法：
    python train_reference.py

输出：
    - reference_model.pth : 训练好的参考模型权重
"""

import numpy as np
import torch
import os
import sys
from dataset import RadarTokenDataset
from trainer import train_reference_model
from models.reference_model import RadarReferenceModel


# ======================================
# 配置参数
# ======================================
FEATURES_DIR = 'features'
VAL_TOKENS_PATH = os.path.join(FEATURES_DIR, 'val_tokens_raw.npy')
VAL_LABELS_PATH = os.path.join(FEATURES_DIR, 'val_labels.npy')

# 参考模型超参数（按论文配置）
PATCH_SIZE = 48
EMBED_DIM = 32
NHEAD = 8
NUM_LAYERS = 2
DIM_FF = 128
MAX_TOKENS = 55

# 训练超参数
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-4
PATIENCE = 30
VAL_SPLIT = 0.2  # 从验证集中再分出 20% 作为参考模型的验证


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # ============================================
    # 步骤 1: 加载 features/ 中已提取好的特征
    # ============================================
    if not os.path.exists(VAL_TOKENS_PATH) or not os.path.exists(VAL_LABELS_PATH):
        print(f'错误: 未找到缓存特征文件！')
        print(f'请先运行 process_ipix.py 提取特征:')
        print(f'    python process_ipix.py')
        sys.exit(1)

    print(f'加载特征: {VAL_TOKENS_PATH}')
    val_tokens = np.load(VAL_TOKENS_PATH)   # (N, 55, 48) 未归一化
    val_labels = np.load(VAL_LABELS_PATH)    # (N,)
    print(f'验证集 tokens: {val_tokens.shape}, labels: {val_labels.shape}')

    # 打印统计
    n_target = int(val_labels.sum())
    n_clutter = len(val_labels) - n_target
    print(f'  目标: {n_target} ({100*n_target/len(val_labels):.1f}%)')
    print(f'  杂波: {n_clutter} ({100*n_clutter/len(val_labels):.1f}%)')

    # ============================================
    # 步骤 2: 初始化参考模型
    # ============================================
    print('\n初始化 RadarReferenceModel...')
    model = RadarReferenceModel(
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        max_tokens=MAX_TOKENS,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参考模型参数量: {total_params:,}')

    # ============================================
    # 步骤 3: 训练（内部 80/20 切分 + 归一化）
    # ============================================
    print('\n开始训练参考模型...')
    mean, std = train_reference_model(
        model=model,
        val_tokens=val_tokens,
        val_labels=val_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        save_path='reference_model.pth',
        patience=PATIENCE,
        val_split=VAL_SPLIT
    )

    print(f'\n✅ 参考模型训练完成，已保存至 reference_model.pth')


if __name__ == '__main__':
    main()
