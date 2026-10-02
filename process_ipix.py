"""
IPIX 雷达数据全流程特征提取脚本
=================================
数据流：.cdf 文件 → 复数回波切片 → 5域多域特征 → (55, 48) Token序列 → 归一化 → 保存

用法：
    python process_ipix.py

输出：
    - ipix_features.npy      : (总样本数, 55, 48) 的 Token 特征矩阵
    - ipix_labels.npy        : (总样本数,) 的标签 (1=目标, 0=杂波)
    - ipix_mean.npy          : (1, 55, 48) 归一化均值（供后续验证/测试集复用）
    - ipix_std.npy           : (1, 55, 48) 归一化标准差
    - 训练/验证/测试各自单独保存一份
"""

import numpy as np
import torch
import os
from dataset import IPIXRawDataset, RadarTokenDataset
from signal_processing import precompute_all_features

# ======================================
# 0. 配置参数（按需修改）
# ======================================
CDF_PATH = 'dataset/19931107_135603_starea.cdf'   # IPIX 原始文件路径
OUTPUT_DIR = 'features'                             # 特征保存目录
WINDOW_LEN = 512                                    # 滑窗长度（必须=512）
PATCH_SIZE = 48                                     # Token 分块大小
TARGET_CELL = 9                                     # 目标所在距离单元（1-14）
POL_TYPE = 'like'                                   # like=HH极化, cross=HV极化
SPLITS = ['train', 'val', 'test']                   # 三个划分阶段


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================
    # 1. 读取原始 .cdf → 复数回波切片 (总体)
    # ============================================
    print("=" * 55)
    print("步骤 1: 读取 IPIX 原始数据并滑窗切片")
    print("=" * 55)

    # 用 'train' split 先拿到训练部分数据
    raw_dataset = {}
    for split in SPLITS:
        raw_dataset[split] = IPIXRawDataset(
            cdf_path=CDF_PATH,
            split=split,
            target_cell=TARGET_CELL,
            window_len=WINDOW_LEN,
            polarization_type=POL_TYPE
        )
        print(f"  {split}: {len(raw_dataset[split])} 个样本")

    # 也可一次性读取全量（不切分），如果需要更大训练集
    total_samples = sum(len(ds) for ds in raw_dataset.values())
    print(f"\n  => 总计 {total_samples} 个复数回波样本")

    # ============================================
    # 2. 批量提取 5 域多域特征 → Token 序列
    # ============================================
    print("\n" + "=" * 55)
    print("步骤 2: 批量提取 5 域特征并分块为 Token 序列")
    print("=" * 55)

    all_tokens = {}  # {split: (N_split, 55, 48)}
    all_labels = {}

    for split in SPLITS:
        print(f"\n--- 正在处理 {split} 集 ---")
        ds = raw_dataset[split]

        tokens = precompute_all_features(
            ds.samples,         # (N, 512) 复数 IQ 信号
            patch_size=PATCH_SIZE,
            save_path=None      # 这里先不存，后面一起存
        )
        # tokens: (N, 55, 48)

        all_tokens[split] = tokens
        all_labels[split] = ds.labels

        print(f"  {split} — Token 形状: {tokens.shape}, "
              f"目标: {ds.labels.sum()}, 杂波: {len(ds.labels) - ds.labels.sum()}")

    # ============================================
    # 3. 拼接训练集做 4D 归一化（验证/测试沿用训练集的 mean/std）
    # ============================================
    print("\n" + "=" * 55)
    print("步骤 3: 4D 通道级归一化（训练集计算 mean/std）")
    print("=" * 55)

    # 只拿训练集计算归一化参数
    train_tokens = all_tokens['train']
    train_labels = all_labels['train']

    # 转为张量做 4D 归一化（复用 RadarTokenDataset 的逻辑）
    train_dataset = RadarTokenDataset(
        tokens=train_tokens,
        labels=train_labels,
        normalize=True  # 自动计算 mean/std
    )

    mean_3d = train_dataset.mean  # (1, 55, 48)
    std_3d = train_dataset.std    # (1, 55, 48)
    print(f"  归一化 mean 形状: {mean_3d.shape}")
    print(f"  归一化 std  形状: {std_3d.shape}")

    # 对验证集和测试集使用训练集的 mean/std
    norm_datasets = {'train': train_dataset}
    for split in ['val', 'test']:
        norm_datasets[split] = RadarTokenDataset(
            tokens=all_tokens[split],
            labels=all_labels[split],
            normalize=True,
            mean=mean_3d,
            std=std_3d
        )

    # ============================================
    # 4. 保存特征到磁盘
    # ============================================
    print("\n" + "=" * 55)
    print("步骤 4: 保存特征到磁盘")
    print("=" * 55)

    # 保存归一化参数
    np.save(os.path.join(OUTPUT_DIR, 'mean.npy'), mean_3d.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, 'std.npy'), std_3d.cpu().numpy())
    print(f"  ✓ 归一化参数保存至 {OUTPUT_DIR}/mean.npy, std.npy")

    # 分别保存每个 split 的归一化后特征
    for split in SPLITS:
        ds = norm_datasets[split]
        # 把归一化后的 tokens 转回 numpy
        tokens_norm = ds.tokens.cpu().numpy()  # (N, 55, 48)
        labels_np = ds.labels.cpu().numpy()     # (N,)

        np.save(os.path.join(OUTPUT_DIR, f'{split}_tokens.npy'), tokens_norm)
        np.save(os.path.join(OUTPUT_DIR, f'{split}_labels.npy'), labels_np)
        print(f"  ✓ {split}: tokens {tokens_norm.shape}, labels {labels_np.shape}")

    # 也保存一份未归一化的原始特征（备查）
    for split in SPLITS:
        np.save(os.path.join(OUTPUT_DIR, f'{split}_tokens_raw.npy'), all_tokens[split])

    print(f"\n  所有特征已保存至 {OUTPUT_DIR}/")

    # ============================================
    # 5. 统计信息汇总
    # ============================================
    print("\n" + "=" * 55)
    print("数据统计汇总")
    print("=" * 55)
    for split in SPLITS:
        tok = all_tokens[split]
        lab = all_labels[split]
        n_target = int(lab.sum())
        n_clutter = len(lab) - n_target
        print(f"  {split:>5}: {len(tok):>5} 样本 | "
              f"目标 {n_target:>4} ({100*n_target/len(lab):.1f}%) | "
              f"杂波 {n_clutter:>4} ({100*n_clutter/len(lab):.1f}%) | "
              f"Token 形状 {tuple(tok.shape)}")


def quick_load_example():
    """
    快速加载已保存特征的示例
    """
    print("\n" + "=" * 55)
    print("[示例] 加载已保存的特征用于训练")
    print("=" * 55)

    # 加载训练集
    train_tokens = np.load(os.path.join(OUTPUT_DIR, 'train_tokens.npy'))
    train_labels = np.load(os.path.join(OUTPUT_DIR, 'train_labels.npy'))
    mean = np.load(os.path.join(OUTPUT_DIR, 'mean.npy'))
    std = np.load(os.path.join(OUTPUT_DIR, 'std.npy'))

    print(f"  训练集 tokens: {train_tokens.shape}")
    print(f"  训练集 labels: {train_labels.shape}")

    # 直接封装成 PyTorch Dataset
    train_dataset = RadarTokenDataset(
        tokens=train_tokens,
        labels=train_labels,
        normalize=False,       # 已经归一化过了
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32)
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(f"  DataLoader 就绪，每个 batch 的形状:")
    for batch_tokens, batch_labels in train_loader:
        print(f"    tokens: {batch_tokens.shape}, labels: {batch_labels.shape}")
        break

    return train_dataset


if __name__ == '__main__':
    main()

    # 如果你想看一下如何加载使用，取消下面这行的注释：
    # train_dataset = quick_load_example()

    print("\n✅ 全部完成！")
