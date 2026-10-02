# 修复参考模型训练流程

## Context

当前有两个问题：

1. **`train_reference.py` 没有直接复用 `features/` 中已缓存的数据** — 虽然代码有缓存检查逻辑，但路径不够清晰，用户不好理解数据流。

2. **`trainer.py` 第 30 行的 `np.split` 是顺序切分未打乱** — 因为 IPIXRawDataset 按距离单元顺序排列（Cell 0→13），前 80% 包含几乎所有目标样本，后 20% 几乎全是纯杂波，导致验证准确率虚高（杂波占 ~90%，无脑预测杂波就有 90% acc）。

## 改动

### 1. `train_reference.py` — 简化，直接加载 features/ 缓存

去掉 `.cdf` → 特征提取的路径，直接假设用户在运行此脚本之前已用 `process_ipix.py` 生成好了 `features/` 目录。

直接加载：
- `features/val_tokens_raw.npy` — 未归一化的 val 特征
- `features/val_labels.npy` — val 标签

如果文件不存在，打印清晰的指引让用户先跑 `process_ipix.py`。

### 2. `trainer.py:25-31` — 切分前打乱

```python
# 改前
train_tokens, val_tokens_split = np.split(val_tokens, [train_size])

# 改后
indices = np.random.permutation(total_size)
val_tokens_shuffled = val_tokens[indices]
val_labels_shuffled = val_labels[indices]
train_tokens, val_tokens_split = np.split(val_tokens_shuffled, [train_size])
train_labels, val_labels_split = np.split(val_labels_shuffled, [train_size])
```

保证训练集和验证集都包含 ~29% 的目标样本，val acc 才有参考价值。

## Verification

```bash
python train_reference.py
```

检查输出：
- "加载缓存特征: features/val_tokens_raw.npy" — 确认使用了缓存
- Train Acc 和 Val Acc 差距在 5% 以内（不再虚高）
- 训练集和验证集的目标比例接近（~29%）
