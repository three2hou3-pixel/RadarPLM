# RadarPLM：基于预训练语言模型的雷达海杂波抑制

用**预训练语言模型（GPT-2）**做雷达目标检测 / 海杂波抑制：把 IPIX 雷达回波的多域时频特征切成 Token 序列，当作「句子」喂给裁剪后的 GPT-2，用**参考模型当考官**做偏好感知微调，最后接自编码决策头输出逐 Token 的二分类结果。

> 论文复现 / 课程项目性质。

## 方法概述

### 1. 特征提取：从复数回波到 Token 序列

`signal_processing.py` 对每段 512 点的复数 IQ 信号提取论文所述的 **5 域序列特征**：

| 特征 | 含义 |
|---|---|
| IP | 瞬时相位（Instantaneous Phase） |
| DSE | 多普勒频谱熵（Doppler Spectral Entropy） |
| SMS | 短时傅里叶幅度谱（STFT Magnitude Spectrum） |
| Amp | 幅度 |
| DP | 多普勒相位（Doppler Phase） |

5 域特征拼接后按 `patch_size=48` 分块，得到 `(55, 48)` 的 Token 序列——**55 个 Token，每个 48 维**，对应 GPT-2 的「词序列」。

### 2. 四阶段训练流水线

| 阶段 | 脚本 | 做什么 | 产出 |
|---|---|---|---|
| 1 | `process_ipix.py` | `.cdf` → 5 域特征 → Token → 通道级 4D 归一化 | `features/*.npy` |
| 2 | `train_reference.py` | 在 val 划分上训练小参考模型，作为偏好对比基线 | `reference_model.pth` |
| 3 | `train_plm.py` | 微调裁剪版 GPT-2 骨干（LoRA + 偏好感知损失） | `plm_model.pth` |
| 4 | `train_autoencoder.py` | 冻结骨干，联合训练自编码决策头与任务不确定性标量 | `autoencoder_head.pth` |

### 3. 模型设计要点

- **骨干裁剪**：载入 HuggingFace 官方 GPT-2 预训练权重，物理裁掉后 8 层只保留**前 4 层**（`RadarPLMBackbone`）。
- **LoRA 注入**：在每层 Attention 的 `c_attn` 上只对 **Q 和 V 分支**挂 LoRA（`r=8, alpha=16`），K 分支保持冻结。
- **可训练参数**：仅 LoRA 矩阵、LayerNorm、雷达嵌入层与分类头参与训练，其余全部冻结。
- **偏好感知损失**（`losses.py`）：以参考模型为「考官」，只有当主模型损失 > `α × 参考模型损失`（`α=0.9`）的位置才激活梯度——即只在「做得比考官差」的地方学习。
- **数据划分**：严格按时间轴顺序切分 **20% / 15% / 65%**（train / val / test）。

## 项目结构

```
RadarPLM/
├── dataset.py               # IPIXRawDataset（.cdf 读取 + 滑窗切片）、RadarTokenDataset（4D 归一化）
├── signal_processing.py     # 5 域特征提取
├── process_ipix.py          # 阶段 1：特征提取主脚本
├── train_reference.py       # 阶段 2：训练参考模型
├── train_plm.py             # 阶段 3：PLM 偏好感知微调
├── train_autoencoder.py     # 阶段 4：自编码决策头
├── trainer.py               # 各阶段训练循环
├── losses.py                # PreferenceAwareLoss / UncertaintyJointLoss
├── models/
│   ├── plm_model.py         # 裁剪版 GPT-2 骨干 + LoRA + 嵌入层
│   ├── reference_model.py   # 参考模型
│   └── autoencoder_head.py  # 自编码决策头
├── dataset/                 # IPIX 原始 .cdf 数据
├── features/                # 中间特征缓存（已 gitignore，见下）
└── *.pth                    # 训练好的权重
```

## 快速开始

```bash
pip install torch transformers numpy scipy netCDF4

# 阶段 1：提取特征（生成 features/，约 347MB，耗时较久）
python process_ipix.py

# 阶段 2 → 3 → 4 依次执行
python train_reference.py
python train_plm.py
python train_autoencoder.py
```

> **注意**：`features/` 目录下的 `.npy` 中间缓存**未纳入版本库**（总体积约 347MB，且单个文件超过 GitHub 100MB 上限）。克隆仓库后请先运行 `python process_ipix.py` 重新生成。原始 `.cdf` 数据已随仓库提供，可完整复现。

## 数据集

使用 **IPIX**（Intelligent Pixel Processing）全相参 X 波段雷达实测数据 `19931107_135603_starea.cdf`，取自 Dartmouth 的 IPIX 公开数据集。默认目标位于**第 9 号距离单元**，采用 **HH（like）极化**。

> IPIX 数据集有其自身的使用条款，请遵守原始数据提供方的引用与使用要求。
