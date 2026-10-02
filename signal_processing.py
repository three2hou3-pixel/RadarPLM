import numpy as np
from scipy.signal import stft
import torch
from torch.utils.data import Dataset

# ---------- 特征提取函数 ----------
def extract_radar_feature(x, n_fft=512):
    """
    针对雷达原始回波序列，提取论文所述的 5 钟多域序列特征

    参数：
    ----------
    x：输入的雷达复数信号向量（IQ信号），在 IPIX 数据中为 512 维，即 512 个采样点
    n_fft：FFT 的点数，默认与信号长度 N 一致
    
    输出：
    ----------
    feature_matrix：提取后的联合特征矩阵，形状为 (5, N)
    """
    N = len(x)
    assert N == n_fft, f"输入信号长度{N}必须与特征对齐长度{n_fft}一致"

    # 1. 瞬时相位（IP）
    feat_ip = np.angle(x)

    # 2. 多普勒频谱熵（DSE）
    X_fft = np.fft.fftshift(np.fft.fft(x))              # 标准傅里叶变换以及中心化移位
    F_fd = np.abs(X_fft)
    F_tilde = F_fd / (np.sum(F_fd) + 1e-12)             # 归一化
    feat_dse = -F_tilde * np.log(F_tilde + 1e-12)

    # 3. 短时傅里叶变换幅度谱（SMS）
    f, t, Zxx = stft(x, window='hamming', nperseg=64, noverlap=32, nfft=n_fft, return_onesided=False)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
    sum_t_magnitude = np.sum(np.abs(Zxx_shifted), axis=1)               # 时间维度累加
    feat_sms = 10 * np.log10(sum_t_magnitude)               # 转化为 dB 尺度

    # 4. 幅度（Amp）
    feat_amp = np.abs(x)

    # 5. 多普勒相位（DP）
    feat_dp = np.angle(X_fft)

    # 堆叠特征
    feature_matrix = np.stack([feat_ip, feat_dse, feat_sms, feat_amp, feat_dp])

    return feature_matrix



# ---------- 特征提取函数 ----------
def patch_radar_features(feature_matrix, patch_size=48):
    """
    将形状为 (5, N) 的联合特征矩阵进行分块并排列，作为模型输入的 Token 序列， 形状为 (K, L)

    参数：
    ----------
    feature_matrix：原始回波序列提取得到的特征矩阵
    patch_size：每个 Token 块的长度 L，论文中设定为 48
    """
    num_features, N = feature_matrix.shape

    # 1. 计算每一行切片后，会产生多少个小块，按论文中设定为 11
    num_patches_per_feature = int(np.ceil(N / patch_size))

    # 2. 对最后一个小块补零
    padded_length = num_patches_per_feature * patch_size
    pad_width = padded_length - N
    if pad_width > 0:
        feature_matrix_padded = np.pad(feature_matrix, ((0, 0), (0, pad_width)),mode='constant', constant_values=0)
    else:
        feature_matrix_padded = feature_matrix

    # 3. 切块并且重新排列
    reshaped_tensor = feature_matrix_padded.reshape(num_features, num_patches_per_feature, patch_size)
    K = num_features * num_patches_per_feature
    patched_tokens = reshaped_tensor.reshape(K, patch_size)

    return patched_tokens



# ----------- 批量离线预计算 ----------
def precompute_all_features(raw_signals, patch_size=48, save_path=None):
    """
    提前批量提取所有雷达特征，一次性完成所有计算
    Args:
        raw_signals: (N, 512) 复数IQ信号
        patch_size: 分块长度
        save_path: 保存路径，如 'radar_features.npy'
    Returns:
        all_tokens: (N, 55, 48) 所有样本的token特征
    """
    all_tokens = []
    for i in range(len(raw_signals)):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(raw_signals)}")
        feature_matrix = extract_radar_feature(raw_signals[i], n_fft=512)
        tokens = patch_radar_features(feature_matrix, patch_size=patch_size)
        all_tokens.append(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.float32)
    
    if save_path is not None:
        np.save(save_path, all_tokens)
        print(f"Features saved to {save_path}")
    
    return all_tokens