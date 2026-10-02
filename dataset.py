import numpy as np
import torch
import netCDF4 as nc
from torch.utils.data import Dataset



# ---------- IPIX 原始数据提取函数 ----------
class IPIXRawDataset(Dataset):
    """
    IPIX 雷达原始全相参复数回波数据集 (PyTorch 标准封装版)
    用于在特征提取前，对特定编号的海杂波/目标进行时空物理隔离切片。
    """
    def __init__(self, cdf_path, split='train', target_cell=9, window_len=512, polarization_type='like'):
        """
        参数：
        ----------
        cdf_path          : str, 本地 .cdf 文件的路径
        split             : str, 划分阶段，可选 'train' (训练), 'val' (验证), 'test' (测试)
        target_cell       : int, 目标所在的距离单元编号 (1-14)，论文中 #17 默认为 9
        window_len        : int, 观测窗口长度 N，论文默认为 512
        polarization_type : str, 极化方式，可选 'like' (HH极化) 或 'cross' (HV极化)
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test'], "split 参数必须在 ['train', 'val', 'test'] 中选择"
        assert 1 <= target_cell <= 14, "IPIX数据集的距离单元范围必须在 1 到 14 之间"
        
        self.window_len = window_len
        self.target_cell_idx = target_cell - 1  # 内部自动转换为 0 轴索引
        
        # 1. 打开 NetCDF 文件并解密 I/Q 索引标量
        dataset = nc.Dataset(cdf_path, mode='r')
        if polarization_type == 'like':
            i_idx = int(dataset.variables['adc_like_I'][:].item())
            q_idx = int(dataset.variables['adc_like_Q'][:].item())
        elif polarization_type == 'cross':
            i_idx = int(dataset.variables['adc_cross_I'][:].item())
            q_idx = int(dataset.variables['adc_cross_Q'][:].item())
        else:
            raise ValueError("polarization_type 必须是 'like' 或 'cross'")
            
        # 2. 载入 4D 核心数据立方体 [131072, 2, 14, 4]
        adc_matrix = dataset.variables['adc_data'][:]
        dataset.close()  # 及时释放磁盘句柄
        
        # 3. 跨维度切片与复数合体 -> 最终转置为标准形状 (14, 131072)
        tx_idx = 0  # H发射凝视
        I_ch = adc_matrix[:, tx_idx, :, i_idx]
        Q_ch = adc_matrix[:, tx_idx, :, q_idx]
        raw_echo = (I_ch + 1j * Q_ch).T  # 形状: (14, 131072), 类型: complex64
        
        # 4. 严格按照论文执行时间轴顺序切割（比例：20% / 15% / 65%）
        total_points = 131072
        split_train = int(total_points * 0.20)  # 26214
        split_val = int(total_points * 0.35)    # 45875
        
        if split == 'train':
            start_ptr, end_ptr = 0, split_train
        elif split == 'val':
            start_ptr, end_ptr = split_train, split_val
        else:  # 'test'
            start_ptr, end_ptr = split_val, total_points
            
        # 5. 按照纵向距离门依次遍历，横向滑动切片
        self.samples = []
        self.labels = []
        
        M_target = 32    # 正样本密集滑窗步长
        M_clutter = 128  # 负样本稀疏滑窗步长
        num_cells = 14
        
        for cell_idx in range(num_cells):
            cell_sequence = raw_echo[cell_idx, start_ptr:end_ptr]
            seq_len = len(cell_sequence)
            
            # 根据当前距离单元是否为目标单元，动态切换步长与标签
            if cell_idx == self.target_cell_idx:
                stride = M_target
                label = 1
            else:
                stride = M_clutter
                label = 0
                
            # 执行滑窗提取
            for i in range(0, seq_len - self.window_len + 1, stride):
                window = cell_sequence[i : i + self.window_len]
                self.samples.append(window)
                self.labels.append(label)
                
        # 最终转化为内存驻留矩阵
        self.samples = np.array(self.samples, dtype=np.complex64)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"📊 [Dataset 成功建立] 阶段: {split.upper()} | 窗口长度: {self.window_len} | 目标单元: {target_cell}")
        print(f"   样本总数: {len(self.labels)} | 目标(1)={sum(self.labels)}, 杂波(0)={len(self.labels)-sum(self.labels)}")

    def __len__(self):
        """返回当前划分下的样本总数"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        根据索引获取单个全相参复数回波样本
        """
        x_sample = self.samples[idx]  # 形状: (window_len,) 的复数 numpy 数组
        y_label = self.labels[idx]    # 标量标签
        
        # 💡 核心细节：转换为 PyTorch 认识的 complex64 张量，便于后续直接接入 DataLoader 流水线
        x_tensor = torch.tensor(x_sample, dtype=torch.complex64)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        
        return x_tensor, y_tensor



# ---------- 特征处理函数 ----------
class RadarTokenDataset(Dataset):
    """
    将离线处理好的特征数据转化为张量，并且执行 4D 通道级归一化
    """
    def __init__(self, tokens, labels, normalize=True, mean=None, std=None, sample_level=False):
        """
        sample_level: 如果为 True, __getitem__ 返回标量标签 (用于 Stage 4 样本级分类);
                      如果为 False (默认), 返回 (55,) 的 token 级复制标签 (用于 Stage 3 偏好损失).
        """
        # 读入预计算好的 3D 特征: (N, 55, 48)
        self.tokens = torch.tensor(tokens, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_level = sample_level
        
        if normalize:
            if mean is None or std is None:
                # 1. 临时弹回 4D 物理形态 (N, 5, 11, 48)
                tokens_4d = self.tokens.view(-1, 5, 11, 48)
                
                # 2. 跨样本、跨切片、跨点求均值与标准差
                # 算出来的 mean_4d 形状为极为纯净的 (1, 5, 1, 1)
                mean_4d = torch.mean(tokens_4d, dim=(0, 2, 3), keepdim=True)
                std_4d = torch.std(tokens_4d, dim=(0, 2, 3), keepdim=True) + 1e-8
                
                # 3. 4D 空间下执行归一化，精准保护 5 大物理域各自的整体尺度，同时保留时频动态起伏
                tokens_4d_norm = (tokens_4d - mean_4d) / std_4d
                
                # 4. 将 mean 和 std 压缩成 3D 形状保存，方便验证集/测试集后续直接对齐使用
                self.mean = mean_4d.expand(-1, -1, 11, 48).contiguous().view(1, 55, 48)
                self.std = std_4d.expand(-1, -1, 11, 48).contiguous().view(1, 55, 48)
                
                # 5. 归一化后的数据恢复为 3D 骨干形态 (N, 55, 48) 分派给模型
                self.tokens = tokens_4d_norm.view(-1, 55, 48)
            else:
                self.mean = mean
                self.std = std
                self.tokens = (self.tokens - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.sample_level:
            return self.tokens[idx], self.labels[idx]  # 标量标签 (Stage 4 样本级分类)
        return self.tokens[idx], torch.full((55,), self.labels[idx], dtype=torch.long) # type: ignore