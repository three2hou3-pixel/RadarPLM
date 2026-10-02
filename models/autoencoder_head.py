import torch
import torch.nn as nn

class RadarAutoencoderHead(nn.Module):
    def __init__(self, seq_len=55, embed_dim=768, latent_dim=64, num_classes=2):
        """
        1x1 卷积自编码决策头（为了对齐论文可训练参数量）
        """
        super(RadarAutoencoderHead, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # 1. 编码器 (Encoder)：基于 1x1 卷积的通道压缩 + 全连接瓶颈
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.flatten_dim = 64 * seq_len  # 64 * 55 = 3520
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)   # 压入 64 维 Latent 空间
        )
        
        # 2. 解码器 (Decoder)：全连接对称膨胀 + 1x1 转置卷积逆重构
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flatten_dim),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, embed_dim, kernel_size=1, stride=1, padding=0) # 完美恢复 768 维
        )
        
        # 3. 最终分类器 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)   # 输出 (B, 2)
        )
        
    def forward(self, F_LN):
        # (B, 55, 768) -> (B, 768, 55)
        x = F_LN.transpose(1, 2)
        
        # Encoder 压缩
        x_conv = self.encoder_conv(x)
        x_flat = x_conv.reshape(-1, self.flatten_dim)
        z = self.encoder_fc(x_flat)
        
        # 分类决策
        logits = self.classifier(z)
        
        # Decoder 重构
        dec_df = self.decoder_fc(z)
        dec_conv_in = dec_df.view(-1, 64, self.seq_len)
        F_LN_recon_trans = self.decoder_conv(dec_conv_in)
        
        # 弹回大模型原生格式: (B, 768, 55) -> (B, 55, 768)
        F_LN_recon = F_LN_recon_trans.transpose(1, 2)
        
        return logits, F_LN_recon