import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from dataset import RadarTokenDataset
from losses import PreferenceAwareLoss
from losses import UncertaintyJointLoss
# ---------- 训练循环驱动模块 ----------

# =========================================

# 1. 参考模型训练器
def train_reference_model(model, val_tokens, val_labels, epochs=300, batch_size=64, lr=1e-4,
                          save_path='reference_model.pth', patience=30, val_split=0.2):
    """
    RadarPLM 参考模型训练器：仅在 15% 的原始验证集数据上训练
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备：{device} 启动参考模型训练...')
    model = model.to(device)

    # 1. 数据集划分：将原始验证集再分为训练集和验证集（监控过拟合）
    #    注意：必须先打乱再切分！IPIXRawDataset 按距离单元顺序排列，
    #    顺序切分会导致前 80% 包含几乎所有目标样本，后 20% 几乎全是杂波，
    #    造成验证准确率虚高。
    total_size = len(val_tokens)
    train_size = int(total_size * (1 - val_split))
    val_size = total_size - train_size

    indices = np.random.permutation(total_size)
    val_tokens_shuffled = val_tokens[indices]
    val_labels_shuffled = val_labels[indices]

    train_tokens, val_tokens_split = np.split(val_tokens_shuffled, [train_size])
    train_labels, val_labels_split = np.split(val_labels_shuffled, [train_size])
    print(f'参考模型训练集大小：{train_size}，验证集大小：{val_size}')

    # 2. 按特征组归一化 Dataset
    train_dataset = RadarTokenDataset(train_tokens, train_labels, normalize=True)
    val_dataset = RadarTokenDataset(val_tokens_split, val_labels_split, normalize=True, mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    # 3. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # 4. 训练状态初始化
    best_val_loss = float('inf')
    patience_counter = 0
    print('============ 开始迭代训练 ==========')

    for epoch in range(1, epochs + 1):

        # 训练阶段
        model.train()
        train_epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            logits = model(batch_tokens)
            loss = criterion(logits.reshape(-1, 2), batch_labels.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_epoch_loss += loss.item() * batch_tokens.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.numel()

        train_avg_loss = train_epoch_loss / train_size
        train_acc = (train_correct / train_total) * 100

        # 验证阶段
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_tokens, batch_labels in val_loader:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_tokens)
                loss = criterion(logits.reshape(-1, 2), batch_labels.reshape(-1))
                
                val_epoch_loss += loss.item() * batch_tokens.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.numel()
        
        val_avg_loss = val_epoch_loss / val_size
        val_acc = (val_correct / val_total) * 100

        # 日志打印与模型保存
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:>3}/{epochs}] | "
                f"Train Loss: {train_avg_loss:.5f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_avg_loss:.5f} | Val Acc: {val_acc:.2f}%"
            )

        # 早停与最优模型保存
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            # 保存最优模型权重
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ 早停触发：连续{patience}个 epoch 验证损失未下降")
                break

    print("==================================================")
    print(f"🎉 参考模型训练圆满完成！最优验证Loss: {best_val_loss:.5f}")
    print(f"💾 最优模型参数已保存至: '{save_path}'")
    print("==================================================\n")

    return (
        train_dataset.mean.cpu().numpy(),  # type:ignore
        train_dataset.std.cpu().numpy()     # type:ignore
    )


# 2. 骨干 PLM 大模型微调训练器
def train_plm_model(model, ref_model, train_tokens, train_labels, val_tokens, val_labels, mean, std,
                    epochs=500, batch_size=64, lr=1e-4, alpha=0.9, save_path='plm_model.pth', patience=30):
    """
    GPT-2 微调：仅训练 LoRA、LayerNorm 和 雷达嵌入层
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备：{device} 启动基于偏好感知损失的大模型微调...')

    model = model.to(device)
    ref_model = ref_model.to(device)
    
    # 将参考模型完全冻结（作为考官）
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # 数据集与 DataLoader（使用按特征组归一化）
    train_dataset = RadarTokenDataset(
        train_tokens, train_labels, normalize=True,
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32)
        )
    
    val_dataset = RadarTokenDataset(
        val_tokens, val_labels, normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    # 损失函数
    criterion = PreferenceAwareLoss(alpha=alpha)
    
    # 优化器参数分组：不同组件使用不同学习率
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'lora_' in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'ln_' in n or 'layer_norm' in n], 'lr': lr * 5},
        {"params": [p for n, p in model.named_parameters() if "radar_embedding" in n and "layer_norm" not in n], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if n.startswith("layer")], "lr": lr}
        ]

    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 训练状态初始化
    best_val_loss = float('inf')
    patience_counter = 0
    print(f'-> 可训练总参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print('=============== 大模型微调开始 ===============')

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_epoch_loss = 0.0
        total_activated_tokens = 0
        total_tokens = 0

        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # 参考模型打分
            with torch.no_grad():
                ref_logits = ref_model(batch_tokens)

            # 主模型预测
            plm_logits, _ = model(batch_tokens)

            # 计算损失
            loss, learning_value = criterion(plm_logits, ref_logits, batch_labels)

            # 反向传播与梯度裁剪
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计指标
            train_epoch_loss += loss.item() * batch_tokens.size(0)
            total_activated_tokens += (learning_value > 0).sum().item()
            total_tokens += batch_labels.numel()

        train_avg_loss = train_epoch_loss / len(train_dataset)
        activation_rate = (total_activated_tokens / total_tokens) * 100

        # 验证阶段
        model.eval()
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for batch_tokens, batch_labels in val_loader:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)

                ref_logits = ref_model(batch_tokens)
                plm_logits, _ = model(batch_tokens)
                loss, _ = criterion(plm_logits, ref_logits, batch_labels)

                val_epoch_loss += loss.item() * batch_tokens.size(0)
        
        val_avg_loss = val_epoch_loss / len(val_dataset)

        # 打印日志与模型保存
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:>3}/{epochs}] | "
                f"Train Loss: {train_avg_loss:.5f} | "
                f"Val Loss: {val_avg_loss:.5f} | "
                f"Activation Rate: {activation_rate:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            
            # 只保存可训练参数，GPT-2 的冻结权重不用保存
            trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
            lora_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_names}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'feature_means': mean,
                'feature_stds': std
            }, save_path)

            print(f'-> 保存最优模型，验证损失：{best_val_loss:.5f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ 早停触发：连续{patience}个Epoch验证损失未下降")
                break

        # 学习率衰减
        scheduler.step()

    print("==================================================")
    print(f"🎉 大模型偏好微调圆满结束！最优验证损失: {best_val_loss:.5f}")
    print(f"💾 轻量化增量权重已保存至: '{save_path}' (大小约200KB)")
    print("==================================================\n")
    
    return best_val_loss



# 3. 自编码器二分类头训练器
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def train_autoencoder_head(
    backbone: nn.Module,
    ae_head: nn.Module,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int = 300,              # 严格对齐论文 Table II 标准 
    batch_size: int = 64,           # 严格对齐论文 Table II 标准 
    eval_batch_size: int = 400,     # 严格对齐论文 Table II 标准 
    lr: float = 1e-5,               # 严格对齐论文 Table II 标准 
    save_path: str = 'autoencoder_head.pth',
    patience: int = 30
) -> float:
    """
    Stage 4 训练器：完全冻结大模型主干，联合训练1x1自编码决策头与任务不确定性标量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device} 启动 Stage 4 自编码器分类头终极重训...")
    
    # ─────────────────────────────────────────────────────────
    # 将 Stage 3 训练好的大模型主干彻底锁死
    # ─────────────────────────────────────────────────────────
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
        
    ae_head = ae_head.to(device)
    
    # 1. 载入 Dataset，并继承 Stage 2 标准化尺寸
    #    Stage 4 使用 sample_level=True: 论文公式 (22) 要求样本级标签, 而非 token 级
    train_dataset = RadarTokenDataset(
        train_tokens, train_labels, normalize=True, sample_level=True,
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32)
    )
    val_dataset = RadarTokenDataset(
        val_tokens, val_labels, normalize=True, sample_level=True,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    # 💡 严格对齐 Table II：训练用 64 批次，验证和盲测强制切换为 400 宽批次 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    # 2. 实例化带实时看盘监控的自适应损失函数
    criterion = UncertaintyJointLoss().to(device)
    
    # 3. 参数合并：自编码器网络参数 + 损失函数内部的自适应变量 
    all_trainable_params = list(ae_head.parameters()) + list(criterion.parameters())
    
    # 定义 Adam 优化器 
    optimizer = optim.Adam(all_trainable_params, lr=lr, betas=(0.9, 0.999))
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"-> 大模型主干成功冻结。当前阶段实际可训练参数量: {sum(p.numel() for p in all_trainable_params if p.requires_grad):,} 个")
    print("================== 自编码器联合重训开始 ==================")
    
    for epoch in range(1, epochs + 1):
        # -------------------------------------------------
        # 训练阶段 (联调大模型只读特征流)
        # -------------------------------------------------
        ae_head.train()
        train_epoch_loss = 0.0
        
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # Step A: 让数据流过大模型，无梯度拦截倒数第二层高维特征流 F_LN
            with torch.no_grad():
                F_LN = backbone(batch_tokens)  # 形状: (B, 55, 768)
                
            # Step B: 喂入自编码器决策头，并行输出判定概率与重构矩阵
            logits, F_LN_recon = ae_head(F_LN)  # logits: (B, 2), F_LN_recon: (B, 55, 768)
            
            # 自编码器吐出样本级二分类 (B, 2), Dataset 已通过 sample_level=True 返回标量标签 (B,)
            sample_labels = batch_labels
            
            # Step C: 自适应联合损失计算
            loss, metrics = criterion(logits, F_LN_recon, sample_labels, F_LN)
            
            # Step D: 反向传播共同刷新网络权重与不确定性分母 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item() * batch_tokens.size(0)
            
        train_avg_loss = train_epoch_loss / len(train_dataset)
        
        # -------------------------------------------------
        # 验证阶段
        # -------------------------------------------------
        ae_head.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_tokens, batch_labels in val_loader:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)
                
                # 双模型全只读前向传播
                F_LN = backbone(batch_tokens)
                logits, F_LN_recon = ae_head(F_LN)
                sample_labels = batch_labels
                
                loss, val_metrics = criterion(logits, F_LN_recon, sample_labels, F_LN)
                val_epoch_loss += loss.item() * batch_tokens.size(0)
                
                # 统计最终样本级别的雷达目标检测准确率 (Accuracy)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == sample_labels).sum().item()
                val_total += sample_labels.numel()
                
        val_avg_loss = val_epoch_loss / len(val_dataset)
        val_acc = (val_correct / val_total) * 100
        
        # -------------------------------------------------
        # 打印日志（基于验证集）
        # -------------------------------------------------
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:>3}/{epochs}] | "
                f"Train Joint Loss: {train_avg_loss:.4f} | "
                f"Val Joint Loss: {val_avg_loss:.4f} | "
                f"Sample-level Val Acc: {val_acc:.2f}%"
            )
            print(
                f"  [重构轨] MSE: {val_metrics['loss_recon_pure']:.5f} -> 实际权重: {val_metrics['weight_recon']:.4f} (σ_recon: {val_metrics['sigma_recon']:.3f})\n"
                f"  [分类轨] CE: {val_metrics['loss_ce_pure']:.5f}  -> 实际权重: {val_metrics['weight_ce']:.4f} (σ_ce: {val_metrics['sigma_ce']:.3f})"
            )
            print("-" * 80)
            
        # 早停与模型持久化存档
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            
            # 建立稳健的 Stage 4 最终 Checkpoint 存档
            torch.save({
                'epoch': epoch,
                'ae_head_state_dict': ae_head.state_dict(),
                'loss_vars_state_dict': criterion.state_dict(),
                'best_val_loss': best_val_loss,
                'val_acc_sample_level': val_acc,
                'mean': mean,
                'std': std
            }, save_path)
            print(f"-> 成功捕获更优多任务交汇点，最优联合验证损失: {best_val_loss:.4f}，样本检测率: {val_acc:.2f}%\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ 早停触发：连续 {patience} 个 Epoch 联合验证损失未下降，终止训练。")
                break
                
    print("==================================================")
    print(f"🎉 RadarPLM 终极自编码决策头重训圆满成功！")
    print(f"💾 最终部署级核心参数已固化至: '{save_path}'")
    print("==================================================\n")
    
    return best_val_loss