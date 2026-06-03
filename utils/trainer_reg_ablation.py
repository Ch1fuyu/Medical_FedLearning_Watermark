"""
正则项消融实验训练器
支持选择性启用新的水印稳定性正则项:
- 漂移惩罚
- 裕量惩罚
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.light_autoencoder import LightAutoencoder
from models.losses.multi_loss import MultiLoss, FocalLoss
from utils.key_matrix_utils import KeyMatrixManager
from utils.trainer_private_enhanced import TrainerPrivateEnhanced, TesterPrivate, accuracy


class TrainerRegAblation(TrainerPrivateEnhanced):
    """正则项消融实验训练器，仅使用新的水印稳定性正则项"""
    
    def __init__(self, model, device, dp, sigma, args=None):
        super().__init__(model, device, dp, sigma, args)
        
        self.use_drift_reg = getattr(args, 'use_drift_reg', True) if args else True
        self.use_margin_reg = getattr(args, 'use_margin_reg', True) if args else True

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """本地更新，仅使用新的水印稳定性正则项"""
        self.model.to(self.device)
        self.model.train()

        if self.mask_manager:
            try:
                self.mask_manager.update_encoder_mask(client_id)
            except Exception as e:
                print(f"⚠️ 客户端{client_id}掩码更新失败: {e}")
        
        # 如果启用学习率调度器，根据全局轮次计算当前学习率
        if self.args and getattr(self.args, 'use_lr_scheduler', False):
            import math
            progress = current_epoch / total_epochs
            lr_min = lr * 0.01
            adjusted_lr = lr_min + (lr - lr_min) * (1 + math.cos(math.pi * progress)) / 2
        else:
            adjusted_lr = lr
        
        # 选择优化器（仅使用模型参数，不再需要 _learnable_ratio）
        if self.args and hasattr(self.args, 'optim'):
            if self.args.optim.lower() == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=adjusted_lr, 
                    weight_decay=getattr(self.args, 'wd', 0.0)
                )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), 
                    lr=adjusted_lr, 
                    momentum=0.9, 
                    weight_decay=getattr(self.args, 'wd', 0.0)
                )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=adjusted_lr, 
                momentum=0.9
            )

        epoch_loss, epoch_acc = [], []

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                pred = self.model(x)
                main_loss = self.get_loss_function(pred, y)

                # 计算最终损失并执行反向传播（方案A核心）
                if current_epoch == 0:
                    # 第一轮只使用主任务损失
                    total_loss = main_loss
                    total_loss.backward()
                else:
                    # 从args读取alpha参数
                    alpha_early = None
                    alpha_late = None
                    if self.args and hasattr(self.args, 'multiloss_alpha_early'):
                        alpha_early = self.args.multiloss_alpha_early
                    if self.args and hasattr(self.args, 'multiloss_alpha_late'):
                        alpha_late = self.args.multiloss_alpha_late
                    
                    # 获取mask
                    if self.mask_manager:
                        target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                    else:
                        target_mask = encoder_mask = effective_mask = None
                    
                    # 调用新的方法：在 backward 过程中更新统计量
                    # 注意：main_loss.backward() 在 compute_loss_and_backward 内部调用
                    total_loss = self.multi_loss.compute_loss_and_backward(
                        main_loss,
                        target_mask, encoder_mask, effective_mask,
                        current_epoch, total_epochs,
                        alpha_early, alpha_late
                    )

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)
                
                # 内存清理
                if batch_idx % 3 == 0:
                    torch.cuda.empty_cache()

            loss_meter /= run_count
            acc_meter /= run_count
            epoch_loss.append(loss_meter)
            torch.cuda.empty_cache()
            epoch_acc.append(acc_meter)

            if epoch + 1 == local_ep:
                from tqdm import tqdm
                tqdm.write(f"C{client_id} E{epoch+1}/{local_ep}: L={loss_meter:.4f} A={acc_meter:.4f} LR={adjusted_lr:.6f}")

        # 每5个epoch进行水印融合
        if (current_epoch + 1) % 5 == 0:
            self._embed_watermark(client_id, current_epoch)

        if current_epoch > 0 and current_epoch % 10 == 0:
            self._cleanup_memory()

        # 差分隐私噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

