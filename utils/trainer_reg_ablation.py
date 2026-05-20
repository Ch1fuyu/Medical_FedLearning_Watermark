"""
正则项消融实验训练器
支持选择性启用三个正则项：
- reg_term1: 梯度平衡正则项
- reg_term2: 方差比例正则项  
- reg_term3: 自适应权重正则项
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
    """正则项消融实验训练器，支持选择性启用正则项"""
    
    def __init__(self, model, device, dp, sigma, args=None):
        super().__init__(model, device, dp, sigma, args)
        
        # 从args读取正则项选择参数
        self.use_reg1 = getattr(args, 'use_reg1', True) if args else True
        self.use_reg2 = getattr(args, 'use_reg2', True) if args else True
        self.use_reg3 = getattr(args, 'use_reg3', True) if args else True
        
        # 记录正则项配置
        reg_config = []
        if self.use_reg1:
            reg_config.append('reg1')
        if self.use_reg2:
            reg_config.append('reg2')
        if self.use_reg3:
            reg_config.append('reg3')
        self.reg_config_str = '+'.join(reg_config) if reg_config else 'none'
        
        print(f"正则项消融实验配置: {self.reg_config_str}")
        print(f"  reg1 (梯度平衡): {'启用' if self.use_reg1 else '禁用'}")
        print(f"  reg2 (方差比例): {'启用' if self.use_reg2 else '禁用'}")
        print(f"  reg3 (自适应权重): {'启用' if self.use_reg3 else '禁用'}")

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """本地更新，支持选择性启用正则项"""
        self.model.to(self.device)  # 确保模型在正确的设备上
        self.model.train()
        
        # 如果启用学习率调度器，根据全局轮次计算当前学习率
        if self.args and getattr(self.args, 'use_lr_scheduler', False):
            import math
            # 余弦退火：基于全局训练轮次
            progress = current_epoch / total_epochs  # 0 到 1
            lr_min = lr * 0.01
            adjusted_lr = lr_min + (lr - lr_min) * (1 + math.cos(math.pi * progress)) / 2
        else:
            adjusted_lr = lr
        
        # 根据args.optim参数选择优化器（使用调整后的学习率）
        if self.args and hasattr(self.args, 'optim'):
            if self.args.optim.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=adjusted_lr, weight_decay=getattr(self.args, 'wd', 0.0))
            else:  # 默认使用SGD
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=adjusted_lr, momentum=0.9, weight_decay=getattr(self.args, 'wd', 0.0))
        else:
            # 向后兼容：如果没有args或optim参数，使用SGD
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=adjusted_lr, momentum=0.9)

        epoch_loss, epoch_acc = [], []
        
        # 初始化梯度统计变量
        total_gradients = None
        total_encoder_gradients = None
        total_target_mask = None
        total_encoder_mask = None
        total_effective_mask = None
        batch_count = 0
        is_first_batch = True  # 标记是否为第一个batch

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0
            
            # 每个 epoch 开始时，保存训练前的参数
            # 注意：这里不判断 is_first_epoch，确保每个 epoch 都有正确的基准
            if self.mask_manager and hasattr(self.multi_loss, 'save_params_before_training'):
                try:
                    wm_masks, nonwm_masks = self.mask_manager.get_param_masks_by_name()
                    self.multi_loss.save_params_before_training(
                        self.model.state_dict(),
                        wm_masks,
                        nonwm_masks
                    )
                except Exception as e:
                    pass

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(x)
                main_loss = self.get_loss_function(pred, y)

                # 计算最终损失（使用当前已有的统计量）
                if current_epoch == 0:
                    total_loss = main_loss
                else:
                    # 从args读取alpha参数
                    alpha_early = None
                    alpha_late = None
                    if self.args and hasattr(self.args, 'multiloss_alpha_early'):
                        alpha_early = self.args.multiloss_alpha_early
                    if self.args and hasattr(self.args, 'multiloss_alpha_late'):
                        alpha_late = self.args.multiloss_alpha_late
                    
                    # 获取当前梯度统计量用于调试
                    stats = self.multi_loss.get_stats()
                    
                    # 传入正则项选择参数
                    total_loss = self.multi_loss.compute_loss(
                        main_loss, current_epoch, total_epochs, 
                        alpha_early, alpha_late,
                        use_reg1=self.use_reg1,
                        use_reg2=self.use_reg2,
                        use_reg3=self.use_reg3
                    )
                    
                    # 每50个batch打印一次正则项统计（调试用）
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        print(f"[Debug] Epoch {current_epoch} Batch {batch_idx}: "
                              f"main_loss={main_loss.item():.6f}, "
                              f"prevGM={stats['prevGM']:.8f}, "
                              f"prevGH={stats['prevGH']:.8f}, "
                              f"prevRatio={stats['prevRatio']:.6f}")

                total_loss.backward()
                
                # 在clip和step之前，更新当前batch的梯度统计量
                # 这样下一个batch的正则项会使用当前batch的统计量
                if self.mask_manager and batch_idx % 5 == 0:
                    try:
                        conv_gradients = []
                        for name, param in self.model.named_parameters():
                            if 'conv' in name and 'weight' in name and param.grad is not None:
                                conv_gradients.append(param.grad.view(-1))
                        
                        if conv_gradients:
                            gradients = torch.cat(conv_gradients)
                            target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                            encoder_gradients = torch.mul(gradients, effective_mask)
                            
                            # 更新统计量为当前batch的梯度
                            self.multi_loss.update_gradient_stats(
                                gradients.detach(),
                                encoder_gradients.detach(),
                                target_mask,
                                encoder_mask,
                                effective_mask
                            )
                            
                            del gradients, encoder_gradients, target_mask, encoder_mask, effective_mask
                    except Exception as e:
                        pass  # 静默处理，避免干扰训练

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 梯度统计收集：训练开始前，先用前一轮的统计量
                # 第一个batch使用trainer的初始统计量（如果有）
                if self.mask_manager and batch_idx == 0 and hasattr(self, '_prev_stats') and self._prev_stats:
                    try:
                        self.multi_loss.prevGM = self._prev_stats['prevGM']
                        self.multi_loss.prevGH = self._prev_stats['prevGH']
                        self.multi_loss.prevRatio = self._prev_stats['prevRatio']
                    except:
                        pass

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)
                
                # 定期清理内存，防止内存泄漏
                if batch_idx % 3 == 0:  # 更频繁的内存清理
                    torch.cuda.empty_cache()
                
                # 每20个批次进行激进内存清理
                if batch_idx % 20 == 0 and batch_idx > 0:
                    self._aggressive_memory_cleanup()

            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            
            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            epoch_acc.append(acc_meter)

            # 简洁输出：只在最后epoch打印
            if epoch + 1 == local_ep:
                from tqdm import tqdm
                tqdm.write(f"C{client_id} E{epoch+1}/{local_ep}: L={loss_meter:.4f} A={acc_meter:.4f} LR={adjusted_lr:.6f}")

        # 本地训练结束后，保存梯度统计量供下一个客户端使用
        if self.mask_manager and batch_count > 0:
            try:
                # 计算平均梯度数据
                avg_gradients = total_gradients / batch_count
                avg_encoder_gradients = total_encoder_gradients / batch_count
                
                # 更新梯度统计量
                self.multi_loss.update_gradient_stats(
                    avg_gradients,
                    avg_encoder_gradients,
                    total_target_mask,
                    total_encoder_mask,
                    total_effective_mask
                )
                
                # 保存统计量供下一个客户端使用
                self._prev_stats = {
                    'prevGM': self.multi_loss.prevGM,
                    'prevGH': self.multi_loss.prevGH,
                    'prevRatio': self.multi_loss.prevRatio
                }
                
            except Exception as e:
                print(f"⚠️ C{client_id} 梯度统计更新失败: {e}")
            finally:
                # 确保清理梯度数据，防止内存泄漏
                if 'total_gradients' in locals():
                    del total_gradients
                if 'total_encoder_gradients' in locals():
                    del total_encoder_gradients
                if 'total_target_mask' in locals():
                    del total_target_mask
                if 'total_encoder_mask' in locals():
                    del total_encoder_mask
                if 'total_effective_mask' in locals():
                    del total_effective_mask
                if 'avg_gradients' in locals():
                    del avg_gradients
                if 'avg_encoder_gradients' in locals():
                    del avg_encoder_gradients
                
                # 强制垃圾回收
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif self.mask_manager:
            # 没有累积数据，尝试用最后一个batch的梯度
            try:
                conv_gradients = []
                for name, param in self.model.named_parameters():
                    if 'conv' in name and 'weight' in name and param.grad is not None:
                        conv_gradients.append(param.grad.view(-1))
                
                if conv_gradients:
                    gradients = torch.cat(conv_gradients)
                    target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                    encoder_gradients = torch.mul(gradients, effective_mask)
                    
                    self.multi_loss.update_gradient_stats(
                        gradients.detach(),
                        encoder_gradients.detach(),
                        target_mask,
                        encoder_mask,
                        effective_mask
                    )
                    
                    # 保存统计量
                    self._prev_stats = {
                        'prevGM': self.multi_loss.prevGM,
                        'prevGH': self.multi_loss.prevGH,
                        'prevRatio': self.multi_loss.prevRatio
                    }
                    
                    del gradients, encoder_gradients, target_mask, encoder_mask, effective_mask
            except Exception as e:
                pass

        # 每5个epoch进行水印融合（放在梯度统计更新之后）
        if (current_epoch + 1) % 5 == 0:
            self._embed_watermark(client_id, current_epoch)
        
        # 更新参数变化量统计（用于水印保护的正则项）
        # 这应该在每轮训练结束后调用，记录本轮参数变化
        if hasattr(self.multi_loss, 'update_param_change_stats'):
            try:
                self.multi_loss.update_param_change_stats(self.model.state_dict())
            except Exception as e:
                pass  # 静默处理

        # 定期清理内存
        if current_epoch > 0 and current_epoch % 10 == 0:
            self._cleanup_memory()

        # 差分隐私噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

