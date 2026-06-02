import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification
    Paper: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (float or list/array)
                   If None, no class weighting
            gamma: Focusing parameter (default: 2.0, typical range: 0-5)
                   Higher gamma gives more attention to hard examples
            reduction: Specifies the reduction to apply to the output
                      'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        # 降低gamma以减轻FocalLoss对困难样本的过度关注
        self.gamma = 1.0  # 降低到1.0，更接近BCE
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Model predictions (logits) of shape (N, C) or (N, C, H, W)
            targets: Ground truth labels of shape (N, C) for multi-label or (N,) for single-label
        
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities for binary/multi-label classification
        probs = torch.sigmoid(inputs)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), 
                                                       reduction='none', 
                                                       weight=self.alpha)
        
        # Compute pt (probability of true class)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLoss:
    """
    多重损失函数，用于联邦水印的鲁棒性训练
    包含主任务损失和三个正则化项
    """
    
    def __init__(self, init_a=0.6523, init_b=0.0000800375825259, model=None):
        """
        初始化MultiLoss
        
        Args:
            init_a: 超参数，调节损失函数敏感性
            init_b: 超参数，调节损失函数敏感性
        """
        self.init_a = init_a
        self.init_b = init_b
        self.device = None
        self.model = model
        
        # 统计量初始化为可学习的 nn.Parameter（重新设计方案）
        # 这些参数需要被加入模型的优化器才能生效
        self._learnable_ratio = nn.Parameter(torch.tensor([1.0, 1.0]))  # [target_ratio, var_ratio]
        
        # 保留旧的 EMA 统计量用于显示
        self._ema_gm = 0.0
        self._ema_gh = 0.0
        self._ema_ratio = 0.0
        
        # 当前batch的统计量
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
        
        # 调试打印标志
        self._debug_printed = False
        
        # 指数移动平均的平滑系数
        self.smoothing = 0.9
        
        # 标记是否已初始化（用于判断是否使用正则项）
        self._is_initialized = False
        
        # 目标值（可调整）
        self.target_gm_gh_ratio = 3.0  # GM/GH 目标值
        self.target_var_ratio = 1.5     # 方差比例目标值
        
        # 对比正则项参数（简化为：漂移 + 裕量）
        self.drift_lambda = 1.0
        self.margin_lambda = 1.0
        self.margin_ratio = 0.001
    
    @property
    def prevGM(self):
        return self._ema_gm
    
    @property
    def prevGH(self):
        return self._ema_gh
    
    @property
    def prevRatio(self):
        return self._ema_ratio
    
    def set_device(self, device):
        """设置设备"""
        self.device = device
        # 注意：_learnable_ratio 不再使用，不再移动到设备
        
    def get_alpha(self, current_epoch, total_epochs, alpha_early=None, alpha_late=None):
        """
        根据训练阶段返回alpha值
        
        Args:
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            alpha_early: 早期训练的alpha值（已从0.000005提升到0.00005）
            alpha_late: 晚期训练的alpha值（已从0.00001提升到0.0001）
            
        Returns:
            alpha值
        """
        # 默认值已提升，增强水印鲁棒性
        if alpha_early is None:
            alpha_early = 0.00005  # 从0.000005提升10倍
        if alpha_late is None:
            alpha_late = 0.0001  # 从0.00001提升10倍
            
        if current_epoch <= 0.3 * total_epochs:
            return alpha_early
        else:
            return alpha_late
    
    def update_gradient_stats(self, target_gradients, encoder_gradients, 
                            target_mask, encoder_mask, effective_mask):
        """
        更新梯度统计量（可微分版本 - 方案A核心）
        
        这个方法在 backward 过程中调用，使用指数移动平均更新统计量，
        使得 prevGM、prevGH、prevRatio 成为计算图的一部分
        
        Args:
            target_gradients: 目标模型梯度（卷积层）
            encoder_gradients: 编码器梯度
            target_mask: 目标模型梯度掩码
            encoder_mask: 编码器区域掩码
            effective_mask: 编码器有效梯度掩码 (encoder_mask × target_mask)
        
        Returns:
            current_gm, current_gh, current_ratio: 当前batch的统计量
        """
        # 确保在正确的设备上
        if self.device is None:
            self.device = target_gradients.device
        
        # 计算当前batch的梯度统计量（非零过滤）
        nonzero_threshold = 0.00001
        
        # 目标模型梯度过滤
        target_nonzero_mask = torch.where(
            torch.abs(target_gradients) <= nonzero_threshold, 
            torch.zeros_like(target_gradients), 
            torch.ones_like(target_gradients)
        )
        target_grad_filtered = torch.abs(target_gradients) * target_nonzero_mask
        target_nonzero_count = torch.sum(target_nonzero_mask)
        
        # 编码器梯度过滤
        encoder_nonzero_mask = torch.where(
            torch.abs(encoder_gradients) <= nonzero_threshold, 
            torch.zeros_like(encoder_gradients), 
            torch.ones_like(encoder_gradients)
        )
        encoder_grad_filtered = torch.abs(encoder_gradients) * encoder_nonzero_mask
        encoder_nonzero_count = torch.sum(encoder_nonzero_mask)
        
        # 计算平均梯度量级
        if target_nonzero_count > 0:
            current_gm = torch.sum(target_grad_filtered) / target_nonzero_count
        else:
            current_gm = torch.tensor(0.0, device=self.device)
            
        if encoder_nonzero_count > 0:
            current_gh = torch.sum(encoder_grad_filtered) / encoder_nonzero_count
        else:
            current_gh = torch.tensor(0.0, device=self.device)
        
        # 存储当前batch统计量（用于后续调试）
        self.current_grad_M = current_gm.detach().item()
        self.current_grad_H = current_gh.detach().item()
        
        # 计算方差
        if target_nonzero_count > 0:
            target_mean = torch.sum(target_grad_filtered) / target_nonzero_count
            target_var = torch.sum(torch.pow(target_grad_filtered - target_mean, 2)) / target_nonzero_count
            self.current_var_M = target_var.detach().item()
        else:
            self.current_var_M = 0.0
            target_var = torch.tensor(0.0, device=self.device)
            
        if encoder_nonzero_count > 0:
            encoder_mean = torch.sum(encoder_grad_filtered) / encoder_nonzero_count
            encoder_var = torch.sum(torch.pow(encoder_grad_filtered - encoder_mean, 2)) / encoder_nonzero_count
            self.current_var_H = encoder_var.detach().item()
        else:
            self.current_var_H = 0.0
            encoder_var = torch.tensor(0.0, device=self.device)
        
        # 计算方差比例
        if encoder_var > 0:
            current_ratio = target_var / encoder_var
        else:
            current_ratio = torch.tensor(1.0, device=self.device)
        
        # EMA 更新用于显示的统计量（不参与优化）
        self._ema_gm = self.smoothing * self._ema_gm + (1 - self.smoothing) * current_gm.detach().item()
        self._ema_gh = self.smoothing * self._ema_gh + (1 - self.smoothing) * current_gh.detach().item()
        self._ema_ratio = self.smoothing * self._ema_ratio + (1 - self.smoothing) * current_ratio.detach().item()
        
        # 标记已初始化
        self._is_initialized = True
        
        return current_gm.detach(), current_gh.detach(), current_ratio.detach()
    
    def update_gradient_stats_simple(self, target_gradients, encoder_gradients, 
                                     target_mask, encoder_mask, effective_mask):
        """
        简化版本的梯度统计更新（非可微分，用于离线统计）
        
        这个方法用于在训练循环外部更新统计量，不需要梯度追踪
        """
        nonzero_threshold = 0.00001
        
        target_nonzero_mask = torch.where(
            torch.abs(target_gradients) <= nonzero_threshold, 
            torch.zeros_like(target_gradients), 
            torch.ones_like(target_gradients)
        )
        target_grad_filtered = torch.abs(target_gradients) * target_nonzero_mask
        target_nonzero_count = torch.sum(target_nonzero_mask).item()
        
        encoder_nonzero_mask = torch.where(
            torch.abs(encoder_gradients) <= nonzero_threshold, 
            torch.zeros_like(encoder_gradients), 
            torch.ones_like(encoder_gradients)
        )
        encoder_grad_filtered = torch.abs(encoder_gradients) * encoder_nonzero_mask
        encoder_nonzero_count = torch.sum(encoder_nonzero_mask).item()
        
        if target_nonzero_count > 0:
            self.current_grad_M = torch.sum(target_grad_filtered).item() / target_nonzero_count
        else:
            self.current_grad_M = 0.0
            
        if encoder_nonzero_count > 0:
            self.current_grad_H = torch.sum(encoder_grad_filtered).item() / encoder_nonzero_count
        else:
            self.current_grad_H = 0.0
        
        # 计算方差
        if target_nonzero_count > 0:
            target_mean = torch.sum(target_grad_filtered) / target_nonzero_count
            target_var = torch.sum(torch.pow(target_grad_filtered - target_mean, 2)) / target_nonzero_count
            self.current_var_M = target_var.item()
        else:
            self.current_var_M = 0.0
            
        if encoder_nonzero_count > 0:
            encoder_mean = torch.sum(encoder_grad_filtered) / encoder_nonzero_count
            encoder_var = torch.sum(torch.pow(encoder_grad_filtered - encoder_mean, 2)) / encoder_nonzero_count
            self.current_var_H = encoder_var.item()
        else:
            self.current_var_H = 0.0
        
        if self.current_var_H > 0:
            self._ema_ratio = self.current_var_M / self.current_var_H
        else:
            self._ema_ratio = 1.0
        
        self._is_initialized = True
    
    def update_gradient_stats_batch(self, gradient_batch_list):
        """
        批量更新梯度统计量（用于每轮联邦训练结束后）
        
        Args:
            gradient_batch_list: 包含多个batch梯度数据的列表
        """
        if not gradient_batch_list:
            return
            
        # 合并所有batch的梯度数据
        all_gradients = []
        all_encoder_gradients = []
        all_target_masks = []
        all_encoder_masks = []
        all_effective_masks = []
        
        for batch_data in gradient_batch_list:
            all_gradients.append(batch_data['gradients'])
            all_encoder_gradients.append(batch_data['encoder_gradients'])
            all_target_masks.append(batch_data['target_mask'])
            all_encoder_masks.append(batch_data['encoder_mask'])
            all_effective_masks.append(batch_data['effective_mask'])
        
        # 合并所有梯度
        combined_gradients = torch.cat(all_gradients, dim=0)
        combined_encoder_gradients = torch.cat(all_encoder_gradients, dim=0)
        combined_target_mask = torch.cat(all_target_masks, dim=0)
        combined_encoder_mask = torch.cat(all_encoder_masks, dim=0)
        combined_effective_mask = torch.cat(all_effective_masks, dim=0)
        
        # 使用合并后的梯度更新统计量
        self.update_gradient_stats(
            combined_gradients, combined_encoder_gradients, 
            combined_target_mask, combined_encoder_mask, combined_effective_mask
        )
    
    def compute_loss(self, main_loss, current_epoch, total_epochs, alpha_early=None, alpha_late=None):
        """
        计算多重损失（方案A：在 backward 后更新统计量，使正则项可微分）
        
        Args:
            main_loss: 主任务损失
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            alpha_early: 早期训练的alpha值（从args传入）
            alpha_late: 晚期训练的alpha值（从args传入）
            
        Returns:
            总损失值
        """
        # 第一轮只使用主任务损失
        if current_epoch == 0:
            return main_loss
            
        # 获取alpha值
        alpha = self.get_alpha(current_epoch, total_epochs, alpha_early, alpha_late)
        
        # 确保设备一致
        device = main_loss.device
        if self.device is None:
            self.set_device(device)
        
        # 只使用 reg_term3 (自适应权重正则项)
        reg_term3 = self._compute_adaptive_weight_term(main_loss) if self._is_initialized else torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss = main_loss + reg_term3
        
        return total_loss
    
    def compute_loss_and_backward(self, main_loss, target_mask, encoder_mask, effective_mask,
                                  current_epoch, total_epochs,
                                  alpha_early=None, alpha_late=None):
        """
        计算损失并进行反向传播。

        逻辑：
        1. 先对 main_loss 反传，得到当前 batch 的梯度
        2. 基于梯度计算对比正则项：约束水印区域梯度小于非水印区域梯度
        3. 对比正则项继续反传，影响模型参数
        """
        device = main_loss.device
        if self.device is None:
            self.set_device(device)

        if current_epoch == 0:
            main_loss.backward()
            return main_loss

        main_loss.backward(retain_graph=True)

        if self.model is None or effective_mask is None:
            return main_loss

        conv_gradients = []
        for name, param in self.model.named_parameters():
            if 'conv' in name and 'weight' in name and param.grad is not None:
                conv_gradients.append(param.grad.view(-1))

        if not conv_gradients:
            return main_loss

        gradients = torch.cat(conv_gradients)
        encoder_gradients = torch.mul(gradients, effective_mask)
        self.update_gradient_stats(gradients, encoder_gradients, target_mask, encoder_mask, effective_mask)

        contrastive_reg = self.compute_contrastive_reg(gradients, effective_mask)
        if contrastive_reg.requires_grad:
            contrastive_reg.backward(retain_graph=True)

        if not hasattr(self, '_epoch_printed'):
            self._epoch_printed = set()
        if current_epoch not in self._epoch_printed:
            self._epoch_printed.add(current_epoch)
            ratio = (contrastive_reg.item() / main_loss.item() * 100) if main_loss.item() > 0 else 0
            print(f"[MultiLoss] E{current_epoch} | loss:{main_loss.item():.4f} | reg_ratio:{ratio:.1f}% | contrastive_reg:{contrastive_reg.item():.6f}")

        return main_loss + contrastive_reg
    
    def _compute_adaptive_weight_term(self, main_loss):
        """计算自适应权重正则项（可微分版本）"""
        if not self._is_initialized or self._ema_gm == 0:
            return torch.tensor(0.0, device=self.device or 'cpu')

        # 防止main_loss过大导致计算不稳定
        main_loss_clamped = torch.clamp(main_loss, max=self.init_a - 1e-6)
        beta2 = self._ema_gm * torch.abs(1 / (self.init_a - main_loss_clamped)) / self.init_b
        reg_term3 = torch.exp(-1 * beta2) * beta2
        return reg_term3

    def compute_contrastive_reg(self, gradients, effective_mask):
        """
        计算联合正则项：水印区域的漂移惩罚 + 裕量惩罚。

        正则形式：
        R = lambda_drift * mean(|g_w|)^2 + lambda_margin * max(0, mean(|g_w|) - delta)^2

        其中：
        - g_w 是水印区域梯度
        - delta 是允许的漂移阈值，按非水印区域梯度的比例自适应设置

        Args:
            gradients: 模型参数的梯度（拼接的一维向量）
            effective_mask: 水印区域掩码（1表示水印区域）

        Returns:
            reg_term: 联合正则项（标量，可微分）
        """
        if gradients is None or effective_mask is None:
            return torch.tensor(0.0, device=self.device or 'cpu')

        eps = 1e-8
        watermark_mask = effective_mask.float()
        non_watermark_mask = 1.0 - watermark_mask

        watermark_grads = torch.abs(gradients) * watermark_mask
        watermark_count = torch.sum(watermark_mask) + eps
        mean_watermark_grad = torch.sum(watermark_grads) / watermark_count

        non_watermark_grads = torch.abs(gradients) * non_watermark_mask
        non_watermark_count = torch.sum(non_watermark_mask) + eps
        mean_non_watermark_grad = torch.sum(non_watermark_grads) / non_watermark_count

        drift_penalty = mean_watermark_grad.pow(2)
        delta = self.margin_ratio * mean_non_watermark_grad.detach() + eps
        margin_penalty = torch.clamp(mean_watermark_grad - delta, min=0.0).pow(2)

        reg_term = self.drift_lambda * drift_penalty + self.margin_lambda * margin_penalty
        return reg_term
    
    def reset_batch_stats(self):
        """重置当前batch的统计量"""
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
    
    def get_stats(self):
        """获取当前统计量"""
        return {
            'prevGM': self._ema_gm,
            'prevGH': self._ema_gh,
            'prevRatio': self._ema_ratio,
            'current_grad_M': self.current_grad_M,
            'current_grad_H': self.current_grad_H,
            'current_var_M': self.current_var_M,
            'current_var_H': self.current_var_H,
            'is_initialized': self._is_initialized,
            'contrastive_ratio': self.drift_lambda,
            'contrastive_lambda': self.margin_lambda,
        }
    
    def reset_stats(self):
        """重置统计量"""
        self._ema_gm = 0.0
        self._ema_gh = 0.0
        self._ema_ratio = 1.0
        self._is_initialized = False
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
