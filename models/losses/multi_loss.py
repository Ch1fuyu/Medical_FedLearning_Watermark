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
    
    def __init__(self, init_a=0.6523, init_b=0.0000800375825259):
        """
        初始化MultiLoss
        
        Args:
            init_a: 超参数，调节损失函数敏感性
            init_b: 超参数，调节损失函数敏感性
        """
        self.init_a = init_a
        self.init_b = init_b
        
        # 统计量初始化
        self.prevGM = 0.0  # 目标模型梯度量级
        self.prevGH = 0.0  # 编码器梯度量级
        self.prevRatio = 1.0  # 方差比例
        
        # 当前batch的统计量
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
        
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
        更新梯度统计量（只处理卷积层，添加非零梯度过滤）
        
        Args:
            target_gradients: 目标模型梯度（卷积层）
            encoder_gradients: 编码器梯度
            target_mask: 目标模型梯度掩码
            encoder_mask: 编码器区域掩码
            effective_mask: 编码器有效梯度掩码 (encoder_mask × target_mask)
        """
        # 添加非零梯度过滤（阈值：0.00001）
        target_nonzero_mask = torch.where(torch.abs(target_gradients) <= 0.00001, 
                                        torch.zeros_like(target_gradients), 
                                        torch.ones_like(target_gradients))
        
        # 计算目标模型非零梯度量级
        target_grad_filtered = torch.mul(torch.abs(target_gradients), target_nonzero_mask)
        self.current_grad_M = torch.sum(target_grad_filtered).item()
        
        # 计算编码器非零梯度量级
        encoder_nonzero_mask = torch.where(torch.abs(encoder_gradients) <= 0.00001, 
                                         torch.zeros_like(encoder_gradients), 
                                         torch.ones_like(encoder_gradients))
        encoder_grad_filtered = torch.mul(torch.abs(encoder_gradients), encoder_nonzero_mask)
        self.current_grad_H = torch.sum(encoder_grad_filtered).item()
        
        # 计算非零梯度数量
        target_nonzero = torch.sum(target_nonzero_mask).item()
        encoder_nonzero = torch.sum(encoder_nonzero_mask).item()
        
        # # 调试信息
        # print(f"梯度统计调试:")
        # print(f"  target_gradients shape: {target_gradients.shape}")
        # print(f"  target_gradients min/max: {target_gradients.min().item():.8f}/{target_gradients.max().item():.8f}")
        # print(f"  target_nonzero: {target_nonzero}")
        # print(f"  current_grad_M: {self.current_grad_M:.8f}")
        # print(f"  encoder_gradients shape: {encoder_gradients.shape}")
        # print(f"  encoder_nonzero: {encoder_nonzero}")
        # print(f"  current_grad_H: {self.current_grad_H:.8f}")
        
        # 计算平均梯度量级
        # prevGM: 卷积层非零梯度的平均量级
        if target_nonzero > 0:
            self.prevGM = self.current_grad_M / target_nonzero
        else:
            self.prevGM = 0.0  # 没有非零梯度时设为0
            
        # prevGH: 编码器区域非零梯度的平均量级
        if encoder_nonzero > 0:
            self.prevGH = self.current_grad_H / encoder_nonzero
        else:
            self.prevGH = 0.0  # 没有非零梯度时设为0
            
        # 计算梯度方差（使用过滤后的梯度）
        if target_nonzero > 0:
            target_mean = torch.sum(target_grad_filtered) / target_nonzero
            target_var = torch.sum(torch.pow(target_grad_filtered - target_mean, 2)) / target_nonzero
            self.current_var_M = target_var.item()
        else:
            self.current_var_M = 0.0
            
        if encoder_nonzero > 0:
            encoder_mean = torch.sum(encoder_grad_filtered) / encoder_nonzero
            encoder_var = torch.sum(torch.pow(encoder_grad_filtered - encoder_mean, 2)) / encoder_nonzero
            self.current_var_H = encoder_var.item()
        else:
            self.current_var_H = 0.0
            
        # 更新方差比例
        if self.current_var_H > 0:
            self.prevRatio = self.current_var_M / self.current_var_H
        else:
            self.prevRatio = 1.0  # 默认比例
    
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
    
    def compute_loss(self, main_loss, current_epoch, total_epochs, alpha_early=None, alpha_late=None,
                     use_reg1=True, use_reg2=True, use_reg3=True):
        """
        计算多重损失
        
        Args:
            main_loss: 主任务损失
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            alpha_early: 早期训练的alpha值（从args传入）
            alpha_late: 晚期训练的alpha值（从args传入）
            use_reg1: 是否使用梯度平衡正则项（reg_term1）
            use_reg2: 是否使用方差比例正则项（reg_term2）
            use_reg3: 是否使用自适应权重正则项（reg_term3）
            
        Returns:
            总损失值
        """
        # 第一轮只使用主任务损失
        if current_epoch == 0:
            return main_loss
            
        # 获取alpha值
        alpha = self.get_alpha(current_epoch, total_epochs, alpha_early, alpha_late)
        
        # 计算正则化项（根据参数选择性启用）
        # 确保禁用时的0张量在正确的设备上
        device = main_loss.device
        reg_term1 = self._compute_gradient_balance_term(alpha) if use_reg1 else torch.tensor(0.0, requires_grad=True, device=device)
        reg_term2 = self._compute_variance_ratio_term(alpha) if use_reg2 else torch.tensor(0.0, requires_grad=True, device=device)
        reg_term3 = self._compute_adaptive_weight_term(main_loss) if use_reg3 else torch.tensor(0.0, requires_grad=True, device=device)
        
        # 总损失
        total_loss = main_loss + reg_term1 + reg_term2 + reg_term3
        
        return total_loss
    
    def _compute_gradient_balance_term(self, alpha):
        """计算梯度平衡正则项"""
        # 如果统计量未初始化，返回0
        if self.prevGM == 0 or self.prevGH == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        beta1 = torch.abs(torch.tensor(self.prevGM / self.prevGH))
        reg_term1 = alpha * (3 - beta1) * (3 - beta1)
        return reg_term1
    
    def _compute_variance_ratio_term(self, alpha):
        """计算方差比例正则项"""
        # 如果方差比例未初始化，返回0
        if self.prevRatio == 1.0:  # 初始值
            return torch.tensor(0.0, requires_grad=True)
            
        reg_term2 = alpha * (1.5 - self.prevRatio) * (1.5 - self.prevRatio)
        return reg_term2
    
    def _compute_adaptive_weight_term(self, main_loss):
        """计算自适应权重正则项"""
        if self.prevGM == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # 防止main_loss过大导致计算不稳定
        main_loss_clamped = torch.clamp(main_loss, max=self.init_a - 1e-6)
        beta2 = self.prevGM * torch.abs(1 / (self.init_a - main_loss_clamped)) / self.init_b
        reg_term3 = torch.exp(-1 * beta2) * beta2
        return reg_term3
    
    def reset_batch_stats(self):
        """重置当前batch的统计量"""
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
    
    def get_stats(self):
        """获取当前统计量"""
        return {
            'prevGM': self.prevGM,
            'prevGH': self.prevGH,
            'prevRatio': self.prevRatio,
            'current_grad_M': self.current_grad_M,
            'current_grad_H': self.current_grad_H,
            'current_var_M': self.current_var_M,
            'current_var_H': self.current_var_H
        }
