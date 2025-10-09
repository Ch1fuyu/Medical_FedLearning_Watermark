import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
    def get_alpha(self, current_epoch, total_epochs):
        """
        根据训练阶段返回alpha值
        
        Args:
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            
        Returns:
            alpha值
        """
        if current_epoch <= 0.3 * total_epochs:
            return 0.000005
        else:
            return 0.00001
    
    def update_gradient_stats(self, target_gradients, encoder_gradients, 
                            target_mask, encoder_mask, effective_mask):
        """
        更新梯度统计量
        
        Args:
            target_gradients: 目标模型梯度
            encoder_gradients: 编码器梯度
            target_mask: 目标模型梯度掩码
            encoder_mask: 编码器区域掩码
            effective_mask: 编码器有效梯度掩码 (encoder_mask × target_mask)
        """
        # 计算目标模型梯度量级
        target_grad_abs = torch.abs(target_gradients)
        masked_target_grad = torch.mul(target_grad_abs, target_mask)
        self.current_grad_M = torch.sum(masked_target_grad).item()
        
        # 计算编码器梯度量级
        encoder_grad_abs = torch.abs(encoder_gradients)
        masked_encoder_grad = torch.mul(encoder_grad_abs, effective_mask)
        self.current_grad_H = torch.sum(masked_encoder_grad).item()
        
        # 计算非零梯度数量
        target_nonzero = torch.sum(target_mask).item()
        encoder_nonzero = torch.sum(effective_mask).item()
        
        # 计算平均梯度量级
        if target_nonzero > 0:
            self.prevGM = self.current_grad_M / target_nonzero
        if encoder_nonzero > 0:
            self.prevGH = self.current_grad_H / encoder_nonzero
            
        # 计算梯度方差
        if target_nonzero > 0:
            target_mean = torch.sum(masked_target_grad) / target_nonzero
            target_var = torch.sum(torch.pow(masked_target_grad - target_mean, 2)) / target_nonzero
            self.current_var_M = target_var.item()
            
        if encoder_nonzero > 0:
            encoder_mean = torch.sum(masked_encoder_grad) / encoder_nonzero
            encoder_var = torch.sum(torch.pow(masked_encoder_grad - encoder_mean, 2)) / encoder_nonzero
            self.current_var_H = encoder_var.item()
            
        # 更新方差比例
        if self.current_var_H > 0:
            self.prevRatio = self.current_var_M / self.current_var_H
    
    def compute_loss(self, main_loss, current_epoch, total_epochs):
        """
        计算多重损失
        
        Args:
            main_loss: 主任务损失
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            
        Returns:
            总损失值
        """
        # 第一轮只使用主任务损失
        if current_epoch == 0:
            return main_loss
            
        # 获取alpha值
        alpha = self.get_alpha(current_epoch, total_epochs)
        
        # 计算正则化项
        reg_term1 = self._compute_gradient_balance_term(alpha)
        reg_term2 = self._compute_variance_ratio_term(alpha)
        reg_term3 = self._compute_adaptive_weight_term(main_loss)
        
        # 总损失
        total_loss = main_loss + reg_term1 + reg_term2 + reg_term3
        
        return total_loss
    
    def _compute_gradient_balance_term(self, alpha):
        """计算梯度平衡正则项"""
        if self.prevGH == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        beta1 = torch.abs(torch.tensor(self.prevGM / self.prevGH))
        reg_term1 = alpha * (3 - beta1) * (3 - beta1)
        return reg_term1
    
    def _compute_variance_ratio_term(self, alpha):
        """计算方差比例正则项"""
        reg_term2 = alpha * (1.5 - self.prevRatio) * (1.5 - self.prevRatio)
        return reg_term2
    
    def _compute_adaptive_weight_term(self, main_loss):
        """计算自适应权重正则项"""
        if self.prevGM == 0:
            return torch.tensor(0.0, requires_grad=True)
            
        beta2 = self.prevGM * torch.abs(1 / (self.init_a - main_loss)) / self.init_b
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
