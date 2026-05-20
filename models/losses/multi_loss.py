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
    
    正则项作用：
    - reg1: 基于上一轮参数变化量，惩罚水印区域的过度变化
    - reg2: 基于上一轮参数变化方差，保持水印区域的稳定更新
    - reg3: 自适应权重调节
    """
    
    def __init__(self, init_a=0.6523, init_b=0.0000800375825259, device='cpu'):
        """
        初始化MultiLoss
        
        Args:
            init_a: 超参数，调节损失函数敏感性
            init_b: 超参数，调节损失函数敏感性
            device: 计算设备
        """
        self.init_a = init_a
        self.init_b = init_b
        self.device = device
        
        # 梯度统计量初始化
        self.prevGM = 0.0  # 目标模型梯度量级
        self.prevGH = 0.0  # 编码器梯度量级
        self.prevRatio = 1.0  # 方差比例
        
        # 当前batch的统计量
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
        
        # ========== 参数变化量跟踪（用于水印保护）==========
        # 训练前的模型参数（用于计算参数变化量）
        self.params_before_training = None
        # 上一轮水印区域的参数变化量
        self.prev_wm_param_change = 0.0
        # 上一轮非水印区域的参数变化量
        self.prev_nonwm_param_change = 0.0
        # 上一轮参数变化方差
        self.prev_param_change_variance = 0.0
        
        # 训练开始前调用的方法，保存当前参数
        self._wm_masks = None
        self._nonwm_masks = None
        
    def get_alpha(self, current_epoch, total_epochs, alpha_early=None, alpha_late=None):
        """
        根据训练阶段返回alpha值
        
        Args:
            current_epoch: 当前epoch
            total_epochs: 总epoch数
            
        Returns:
            alpha值
        """
        # 根据训练阶段返回alpha值
        # 默认值与 args.py 保持一致
        if alpha_early is None:
            alpha_early = 0.01
        if alpha_late is None:
            alpha_late = 0.02
            
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
        # 降低非零梯度阈值，允许更小的梯度参与统计
        # 使用 1e-10 作为阈值，避免因梯度过小导致统计失效
        nonzero_threshold = 1e-10
        target_nonzero_mask = torch.where(torch.abs(target_gradients) <= nonzero_threshold, 
                                        torch.zeros_like(target_gradients), 
                                        torch.ones_like(target_gradients))
        
        # 计算目标模型非零梯度量级
        target_grad_filtered = torch.mul(torch.abs(target_gradients), target_nonzero_mask)
        self.current_grad_M = torch.sum(target_grad_filtered).item()
        
        # 计算编码器非零梯度量级
        encoder_nonzero_mask = torch.where(torch.abs(encoder_gradients) <= nonzero_threshold, 
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
        # 确保禁用时的0张量在正确的设备上，并连接到计算图
        device = main_loss.device
        if use_reg1:
            reg_term1 = self._compute_gradient_balance_term(alpha, main_loss)
        else:
            reg_term1 = main_loss * 0.0  # 连接到计算图
        
        if use_reg2:
            reg_term2 = self._compute_variance_ratio_term(alpha, main_loss)
        else:
            reg_term2 = main_loss * 0.0  # 连接到计算图
        
        if use_reg3:
            reg_term3 = self._compute_adaptive_weight_term(main_loss)
        else:
            reg_term3 = main_loss * 0.0  # 连接到计算图
        
        # 总损失
        total_loss = main_loss + reg_term1 + reg_term2 + reg_term3
        
        return total_loss
    
    def _compute_gradient_balance_term(self, alpha, main_loss):
        """计算水印区域变化惩罚正则项（保护水印）
        
        惩罚逻辑：当水印区域的参数变化量超过阈值时施加惩罚
        直接限制水印区域的变化，而不是只惩罚不平衡
        
        通过 main_loss 构建计算图连接
        """
        wm_change = self.prev_wm_param_change
        nonwm_change = self.prev_nonwm_param_change
        
        # 水印变化阈值：如果水印区域变化超过这个值，就惩罚
        # 阈值基于非水印区域变化量来设定，确保水印变化 <= 非水印变化
        wm_threshold = nonwm_change * 1.0  # 目标：水印变化不超过非水印
        
        # 计算惩罚项：水印变化超过阈值时惩罚
        wm_penalty = max(0, wm_change - wm_threshold)
        
        # 同时惩罚绝对变化：如果水印变化本身过大，也惩罚
        # 设定一个绝对阈值（比如 1e-5），超过就惩罚
        abs_threshold = 1e-4
        abs_penalty = max(0, wm_change - abs_threshold)
        
        # 合并两种惩罚
        total_penalty = wm_penalty + 0.1 * abs_penalty
        
        # 构建计算图连接，确保梯度回传
        reg_term1 = alpha * total_penalty * main_loss
        return reg_term1
    
    def _compute_variance_ratio_term(self, alpha, main_loss):
        """计算参数变化方差正则项（保持稳定更新）
        
        惩罚逻辑：如果上一轮参数变化方差过大，则增加惩罚
        这会让模型倾向于更均匀地更新所有参数
        
        通过 main_loss 构建计算图连接
        """
        # 上一轮参数变化的方差
        variance = self.prev_param_change_variance
        
        # 方差阈值：超过这个值就惩罚
        variance_threshold = 1e-6
        variance_penalty = max(0, variance - variance_threshold)
        
        # 通过 main_loss 构建计算图连接
        reg_term2 = alpha * variance_penalty * main_loss
        return reg_term2
    
    def _compute_adaptive_weight_term(self, main_loss):
        """计算自适应权重正则项"""
        # 使用最小的非零值代替0，避免返回0
        grad_M = max(self.prevGM, 1e-10)
        
        # 防止main_loss过大导致计算不稳定
        main_loss_clamped = torch.clamp(main_loss, max=self.init_a - 1e-6)
        # 确保计算在正确的设备上，使用 tensor 进行计算
        init_a_tensor = torch.tensor(self.init_a, dtype=torch.float32, device=main_loss.device)
        init_b_tensor = torch.tensor(self.init_b, dtype=torch.float32, device=main_loss.device)
        grad_M_tensor = torch.tensor(grad_M, dtype=torch.float32, device=main_loss.device)
        
        beta2 = grad_M_tensor * torch.abs(1 / (init_a_tensor - main_loss_clamped)) / init_b_tensor
        reg_term3 = torch.exp(-1 * beta2) * beta2
        return reg_term3
    
    def reset_batch_stats(self):
        """重置当前batch的统计量"""
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
    
    def save_params_before_training(self, current_params, wm_masks, nonwm_masks):
        """
        保存训练前的参数（用于计算训练后的参数变化）
        
        在每轮训练开始前调用
        
        Args:
            current_params: 当前模型参数字典 (OrderedDict)
            wm_masks: 水印区域掩码字典
            nonwm_masks: 非水印区域掩码字典
        """
        self.params_before_training = {k: v.clone() for k, v in current_params.items()}
        self._wm_masks = wm_masks
        self._nonwm_masks = nonwm_masks
    
    def update_param_change_stats(self, current_params):
        """
        计算并更新参数变化量统计
        
        在每轮训练结束后调用，计算训练前后的参数变化
        
        Args:
            current_params: 训练后的模型参数字典 (OrderedDict)
        """
        if self.params_before_training is None:
            # 第一轮，没有基准参数
            return
        
        wm_changes = []
        nonwm_changes = []
        all_changes = []
        
        for name, param in current_params.items():
            if name not in self.params_before_training:
                continue
            
            prev_param = self.params_before_training[name]
            change = torch.abs(param - prev_param)
            
            # 根据掩码分类
            wm_mask = self._wm_masks.get(name) if self._wm_masks else None
            nonwm_mask = self._nonwm_masks.get(name) if self._nonwm_masks else None
            
            if wm_mask is not None:
                wm_changes.append(change[wm_mask.bool()].sum().item())
            else:
                wm_changes.append(change.sum().item())
                
            if nonwm_mask is not None:
                nonwm_changes.append(change[nonwm_mask.bool()].sum().item())
            else:
                nonwm_changes.append(change.sum().item())
            
            all_changes.extend(change.flatten().tolist())
        
        # 计算统计量
        self.prev_wm_param_change = sum(wm_changes) / max(len(wm_changes), 1)
        self.prev_nonwm_param_change = sum(nonwm_changes) / max(len(nonwm_changes), 1)
        
        if len(all_changes) > 0:
            mean_change = sum(all_changes) / len(all_changes)
            variance = sum((x - mean_change) ** 2 for x in all_changes) / len(all_changes)
            self.prev_param_change_variance = variance
        
        # 清空训练前参数
        self.params_before_training = None
    
    def get_stats(self):
        """获取当前统计量"""
        return {
            'prevGM': self.prevGM,
            'prevGH': self.prevGH,
            'prevRatio': self.prevRatio,
            'current_grad_M': self.current_grad_M,
            'current_grad_H': self.current_grad_H,
            'current_var_M': self.current_var_M,
            'current_var_H': self.current_var_H,
            # 参数变化量统计
            'prev_wm_param_change': self.prev_wm_param_change,
            'prev_nonwm_param_change': self.prev_nonwm_param_change,
            'prev_param_change_variance': self.prev_param_change_variance
        }
