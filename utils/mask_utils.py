import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.key_matrix_utils import KeyMatrixManager

class MaskManager:
    """
    掩码管理器，用于处理目标模型和编码器的掩码操作
    """
    
    def __init__(self, model, key_matrix_manager: Optional[KeyMatrixManager] = None):
        """
        初始化掩码管理器
        
        Args:
            model: 目标模型（ResNet18）
            key_matrix_manager: 密钥矩阵管理器
        """
        self.model = model
        self.key_matrix_manager = key_matrix_manager
        self.target_mask = None
        self.encoder_mask = None
        self.effective_mask = None
        self.param_positions = {}
        
        # 初始化掩码
        self._initialize_masks()
    
    def _initialize_masks(self):
        """初始化各种掩码"""
        # 只获取卷积层参数总数
        conv_params = []
        for name, param in self.model.named_parameters():
            if 'conv' in name and 'weight' in name:
                conv_params.append(param)
        
        total_conv_params = sum(p.numel() for p in conv_params)
        
        # 初始化目标模型掩码（全1，表示所有卷积层参数位置）
        self.target_mask = torch.ones(total_conv_params, dtype=torch.float32)
        
        # 初始化编码器掩码（全0，后续根据密钥矩阵更新）
        self.encoder_mask = torch.zeros(total_conv_params, dtype=torch.float32)
        
        # 构建卷积层参数位置映射
        self._build_conv_param_positions()
    
    def _build_conv_param_positions(self):
        """构建卷积层参数名称到全局索引的映射"""
        current_idx = 0
        for name, param in self.model.named_parameters():
            if 'conv' in name and 'weight' in name:
                param_size = param.numel()
                self.param_positions[name] = (current_idx, current_idx + param_size)
                current_idx += param_size
    
    def update_encoder_mask(self, client_id: int = None):
        """
        更新编码器掩码，包含所有客户端的水印位置
        
        Args:
            client_id: 客户端ID（可选，如果为None则更新所有客户端）
        """
        if self.key_matrix_manager is None:
            print("警告: 没有密钥矩阵管理器，无法更新编码器掩码")
            return
        
        try:
            # 重置编码器掩码
            self.encoder_mask.zero_()
            
            # 获取所有客户端ID
            if client_id is not None:
                client_ids = [client_id]
            else:
                client_ids = self.key_matrix_manager.list_clients()
            
            # 加载所有客户端的位置信息并合并
            all_positions = set()
            for cid in client_ids:
                try:
                    positions = self.key_matrix_manager.load_positions(cid)
                    all_positions.update(positions)
                except Exception as e:
                    print(f"加载客户端 {cid} 位置信息失败: {e}")
                    continue
            
            # 更新编码器掩码（包含所有客户端的位置）
            for param_name, local_idx in all_positions:
                if param_name in self.param_positions:
                    start_idx, end_idx = self.param_positions[param_name]
                    param_length = end_idx - start_idx
                    
                    # 检查局部索引是否在参数范围内
                    if local_idx < param_length:
                        global_idx = start_idx + local_idx
                        
                        if global_idx < len(self.encoder_mask):
                            self.encoder_mask[global_idx] = 1.0
                        else:
                            print(f"⚠️  全局索引超出掩码范围: {param_name}[{local_idx}] -> {global_idx} >= {len(self.encoder_mask)}")
                    else:
                        print(f"⚠️  局部索引超出参数范围: {param_name}[{local_idx}] >= {param_length}")
            
            print(f"✓ 编码器掩码已更新，包含 {len(all_positions)} 个水印位置")
                        
        except Exception as e:
            print(f"更新编码器掩码失败: {e}")
            self.encoder_mask.zero_()
    
    def get_masks(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取当前掩码
        
        Args:
            device: 设备
            
        Returns:
            (target_mask, encoder_mask, effective_mask)
        """
        # 计算有效掩码 = 编码器掩码 × 目标模型掩码
        effective_mask = self.encoder_mask * self.target_mask
        
        return (self.target_mask.to(device), 
                self.encoder_mask.to(device), 
                effective_mask.to(device))
    
    def apply_mask_to_gradients(self, gradients: torch.Tensor, 
                              mask: torch.Tensor) -> torch.Tensor:
        """
        将掩码应用到梯度上
        
        Args:
            gradients: 梯度张量
            mask: 掩码张量
            
        Returns:
            应用掩码后的梯度
        """
        return torch.mul(gradients, mask)
    
    def get_encoder_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        提取编码器区域的梯度
        
        Args:
            gradients: 完整梯度张量
            
        Returns:
            编码器区域的梯度
        """
        _, _, effective_mask = self.get_masks(gradients.device)
        return self.apply_mask_to_gradients(gradients, effective_mask)
    
    def get_target_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        提取目标模型的梯度
        
        Args:
            gradients: 完整梯度张量
            
        Returns:
            目标模型的梯度
        """
        target_mask, _, _ = self.get_masks(gradients.device)
        return self.apply_mask_to_gradients(gradients, target_mask)
    
    def get_mask_stats(self) -> Dict[str, int]:
        """
        获取掩码统计信息
        
        Returns:
            掩码统计信息
        """
        target_count = int(torch.sum(self.target_mask).item())
        encoder_count = int(torch.sum(self.encoder_mask).item())
        effective_count = int(torch.sum(self.encoder_mask * self.target_mask).item())
        
        return {
            'target_params': target_count,
            'encoder_params': encoder_count,
            'effective_params': effective_count,
            'encoder_ratio': encoder_count / target_count if target_count > 0 else 0
        }

def create_mask_manager(model, key_matrix_dir: str, args=None) -> MaskManager:
    """
    便捷函数：创建掩码管理器
    
    Args:
        model: 目标模型
        key_matrix_dir: 密钥矩阵目录
        args: 参数对象，包含水印缩放相关配置
        
    Returns:
        MaskManager实例
    """
    try:
        key_manager = KeyMatrixManager(key_matrix_dir, args)
        return MaskManager(model, key_manager)
    except Exception as e:
        print(f"创建密钥矩阵管理器失败: {e}")
        return MaskManager(model, None)
