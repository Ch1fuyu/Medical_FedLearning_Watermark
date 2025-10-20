from typing import Dict

import torch


class WatermarkScaling:
    """水印参数固定缩放管理器（单例模式）"""
    
    _instances = {}  # 类变量，存储不同缩放因子的实例
    
    def __new__(cls, scaling_factor=0.1):
        """
        单例模式：相同缩放因子只创建一个实例
        
        Args:
            scaling_factor: 固定缩放因子（默认0.1）
        """
        if scaling_factor not in cls._instances:
            instance = super(WatermarkScaling, cls).__new__(cls)
            cls._instances[scaling_factor] = instance
        return cls._instances[scaling_factor]
    
    def __init__(self, scaling_factor=0.1):
        """
        初始化水印缩放管理器
        
        Args:
            scaling_factor: 固定缩放因子（默认0.1）
        """
        # 避免重复初始化
        if hasattr(self, 'scaling_factor'):
            return
            
        self.scaling_factor = scaling_factor
        self.scaling_stats = {}  # 存储缩放统计信息
        
    def calculate_scaling_factor(self, model_params: Dict[str, torch.Tensor], 
                                watermark_positions: list) -> float:
        """
        获取固定缩放因子
        
        Args:
            model_params: 模型参数字典
            watermark_positions: 水印位置列表
            
        Returns:
            固定缩放因子
        """
        return self.scaling_factor
    
    def scale_watermark_parameters(self, watermark_values: torch.Tensor, 
                                  scale_factor: float = None) -> torch.Tensor:
        """
        缩放水印参数
        
        Args:
            watermark_values: 原始水印参数
            scale_factor: 缩放因子（如果为None则使用默认值）
            
        Returns:
            缩放后的水印参数
        """
        if scale_factor is None:
            scale_factor = self.scaling_factor
            
        if scale_factor == 1.0:
            return watermark_values.clone()
        
        # 使用double精度进行计算，然后转换回float
        scale_factor_tensor = torch.tensor(scale_factor, dtype=torch.float64, device=watermark_values.device)
        watermark_double = watermark_values.double()
        scaled_values = (watermark_double * scale_factor_tensor).float()
        
        print(f"🔧 水印参数缩放: {scale_factor:.6f}x")
        print(f"   原始范围: [{watermark_values.min().item():.6f}, {watermark_values.max().item():.6f}]")
        print(f"   缩放后范围: [{scaled_values.min().item():.6f}, {scaled_values.max().item():.6f}]")
        
        return scaled_values
    
    def restore_watermark_parameters(self, scaled_watermark_values: torch.Tensor, 
                                   scale_factor: float = None) -> torch.Tensor:
        """
        恢复水印参数（用于水印提取）
        
        Args:
            scaled_watermark_values: 缩放后的水印参数
            scale_factor: 缩放因子（如果为None则使用默认值）
            
        Returns:
            恢复后的水印参数
        """
        if scale_factor is None:
            scale_factor = self.scaling_factor
            
        if scale_factor == 1.0:
            return scaled_watermark_values.clone()
        
        # 使用double精度进行计算，然后转换回float
        scale_factor_tensor = torch.tensor(scale_factor, dtype=torch.float64, device=scaled_watermark_values.device)
        scaled_double = scaled_watermark_values.double()
        restored_values = (scaled_double / scale_factor_tensor).float()
        
        return restored_values
    
    @classmethod
    def clear_instances(cls):
        """清理所有实例缓存"""
        cls._instances.clear()
    
    @classmethod
    def get_instance_count(cls):
        """获取当前实例数量"""
        return len(cls._instances)
    
    @classmethod
    def get_instance_info(cls):
        """获取实例信息"""
        return {factor: f"WatermarkScaling(factor={factor})" for factor in cls._instances.keys()}


def create_watermark_scaler(scaling_factor=0.1) -> WatermarkScaling:
    """
    便捷函数：创建水印缩放器
    
    Args:
        scaling_factor: 固定缩放因子
        
    Returns:
        WatermarkScaling实例
    """
    return WatermarkScaling(scaling_factor)