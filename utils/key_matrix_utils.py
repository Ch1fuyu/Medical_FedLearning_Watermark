import torch
import json
import os
from typing import Dict, List, Tuple, Optional


class KeyMatrixManager:
    """密钥矩阵管理器，用于加载和管理密钥矩阵"""
    
    _instance = None
    
    def __new__(cls, key_matrix_dir: str):
        if cls._instance is None:
            cls._instance = super(KeyMatrixManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, key_matrix_dir: str):
        if hasattr(self, 'key_matrix_dir'):
            return
            
        self.key_matrix_dir = key_matrix_dir
        self.info = self._load_info()
        
        # 密钥矩阵和位置文件路径
        self.key_matrix_path = os.path.join(key_matrix_dir, 'key_matrix.pth')
        self.positions_path = os.path.join(key_matrix_dir, 'positions.json')
        
        # 缓存
        self._key_matrix = None
        self._positions = None
    
    def _load_info(self) -> dict:
        """加载密钥矩阵信息"""
        info_path = os.path.join(self.key_matrix_dir, 'key_matrix_info.json')
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"密钥矩阵信息文件不存在: {info_path}")
        
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def load_key_matrix(self) -> Dict[str, torch.Tensor]:
        """加载密钥矩阵"""
        if self._key_matrix is None:
            if not os.path.exists(self.key_matrix_path):
                raise FileNotFoundError(f"密钥矩阵不存在: {self.key_matrix_path}")
            self._key_matrix = torch.load(self.key_matrix_path, map_location='cpu', weights_only=False)
        return self._key_matrix
    
    def load_positions(self) -> List[Tuple[str, int]]:
        """加载水印位置
        
        Returns:
            位置列表 [(param_name, param_idx), ...]
        """
        if self._positions is None:
            if not os.path.exists(self.positions_path):
                raise FileNotFoundError(f"位置文件不存在: {self.positions_path}")
            
            with open(self.positions_path, 'r') as f:
                positions = json.load(f)
            
            # 将列表格式转换为元组格式
            self._positions = [(pos[0], pos[1]) for pos in positions]
        
        return self._positions
    
    def list_clients(self) -> List[str]:
        """
        列出密钥矩阵目录下的所有客户端子目录
        
        Returns:
            客户端ID列表，如 ['client5', 'client10', ...]
        """
        if not hasattr(self, '_client_ids'):
            self._client_ids = []
            if os.path.isdir(self.key_matrix_dir):
                for name in os.listdir(self.key_matrix_dir):
                    full_path = os.path.join(self.key_matrix_dir, name)
                    if os.path.isdir(full_path) and name.startswith('client'):
                        self._client_ids.append(name)
            self._client_ids = sorted(self._client_ids)
        return self._client_ids
    
    def embed_watermark(self, model_params: Dict[str, torch.Tensor], 
                       watermark_values: torch.Tensor = None, 
                       model=None,
                       encoder_params: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        将水印嵌入到模型参数中
        
        Args:
            model_params: 模型参数字典
            watermark_values: 水印值
            model: 模型对象（可选）
        
        Returns:
            嵌入水印后的模型参数
        """
        key_matrix = self.load_key_matrix()
        positions = self.load_positions()
        
        # 构建参数偏移映射
        needs_conversion = False
        param_offset_map = {}
        
        for param_name, idx in positions:
            if param_name in model_params:
                param_size = model_params[param_name].numel()
                if idx >= param_size:
                    needs_conversion = True
                    break
        
        if needs_conversion:
            current_offset = 0
            if model is not None:
                param_iter = model.named_parameters()
            else:
                param_iter = sorted(model_params.items(), key=lambda x: x[0])
            
            for name, param in param_iter:
                if name not in model_params:
                    continue
                param = model_params[name]
                
                is_conv_weight = (
                    'conv' in name.lower() and 'weight' in name.lower()
                ) or (
                    'downsample.0.weight' in name.lower()
                )
                
                if is_conv_weight and len(param.shape) == 4:
                    param_offset_map[name] = current_offset
                    current_offset += param.numel()
        
        # 复制模型参数
        watermarked_params = {}
        for name, param in model_params.items():
            watermarked_params[name] = param.clone()
        
        # 嵌入水印
        watermark_idx = 0
        for param_name_in_file, idx in positions:
            if watermark_idx >= len(watermark_values):
                break
            
            if needs_conversion:
                if param_name_in_file in watermarked_params:
                    param_size = watermarked_params[param_name_in_file].numel()
                    if idx < param_size:
                        actual_param_name = param_name_in_file
                        local_idx = idx
                    elif param_name_in_file in param_offset_map:
                        param_offset = param_offset_map[param_name_in_file]
                        if param_offset <= idx < param_offset + param_size:
                            actual_param_name = param_name_in_file
                            local_idx = idx - param_offset
                        else:
                            actual_param_name = None
                            for name, offset in param_offset_map.items():
                                if offset <= idx < offset + watermarked_params[name].numel():
                                    actual_param_name = name
                                    local_idx = idx - offset
                                    break
                            if actual_param_name is None:
                                raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
                    else:
                        actual_param_name = None
                        for name, offset in param_offset_map.items():
                            if offset <= idx < offset + watermarked_params[name].numel():
                                actual_param_name = name
                                local_idx = idx - offset
                                break
                        if actual_param_name is None:
                            raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
                else:
                    actual_param_name = None
                    for name, offset in param_offset_map.items():
                        if offset <= idx < offset + watermarked_params[name].numel():
                            actual_param_name = name
                            local_idx = idx - offset
                            break
                    if actual_param_name is None:
                        raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
            else:
                actual_param_name = param_name_in_file
                local_idx = idx
            
            if actual_param_name in watermarked_params:
                param_tensor = watermarked_params[actual_param_name].view(-1)
                param_size = param_tensor.numel()
                
                if local_idx < 0 or local_idx >= param_size:
                    raise IndexError(
                        f"错误: 局部索引 {local_idx} 超出参数 {actual_param_name} 的范围 [0, {param_size})"
                    )
                
                param_tensor[local_idx] = watermark_values[watermark_idx]
                watermark_idx += 1
            else:
                raise KeyError(f"错误: 参数名 '{actual_param_name}' 不在模型参数中")
        
        return watermarked_params
    
    def extract_watermark(self, model_params: Dict[str, torch.Tensor],
                         check_pruning: bool = False, model=None) -> torch.Tensor:
        """从模型参数中提取水印（单客户端模式，使用全局 positions）
        """
        positions = self.load_positions()
        
        needs_conversion = False
        param_offset_map = {}
        
        for param_name, idx in positions:
            if param_name in model_params:
                if idx >= model_params[param_name].numel():
                    needs_conversion = True
                    break
        
        if needs_conversion:
            current_offset = 0
            if model is not None:
                param_iter = model.named_parameters()
            else:
                param_iter = sorted(model_params.items(), key=lambda x: x[0])
            
            for name, param in param_iter:
                if name not in model_params:
                    continue
                param = model_params[name]
                
                is_conv_weight = (
                    'conv' in name.lower() and 'weight' in name.lower()
                ) or (
                    'downsample.0.weight' in name.lower()
                )
                
                if is_conv_weight and len(param.shape) == 4:
                    param_offset_map[name] = current_offset
                    current_offset += param.numel()
        
        watermark_values = []
        for param_name_in_file, idx in positions:
            if needs_conversion:
                if param_name_in_file in model_params:
                    param_size = model_params[param_name_in_file].numel()
                    if idx < param_size:
                        actual_param_name = param_name_in_file
                        local_idx = idx
                    elif param_name_in_file in param_offset_map:
                        param_offset = param_offset_map[param_name_in_file]
                        if param_offset <= idx < param_offset + param_size:
                            actual_param_name = param_name_in_file
                            local_idx = idx - param_offset
                        else:
                            actual_param_name = None
                            for name, offset in param_offset_map.items():
                                if offset <= idx < offset + model_params[name].numel():
                                    actual_param_name = name
                                    local_idx = idx - offset
                                    break
                            if actual_param_name is None:
                                raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
                    else:
                        actual_param_name = None
                        for name, offset in param_offset_map.items():
                            if offset <= idx < offset + model_params[name].numel():
                                actual_param_name = name
                                local_idx = idx - offset
                                break
                        if actual_param_name is None:
                            raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
                else:
                    actual_param_name = None
                    for name, offset in param_offset_map.items():
                        if offset <= idx < offset + model_params[name].numel():
                            actual_param_name = name
                            local_idx = idx - offset
                            break
                    if actual_param_name is None:
                        raise IndexError(f"错误: 全局索引 {idx} 无法映射到任何卷积层参数")
            else:
                actual_param_name = param_name_in_file
                local_idx = idx
            
            if actual_param_name in model_params:
                param_tensor = model_params[actual_param_name].view(-1)
                param_size = param_tensor.numel()
                
                if local_idx < 0 or local_idx >= param_size:
                    raise IndexError(
                        f"错误: 局部索引 {local_idx} 超出参数 {actual_param_name} 的范围 [0, {param_size})"
                    )
                
                watermark_value = param_tensor[local_idx]
                watermark_values.append(watermark_value)
            else:
                raise KeyError(f"错误: 参数名 '{actual_param_name}' 不在模型参数中")
        
        return torch.stack(watermark_values)
    
    def get_info(self) -> dict:
        """获取密钥矩阵信息"""
        return self.info.copy()
    
    def verify_key_matrix(self) -> bool:
        """验证密钥矩阵的完整性"""
        try:
            key_matrix = self.load_key_matrix()
            positions = self.load_positions()
            
            total_ones = sum(tensor.sum().item() for tensor in key_matrix.values())
            return int(total_ones) == len(positions)
        except Exception as e:
            print(f"密钥矩阵验证失败: {e}")
            return False
    
    @classmethod
    def clear_instance(cls):
        """清理实例缓存"""
        cls._instance = None


def load_key_matrix_manager(key_matrix_dir: str) -> KeyMatrixManager:
    """便捷函数：加载密钥矩阵管理器"""
    return KeyMatrixManager(key_matrix_dir)
