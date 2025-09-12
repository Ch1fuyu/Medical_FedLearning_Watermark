import torch
import json
import os
from typing import Dict, List, Tuple, Optional

class KeyMatrixManager:
    """密钥矩阵管理器，用于加载和管理密钥矩阵"""
    
    def __init__(self, key_matrix_dir: str):
        """
        初始化密钥矩阵管理器
        
        Args:
            key_matrix_dir: 密钥矩阵保存目录
        """
        self.key_matrix_dir = key_matrix_dir
        self.info = self._load_info()
        self.client_num = self.info['client_num']
        
    def _load_info(self) -> dict:
        """加载密钥矩阵信息"""
        info_path = os.path.join(self.key_matrix_dir, 'key_matrix_info.json')
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"密钥矩阵信息文件不存在: {info_path}")
        
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def load_key_matrix(self, client_id: int) -> Dict[str, torch.Tensor]:
        """
        加载指定客户端的密钥矩阵
        
        Args:
            client_id: 客户端ID
            
        Returns:
            密钥矩阵字典
        """
        if client_id < 0 or client_id >= self.client_num:
            raise ValueError(f"客户端ID {client_id} 超出范围 [0, {self.client_num-1}]")
        
        client_dir = os.path.join(self.key_matrix_dir, f'client_{client_id}')
        key_matrix_path = os.path.join(client_dir, 'key_matrix.pth')
        
        if not os.path.exists(key_matrix_path):
            raise FileNotFoundError(f"客户端 {client_id} 的密钥矩阵不存在: {key_matrix_path}")
        
        return torch.load(key_matrix_path, map_location='cpu', weights_only=False)
    
    def load_positions(self, client_id: int) -> List[Tuple[str, int]]:
        """
        加载指定客户端的水印位置
        
        Args:
            client_id: 客户端ID
            
        Returns:
            位置列表 [(param_name, param_idx), ...]
        """
        if client_id < 0 or client_id >= self.client_num:
            raise ValueError(f"客户端ID {client_id} 超出范围 [0, {self.client_num-1}]")
        
        client_dir = os.path.join(self.key_matrix_dir, f'client_{client_id}')
        position_path = os.path.join(client_dir, 'positions.json')
        
        if not os.path.exists(position_path):
            raise FileNotFoundError(f"客户端 {client_id} 的位置文件不存在: {position_path}")
        
        with open(position_path, 'r') as f:
            return json.load(f)
    
    def get_watermark_positions(self, client_id: int) -> List[Tuple[str, int]]:
        """获取客户端的水印位置（兼容旧接口）"""
        return self.load_positions(client_id)
    
    def embed_watermark(self, model_params: Dict[str, torch.Tensor], 
                       client_id: int, watermark_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将水印嵌入到模型参数中
        
        Args:
            model_params: 模型参数字典
            client_id: 客户端ID
            watermark_values: 水印值
            
        Returns:
            嵌入水印后的模型参数
        """
        key_matrix = self.load_key_matrix(client_id)
        positions = self.load_positions(client_id)
        
        # 复制模型参数
        watermarked_params = {}
        for name, param in model_params.items():
            watermarked_params[name] = param.clone()
        
        # 嵌入水印
        watermark_idx = 0
        for param_name, param_idx in positions:
            if param_name in watermarked_params:
                # 将一维索引转换为多维索引
                param_shape = watermarked_params[param_name].shape
                multi_idx = torch.unravel_index(torch.tensor(param_idx), param_shape)
                
                # 设置水印值
                if watermark_idx < len(watermark_values):
                    watermarked_params[param_name][multi_idx] = watermark_values[watermark_idx]
                    watermark_idx += 1
        
        return watermarked_params
    
    def extract_watermark(self, model_params: Dict[str, torch.Tensor], 
                         client_id: int) -> torch.Tensor:
        """
        从模型参数中提取水印
        
        Args:
            model_params: 模型参数字典
            client_id: 客户端ID
            
        Returns:
            提取的水印值
        """
        positions = self.load_positions(client_id)
        
        watermark_values = []
        for param_name, param_idx in positions:
            if param_name in model_params:
                # 将一维索引转换为多维索引
                param_shape = model_params[param_name].shape
                multi_idx = torch.unravel_index(torch.tensor(param_idx), param_shape)
                
                # 提取水印值
                watermark_values.append(model_params[param_name][multi_idx].item())
        
        return torch.tensor(watermark_values)
    
    def get_info(self) -> dict:
        """获取密钥矩阵信息"""
        return self.info.copy()
    
    def list_clients(self) -> List[int]:
        """列出所有可用的客户端ID"""
        return list(range(self.client_num))
    
    def verify_key_matrices(self) -> Dict[int, bool]:
        """
        验证所有密钥矩阵的完整性
        
        Returns:
            每个客户端的验证结果
        """
        results = {}
        
        for client_id in range(self.client_num):
            try:
                key_matrix = self.load_key_matrix(client_id)
                positions = self.load_positions(client_id)
                
                # 验证密钥矩阵中1的数量与位置数量一致
                total_ones = sum(tensor.sum().item() for tensor in key_matrix.values())
                expected_ones = len(positions)
                
                results[client_id] = total_ones == expected_ones
                
                if not results[client_id]:
                    print(f"警告: 客户端 {client_id} 的密钥矩阵验证失败")
                    print(f"  期望的1数量: {expected_ones}, 实际的1数量: {int(total_ones)}")
                    
            except Exception as e:
                print(f"错误: 客户端 {client_id} 验证失败: {e}")
                results[client_id] = False
        
        return results

def load_key_matrix_manager(key_matrix_dir: str) -> KeyMatrixManager:
    """
    便捷函数：加载密钥矩阵管理器
    
    Args:
        key_matrix_dir: 密钥矩阵保存目录
        
    Returns:
        KeyMatrixManager实例
    """
    return KeyMatrixManager(key_matrix_dir)

# 兼容性函数
def construct_key_matrices_from_saved(key_matrix_dir: str) -> Dict[int, List[Tuple[str, int]]]:
    """
    从保存的密钥矩阵中构造位置字典（兼容旧接口）
    
    Args:
        key_matrix_dir: 密钥矩阵保存目录
        
    Returns:
        位置字典 {client_id: positions}
    """
    manager = KeyMatrixManager(key_matrix_dir)
    position_dict = {}
    
    for client_id in range(manager.client_num):
        positions = manager.load_positions(client_id)
        # 确保位置信息是元组格式（JSON会将元组转换为列表）
        tuple_positions = []
        for pos in positions:
            if isinstance(pos, list):
                tuple_positions.append(tuple(pos))
            else:
                tuple_positions.append(pos)
        position_dict[client_id] = tuple_positions
    
    return position_dict
