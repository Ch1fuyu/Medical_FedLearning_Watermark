import torch
import json
import os
from typing import Dict, List, Tuple, Optional

def get_key_matrix_path(base_dir: str, model_type: str, client_num: int) -> str:
    """
    根据模型类型和客户端数量生成密钥矩阵路径
    
    Args:
        base_dir: 基础目录
        model_type: 模型类型 (resnet, alexnet)
        client_num: 客户端数量
        
    Returns:
        密钥矩阵目录路径
    """
    return os.path.join(base_dir, model_type, f'client{client_num}').replace('\\', '/')

def find_key_matrix_path(base_dir: str, model_type: str, client_num: int) -> Optional[str]:
    """
    查找密钥矩阵路径，如果不存在则返回None
    
    Args:
        base_dir: 基础目录
        model_type: 模型类型 (resnet, alexnet)
        client_num: 客户端数量
        
    Returns:
        密钥矩阵目录路径，如果不存在则返回None
    """
    key_matrix_path = get_key_matrix_path(base_dir, model_type, client_num)
    if os.path.exists(key_matrix_path):
        return key_matrix_path
    return None

class KeyMatrixManager:
    """密钥矩阵管理器，用于加载和管理密钥矩阵（支持实例缓存）"""
    
    _instances = {}  # 类变量，存储不同配置的实例
    
    def __new__(cls, key_matrix_dir: str, args=None):
        """
        单例模式：相同配置只创建一个实例
        
        Args:
            key_matrix_dir: 密钥矩阵保存目录
            args: 参数对象（保留用于兼容性，但不再使用）
        """
        instance_key = key_matrix_dir
        
        if instance_key not in cls._instances:
            instance = super(KeyMatrixManager, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, key_matrix_dir: str, args=None):
        """
        初始化密钥矩阵管理器
        
        Args:
            key_matrix_dir: 密钥矩阵保存目录
            args: 参数对象（保留用于兼容性，但不再使用）
        """
        # 避免重复初始化
        if hasattr(self, 'key_matrix_dir'):
            return
            
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
            positions = json.load(f)
        
        # 将列表格式转换为元组格式
        return [(pos[0], pos[1]) for pos in positions]
    
    
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
        
        # 嵌入水印到各个参数中
        watermark_idx = 0
        for param_name, param_idx in positions:
            if param_name in watermarked_params and watermark_idx < len(watermark_values):
                # param_idx 是局部索引，直接使用
                param_tensor = watermarked_params[param_name].view(-1)  # 扁平化参数
                
                if param_idx < param_tensor.numel():
                    param_tensor[param_idx] = watermark_values[watermark_idx]
                    watermark_idx += 1
        
        return watermarked_params
    
    def extract_watermark(self, model_params: Dict[str, torch.Tensor], 
                         client_id: int, check_pruning: bool = False) -> torch.Tensor:
        """
        从模型参数中提取水印
        
        Args:
            model_params: 模型参数字典
            client_id: 客户端ID
            check_pruning: 是否检查剪枝对水印的影响
            
        Returns:
            提取的水印值
        """
        positions = self.load_positions(client_id)
        
        watermark_values = []
        for param_name, param_idx in positions:
            if param_name in model_params:
                # param_idx 是局部索引，直接使用
                param_tensor = model_params[param_name].view(-1)  # 扁平化参数
                
                if param_idx < param_tensor.numel():
                    watermark_value = param_tensor[param_idx]  # 保持tensor格式，避免精度损失
                    
                    # 如果启用剪枝检查，检测水印位置是否被剪掉
                    if check_pruning:
                        # 检查参数是否被剪枝（完全等于0）
                        if watermark_value.item() == 0.0:
                            # 记录被剪枝的位置，但不修改值
                            pass  # 不输出详细信息
                    
                    watermark_values.append(watermark_value)
                else:
                    print(f"警告: 局部索引 {param_idx} 超出参数 {param_name} 的范围 {param_tensor.numel()}")
                    watermark_values.append(torch.tensor(0.0, device=param_tensor.device, dtype=param_tensor.dtype))
            else:
                print(f"警告: 参数名 {param_name} 不在模型参数中")
                # 需要从现有参数获取设备和数据类型
                if watermark_values:
                    device = watermark_values[0].device
                    dtype = watermark_values[0].dtype
                else:
                    device = torch.device('cpu')
                    dtype = torch.float32
                watermark_values.append(torch.tensor(0.0, device=device, dtype=dtype))
        
        # 直接堆叠tensor，避免精度损失
        watermark_tensor = torch.stack(watermark_values)
        
        return watermark_tensor
    
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
        return {key: f"KeyMatrixManager(dir={key})" 
                for key in cls._instances.keys()}

def load_key_matrix_manager(key_matrix_dir: str, args=None) -> KeyMatrixManager:
    """
    便捷函数：加载密钥矩阵管理器
    
    Args:
        key_matrix_dir: 密钥矩阵保存目录
        args: 参数对象（保留用于兼容性，但不再使用）
        
    Returns:
        KeyMatrixManager实例
    """
    return KeyMatrixManager(key_matrix_dir, args)

