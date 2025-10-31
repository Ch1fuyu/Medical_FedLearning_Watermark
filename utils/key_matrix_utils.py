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
                       client_id: int, watermark_values: torch.Tensor, model=None) -> Dict[str, torch.Tensor]:
        """
        将水印嵌入到模型参数中
        自动检测并转换全局索引为局部索引（兼容旧的位置文件）
        
        Args:
            model_params: 模型参数字典
            client_id: 客户端ID
            watermark_values: 水印值
            model: 模型对象（可选），如果提供，将使用 named_parameters() 确保顺序与密钥矩阵生成时一致
        
        Returns:
            嵌入水印后的模型参数
        """
        key_matrix = self.load_key_matrix(client_id)
        positions = self.load_positions(client_id)
        
        # 检查是否需要构建全局偏移映射（用于转换全局索引）
        needs_conversion = False
        param_offset_map = {}
        
        # 先检查是否需要转换
        for param_name, idx in positions:
            if param_name in model_params:
                param_size = model_params[param_name].numel()
                if idx >= param_size:
                    needs_conversion = True
                    break
        
        # 如果需要转换，构建参数偏移映射
        if needs_conversion:
            current_offset = 0
            # 只处理卷积层参数（与 train_key_matrix.py 保持一致）
            # 如果提供了模型对象，使用 named_parameters() 确保顺序一致
            # 否则使用参数名排序（可能不完全准确，但兼容性更好）
            if model is not None:
                # 使用模型对象的 named_parameters() 确保顺序与 train_key_matrix.py 一致
                param_iter = model.named_parameters()
            else:
                # 回退到字典顺序（按名称排序）
                param_iter = sorted(model_params.items(), key=lambda x: x[0])
            
            for name, param in param_iter:
                # 确保参数存在于 model_params 中
                if name not in model_params:
                    continue
                
                # 使用 model_params 中的参数（可能已被修改），但保持顺序一致
                param = model_params[name]
                
                # 检查是否为卷积层参数：1. 包含 'conv' 和 'weight'
                # 2. 或包含 'downsample.0.weight' (ResNet 的 1x1 卷积) 3. 参数维度为 4D (卷积层权重)
                is_conv_weight = (
                    'conv' in name.lower() and 'weight' in name.lower()
                ) or (
                    'downsample.0.weight' in name.lower()  # ResNet downsample conv
                )
                
                # 还必须是4D张量（卷积层权重的形状）
                if is_conv_weight and len(param.shape) == 4:
                    param_offset_map[name] = current_offset
                    current_offset += param.numel()
        
        # 复制模型参数
        watermarked_params = {}
        for name, param in model_params.items():
            watermarked_params[name] = param.clone()
        
        # 嵌入水印到各个参数中（自动处理全局/局部索引）
        watermark_idx = 0
        for param_name_in_file, idx in positions:
            if watermark_idx >= len(watermark_values):
                break
                
            # 如果使用全局索引，需要根据全局索引找到正确的参数
            if needs_conversion:
                # 首先尝试从位置文件中的参数名获取
                # 如果索引在参数范围内，使用该参数
                if param_name_in_file in watermarked_params:
                    param_size = watermarked_params[param_name_in_file].numel()
                    # 如果索引在参数范围内，说明是局部索引
                    if idx < param_size:
                        actual_param_name = param_name_in_file
                        local_idx = idx
                    # 如果索引不在参数范围内，说明是全局索引，需要找到正确的参数
                    elif param_name_in_file in param_offset_map:
                        param_offset = param_offset_map[param_name_in_file]
                        # 检查全局索引是否在这个参数的范围内
                        if param_offset <= idx < param_offset + param_size:
                            actual_param_name = param_name_in_file
                            local_idx = idx - param_offset
                        else:
                            # 全局索引不在文件指定的参数中，需要遍历找到正确的参数
                            actual_param_name = None
                            for name, offset in param_offset_map.items():
                                param_size_check = watermarked_params[name].numel()
                                if offset <= idx < offset + param_size_check:
                                    actual_param_name = name
                                    local_idx = idx - offset
                                    break
                            if actual_param_name is None:
                                raise IndexError(
                                    f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                                )
                    else:
                        # 文件中的参数名不在偏移映射中（可能是非卷积层），需要遍历找到正确的参数
                        actual_param_name = None
                        for name, offset in param_offset_map.items():
                            param_size_check = watermarked_params[name].numel()
                            if offset <= idx < offset + param_size_check:
                                actual_param_name = name
                                local_idx = idx - offset
                                break
                        if actual_param_name is None:
                            raise IndexError(
                                f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                            )
                else:
                    # 文件中的参数名不存在，需要根据全局索引找到正确的参数
                    actual_param_name = None
                    for name, offset in param_offset_map.items():
                        param_size_check = watermarked_params[name].numel()
                        if offset <= idx < offset + param_size_check:
                            actual_param_name = name
                            local_idx = idx - offset
                            break
                    if actual_param_name is None:
                        raise IndexError(
                            f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                        )
            else:
                # 使用局部索引，直接使用文件中的参数名
                actual_param_name = param_name_in_file
                local_idx = idx
            
            # 现在嵌入水印值
            if actual_param_name in watermarked_params:
                param_tensor = watermarked_params[actual_param_name].view(-1)  # 扁平化参数
                param_size = param_tensor.numel()
                
                # 验证索引范围
                if local_idx < 0 or local_idx >= param_size:
                    raise IndexError(
                        f"错误: 局部索引 {local_idx} 超出参数 {actual_param_name} 的范围 "
                        f"[0, {param_size})。全局索引: {idx}。"
                    )
                
                param_tensor[local_idx] = watermark_values[watermark_idx]
                watermark_idx += 1
            else:
                raise KeyError(
                    f"错误: 参数名 '{actual_param_name}' 不在模型参数中。"
                )
        
        return watermarked_params
    
    def _build_param_offset_map(self, model_params: Dict[str, torch.Tensor], model=None) -> Dict[str, int]:
        """
        构建参数全局偏移映射（用于将全局索引转换为局部索引）
        
        Args:
            model_params: 模型参数字典
            model: 模型对象（可选，用于确保顺序与密钥矩阵生成时一致）
            
        Returns:
            每个参数的全局起始偏移量字典
        """
        offset_map = {}
        current_offset = 0
        
        # 如果提供了模型，使用 named_parameters() 来确保顺序与密钥矩阵生成时一致
        # 否则使用字典顺序（可能不一致，但兼容性更好）
        if model is not None:
            param_iter = model.named_parameters()
        else:
            param_iter = model_params.items()
        
        # 按照参数出现的顺序（应该是与密钥矩阵生成时一致）
        # 只考虑卷积层参数（与密钥矩阵生成逻辑一致）
        for name, param in param_iter:
            # 确保使用模型参数而不是字典中的参数（可能已被修改）
            if name not in model_params:
                continue
            
            # 检查是否为卷积层参数（与密钥矩阵生成逻辑一致）
            is_conv_weight = (
                'conv' in name.lower() and 'weight' in name.lower()
            ) or (
                'downsample.0.weight' in name.lower()  # ResNet downsample conv
            )
            
            if is_conv_weight and len(model_params[name].shape) == 4:
                offset_map[name] = current_offset
                current_offset += model_params[name].numel()
        
        return offset_map
    
    def _convert_global_to_local_index(self, global_idx: int, param_name: str, 
                                       offset_map: Dict[str, int], 
                                       model_params: Dict[str, torch.Tensor]) -> int:
        """
        将全局索引转换为局部索引
        
        Args:
            global_idx: 全局索引
            param_name: 参数名
            offset_map: 参数偏移映射
            model_params: 模型参数字典
            
        Returns:
            局部索引
        """
        if param_name not in offset_map:
            raise KeyError(f"参数 {param_name} 不在偏移映射中")
        
        param_offset = offset_map[param_name]
        local_idx = global_idx - param_offset
        
        # 验证转换后的索引是否有效
        param_size = model_params[param_name].numel()
        if local_idx < 0 or local_idx >= param_size:
            raise IndexError(
                f"全局索引 {global_idx} 转换为局部索引 {local_idx} 后超出范围 "
                f"(参数 {param_name} 大小: {param_size}, 偏移: {param_offset})"
            )
        
        return local_idx
    
    def extract_watermark(self, model_params: Dict[str, torch.Tensor], 
                         client_id: int, check_pruning: bool = False, model=None) -> torch.Tensor:
        """
        从模型参数中提取水印（使用局部索引）
        自动检测并转换全局索引为局部索引（兼容旧的位置文件）
        
        Args:
            model_params: 模型参数字典
            client_id: 客户端ID
            check_pruning: 是否检查剪枝对水印的影响
            model: 模型对象（可选），如果提供，将使用 named_parameters() 确保顺序与密钥矩阵生成时一致
            
        Returns:
            提取的水印值
        """
        positions = self.load_positions(client_id)
        
        # 检查是否需要构建全局偏移映射（用于转换全局索引）
        # 如果任何索引值大于对应参数的大小，说明是全局索引
        needs_conversion = False
        param_offset_map = {}
        
        # 先检查是否需要转换
        for param_name, idx in positions:
            if param_name in model_params:
                param_size = model_params[param_name].numel()
                if idx >= param_size:
                    needs_conversion = True
                    break
        
        # 如果需要转换，构建参数偏移映射
        if needs_conversion:
            current_offset = 0
            # 只处理卷积层参数（与 train_key_matrix.py 保持一致）
            # 如果提供了模型对象，使用 named_parameters() 确保顺序一致
            # 否则使用参数名排序（可能不完全准确，但兼容性更好）
            if model is not None:
                # 使用模型对象的 named_parameters() 确保顺序与 train_key_matrix.py 一致
                param_iter = model.named_parameters()
            else:
                # 回退到字典顺序（按名称排序）
                param_iter = sorted(model_params.items(), key=lambda x: x[0])
            
            for name, param in param_iter:
                # 确保参数存在于 model_params 中
                if name not in model_params:
                    continue
                
                # 使用 model_params 中的参数（可能已被修改），但保持顺序一致
                param = model_params[name]
                
                # 检查是否为卷积层参数：1. 包含 'conv' 和 'weight'
                # 2. 或包含 'downsample.0.weight' (ResNet 的 1x1 卷积) 3. 参数维度为 4D (卷积层权重)
                is_conv_weight = (
                    'conv' in name.lower() and 'weight' in name.lower()
                ) or (
                    'downsample.0.weight' in name.lower()  # ResNet downsample conv
                )
                
                # 还必须是4D张量（卷积层权重的形状）
                if is_conv_weight and len(param.shape) == 4:
                    param_offset_map[name] = current_offset
                    current_offset += param.numel()
        
        watermark_values = []
        for param_name_in_file, idx in positions:
            # 如果使用全局索引，需要根据全局索引找到正确的参数
            if needs_conversion:
                # 首先尝试从位置文件中的参数名获取
                # 如果索引在参数范围内，使用该参数
                if param_name_in_file in model_params:
                    param_size = model_params[param_name_in_file].numel()
                    # 如果索引在参数范围内，说明是局部索引
                    if idx < param_size:
                        actual_param_name = param_name_in_file
                        local_idx = idx
                    # 如果索引不在参数范围内，说明是全局索引，需要找到正确的参数
                    elif param_name_in_file in param_offset_map:
                        param_offset = param_offset_map[param_name_in_file]
                        # 检查全局索引是否在这个参数的范围内
                        if param_offset <= idx < param_offset + param_size:
                            actual_param_name = param_name_in_file
                            local_idx = idx - param_offset
                        else:
                            # 全局索引不在文件指定的参数中，需要遍历找到正确的参数
                            actual_param_name = None
                            for name, offset in param_offset_map.items():
                                param_size_check = model_params[name].numel()
                                if offset <= idx < offset + param_size_check:
                                    actual_param_name = name
                                    local_idx = idx - offset
                                    break
                            if actual_param_name is None:
                                raise IndexError(
                                    f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                                )
                    else:
                        # 文件中的参数名不在偏移映射中（可能是非卷积层），需要遍历找到正确的参数
                        actual_param_name = None
                        for name, offset in param_offset_map.items():
                            param_size_check = model_params[name].numel()
                            if offset <= idx < offset + param_size_check:
                                actual_param_name = name
                                local_idx = idx - offset
                                break
                        if actual_param_name is None:
                            raise IndexError(
                                f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                            )
                else:
                    # 文件中的参数名不存在，需要根据全局索引找到正确的参数
                    actual_param_name = None
                    for name, offset in param_offset_map.items():
                        param_size_check = model_params[name].numel()
                        if offset <= idx < offset + param_size_check:
                            actual_param_name = name
                            local_idx = idx - offset
                            break
                    if actual_param_name is None:
                        raise IndexError(
                            f"错误: 全局索引 {idx} 无法映射到任何卷积层参数。"
                        )
            else:
                # 使用局部索引，直接使用文件中的参数名
                actual_param_name = param_name_in_file
                local_idx = idx
            
            # 现在提取水印值
            if actual_param_name in model_params:
                param_tensor = model_params[actual_param_name].view(-1)  # 扁平化参数
                param_size = param_tensor.numel()
                
                # 验证索引范围
                if local_idx < 0 or local_idx >= param_size:
                    raise IndexError(
                        f"错误: 局部索引 {local_idx} 超出参数 {actual_param_name} 的范围 "
                        f"[0, {param_size})。全局索引: {idx}。"
                    )
                
                watermark_value = param_tensor[local_idx]  # 保持tensor格式，避免精度损失
                
                # 如果启用剪枝检查，检测水印位置是否被剪掉
                if check_pruning:
                    # 检查参数是否被剪枝（完全等于0）
                    if watermark_value.item() == 0.0:
                        # 记录被剪枝的位置，但不修改值
                        pass  # 不输出详细信息
                
                watermark_values.append(watermark_value)
            else:
                # 参数名不存在，直接抛出异常并退出
                raise KeyError(
                    f"错误: 参数名 '{actual_param_name}' 不在模型参数中。这通常表示模型结构发生了变化，"
                    f"无法正确提取水印。请确保使用正确的模型结构。"
                )
        
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

