import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
import argparse

from models.light_autoencoder import LightAutoencoder
from models.resnet import resnet18
from models.alexnet import AlexNet
from config.globals import set_seed

# 设置随机种子
set_seed()

# 数据集预设配置
DATASET_PRESETS = {
    'cifar10': {
        'num_classes': 10,
        'in_channels': 3,
        'input_size': 32,
    },
    'cifar100': {
        'num_classes': 100,
        'in_channels': 3,
        'input_size': 32,
    },
    'imagenet': {
        'num_classes': 1000,
        'in_channels': 3,
        'input_size': 224,
    },
}

class KeyMatrixGenerator:
    """密钥矩阵生成器"""
    
    def __init__(self, model, client_num, watermark_strategy='equal', seed=42):
        """
        初始化密钥矩阵生成器
        
        Args:
            model: 主任务模型
            client_num: 客户端数量
            watermark_strategy: 水印分配策略 ('equal', 'proportional', 'custom')
            seed: 随机种子
        """
        self.model = model
        self.client_num = client_num
        self.watermark_strategy = watermark_strategy
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 获取编码器参数数量作为总水印大小
        self.encoder = LightAutoencoder().encoder
        self.encoder_params = sum(param.numel() for param in self.encoder.parameters())
        
        # 获取主任务模型参数信息
        self.model_param_info = self._get_model_param_info()
        self.total_model_params = sum(param_info['numel'] for param_info in self.model_param_info)
        
        # 根据水印比例计算实际水印大小（使用编码器参数作为水印总量）
        self.total_watermark_size = self.encoder_params
        
        print(f"主任务模型总参数数量: {self.total_model_params:,}")
        print(f"编码器参数数量: {self.encoder_params:,}")
        print(f"水印大小: {self.total_watermark_size:,}")
    
    def _get_model_param_info(self):
        """获取模型参数信息（仅包含卷积层参数）"""
        param_info = []
        start_idx = 0
        for name, param in self.model.named_parameters():
            # 只包含卷积层参数
            if not ('conv' in name.lower() and 'weight' in name.lower()):
                print(f"跳过非卷积层参数: {name} (shape: {param.shape})")
                continue
                
            param_info.append({
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'start_idx': start_idx
            })
            start_idx += param.numel()
        return param_info
    
    def _calculate_watermark_sizes(self):
        """计算每个客户端的水印大小"""
        if self.watermark_strategy == 'equal':
            # 平均分配
            base_size = self.total_watermark_size // self.client_num
            remainder = self.total_watermark_size % self.client_num
            
            sizes = [base_size] * self.client_num
            # 将余数分配给前几个客户端
            for i in range(remainder):
                sizes[i] += 1
                
        elif self.watermark_strategy == 'proportional':
            # 按比例分配（这里简化为平均分配，可根据需要扩展）
            sizes = [self.total_watermark_size // self.client_num] * self.client_num
            remainder = self.total_watermark_size % self.client_num
            for i in range(remainder):
                sizes[i] += 1
                
        else:
            raise ValueError(f"不支持的水印分配策略: {self.watermark_strategy}")
        
        return sizes
    
    def generate_key_matrices(self):
        """生成所有客户端的密钥矩阵"""
        watermark_sizes = self._calculate_watermark_sizes()
        
        # 获取所有可用的参数位置
        all_positions = []
        for param_info in self.model_param_info:
            for i in range(param_info['numel']):
                all_positions.append((param_info['name'], i))
        
        # 随机打乱位置
        np.random.shuffle(all_positions)
        
        # 为每个客户端分配位置
        key_matrices = {}
        position_dict = {}
        start_idx = 0
        
        for client_id in range(self.client_num):
            watermark_size = watermark_sizes[client_id]
            end_idx = start_idx + watermark_size
            
            # 获取该客户端的水印位置
            client_positions = all_positions[start_idx:end_idx]
            position_dict[client_id] = client_positions
            
            # 生成密钥矩阵（与模型形状相同，但只有指定位置为1）
            key_matrix = self._create_key_matrix(client_positions)
            key_matrices[client_id] = key_matrix
            
            print(f"客户端 {client_id}: 水印大小 {watermark_size}, 位置范围 {start_idx}-{end_idx-1}")
            start_idx = end_idx
        
        return key_matrices, position_dict
    
    def _create_key_matrix(self, positions):
        """为指定位置创建密钥矩阵"""
        key_matrix = {}
        
        # 初始化所有参数为0
        for param_info in self.model_param_info:
            key_matrix[param_info['name']] = torch.zeros(param_info['shape'])
        
        # 在指定位置设置为1
        for param_name, param_idx in positions:
            # 将一维索引转换为多维索引
            param_shape = key_matrix[param_name].shape
            multi_idx = np.unravel_index(param_idx, param_shape)
            key_matrix[param_name][multi_idx] = 1.0
        
        return key_matrix
    
    def save_key_matrices(self, key_matrices, position_dict, save_dir, model_type, client_num):
        """保存密钥矩阵到文件"""
        # 创建新的目录结构: save_dir/model_type/client{client_num}/
        model_dir = os.path.join(save_dir, model_type)
        client_dir = os.path.join(model_dir, f'client{client_num}')
        os.makedirs(client_dir, exist_ok=True)
        
        # 保存每个客户端的密钥矩阵
        for client_id, key_matrix in key_matrices.items():
            client_subdir = os.path.join(client_dir, f'client_{client_id}')
            os.makedirs(client_subdir, exist_ok=True)
            
            # 保存密钥矩阵
            key_matrix_path = os.path.join(client_subdir, 'key_matrix.pth')
            torch.save(key_matrix, key_matrix_path)
            
            # 保存位置信息
            position_path = os.path.join(client_subdir, 'positions.json')
            with open(position_path, 'w') as f:
                json.dump(position_dict[client_id], f, indent=2)
        
        # 保存总体信息
        info = {
            'client_num': self.client_num,
            'watermark_strategy': self.watermark_strategy,
            'total_watermark_size': self.total_watermark_size,
            'total_model_params': self.total_model_params,
            'encoder_params': self.encoder_params,
            'model_type': self.model.__class__.__name__,
            'created_time': datetime.now().isoformat(),
            'seed': self.seed
        }
        
        info_path = os.path.join(client_dir, 'key_matrix_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"密钥矩阵已保存到: {client_dir}")
        return info_path

def load_key_matrix(save_dir, client_id):
    """加载指定客户端的密钥矩阵"""
    client_dir = os.path.join(save_dir, f'client_{client_id}')
    key_matrix_path = os.path.join(client_dir, 'key_matrix.pth')
    
    if not os.path.exists(key_matrix_path):
        raise FileNotFoundError(f"客户端 {client_id} 的密钥矩阵不存在: {key_matrix_path}")
    
    return torch.load(key_matrix_path, map_location='cpu')

def load_key_matrix_info(save_dir):
    """加载密钥矩阵信息"""
    info_path = os.path.join(save_dir, 'key_matrix_info.json')
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"密钥矩阵信息文件不存在: {info_path}")
    
    with open(info_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='生成联邦学习密钥矩阵')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=['resnet', 'alexnet'], help='主任务模型类型')
    parser.add_argument('--client_num', type=int, default=5, help='客户端数量')
    parser.add_argument('--watermark_strategy', type=str, default='equal',
                       choices=['equal', 'proportional'], help='水印分配策略')
    parser.add_argument('--save_dir', type=str, default='./save/key_matrix', 
                       help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 
                               'cifar10', 'cifar100', 'imagenet'],
                       help='数据集类型')
    
    args = parser.parse_args()
    
    # 使用预设配置获取数据集参数
    dataset_key = args.dataset.lower()
    if dataset_key in DATASET_PRESETS:
        dataset_config = DATASET_PRESETS[dataset_key]
        num_classes = dataset_config['num_classes']
        in_channels = dataset_config['in_channels']
        input_size = dataset_config['input_size']
    else:
        # 回退到手动配置（用于MedMNIST数据集）
        if args.dataset == 'chestmnist':
            num_classes = 14
            in_channels = 1
            input_size = 28
        elif args.dataset == 'dermamnist':
            num_classes = 7
            in_channels = 3
            input_size = 28
        elif args.dataset == 'octmnist':
            num_classes = 4
            in_channels = 1
            input_size = 28
        elif args.dataset == 'pneumoniamnist':
            num_classes = 2
            in_channels = 1
            input_size = 28
        elif args.dataset == 'retinamnist':
            num_classes = 5
            in_channels = 3
            input_size = 28
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建模型
    if args.model_type == 'resnet':
        model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    elif args.model_type == 'alexnet':
        model = AlexNet(in_channels, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    print(f"使用模型: {args.model_type}")
    print(f"数据集: {args.dataset}")
    print(f"类别数: {num_classes}, 输入通道数: {in_channels}, 输入尺寸: {input_size}")
    print(f"客户端数量: {args.client_num}")
    print(f"水印分配策略: {args.watermark_strategy}")
    
    # 生成密钥矩阵
    generator = KeyMatrixGenerator(
        model=model,
        client_num=args.client_num,
        watermark_strategy=args.watermark_strategy,
        seed=args.seed
    )
    
    key_matrices, position_dict = generator.generate_key_matrices()
    
    # 保存密钥矩阵
    info_path = generator.save_key_matrices(key_matrices, position_dict, args.save_dir, args.model_type, args.client_num)
    
    # 显示统计信息
    print("\n=== 密钥矩阵生成完成 ===")
    print(f"保存目录: {args.save_dir}")
    print(f"信息文件: {info_path}")
    
    # 验证密钥矩阵
    print("\n=== 验证密钥矩阵 ===")
    for client_id in range(args.client_num):
        key_matrix = key_matrices[client_id]
        total_ones = sum(tensor.sum().item() for tensor in key_matrix.values())
        print(f"客户端 {client_id}: 密钥矩阵中1的数量 = {int(total_ones)}")
    
    print("\n密钥矩阵生成完成！")

if __name__ == '__main__':
    main()
