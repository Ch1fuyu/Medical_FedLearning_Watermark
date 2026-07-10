import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from config.globals import set_seed
from models.alexnet import AlexNet
from models.light_autoencoder import LightAutoencoder
from models.resnet import resnet18

# 设置随机种子
set_seed()


class KeyMatrixGenerator:
    """密钥矩阵生成器"""
    
    def __init__(self, model, seed=42):
        """
        初始化密钥矩阵生成器
        
        Args:
            model: 主任务模型
            seed: 随机种子
        """
        self.model = model
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 获取编码器参数数量作为水印大小
        self.encoder = LightAutoencoder().encoder
        self.watermark_size = sum(param.numel() for param in self.encoder.parameters())
        
        # 获取主任务模型参数信息
        self.model_param_info = self._get_model_param_info()
        self.total_model_params = sum(param_info['numel'] for param_info in self.model_param_info)
        
        print(f"主任务模型卷积层参数数量: {self.total_model_params:,}")
        print(f"水印大小（编码器参数数量）: {self.watermark_size:,}")
    
    def _get_model_param_info(self):
        """获取模型参数信息（仅包含卷积层参数）"""
        param_info = []
        skipped_count = 0
        for name, param in self.model.named_parameters():
            is_conv_weight = (
                'conv' in name.lower() and 'weight' in name.lower()
            ) or (
                'downsample.0.weight' in name.lower()
            )
            
            if is_conv_weight and len(param.shape) == 4:
                param_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'numel': param.numel(),
                })
            else:
                skipped_count += 1
                
        if skipped_count > 0:
            print(f"已跳过 {skipped_count} 个非卷积层参数（BN、FC层等）")
        print(f"共提取 {len(param_info)} 个卷积层")
        return param_info
    
    def generate_key_matrix(self):
        """生成密钥矩阵"""
        # 获取所有可用的参数位置
        all_positions = []
        for param_info in self.model_param_info:
            for i in range(param_info['numel']):
                all_positions.append((param_info['name'], i))
        
        # 随机打乱位置
        np.random.shuffle(all_positions)
        
        # 选取水印位置
        watermark_positions = all_positions[:self.watermark_size]
        
        # 生成密钥矩阵
        key_matrix = self._create_key_matrix(watermark_positions)
        
        print(f"水印位置数量: {len(watermark_positions)}")
        
        return key_matrix, watermark_positions
    
    def _create_key_matrix(self, positions):
        """为指定位置创建密钥矩阵"""
        key_matrix = {}
        
        for param_info in self.model_param_info:
            key_matrix[param_info['name']] = torch.zeros(param_info['shape'])
        
        for param_name, local_idx in positions:
            if param_name not in key_matrix:
                continue
            
            param_info = None
            for info in self.model_param_info:
                if info['name'] == param_name:
                    param_info = info
                    break
            
            if param_info is None:
                continue
            
            if local_idx < 0 or local_idx >= param_info['numel']:
                continue
            
            param_shape = key_matrix[param_name].shape
            multi_idx = np.unravel_index(local_idx, param_shape)
            key_matrix[param_name][multi_idx] = 1.0
        
        return key_matrix
    
    def save_key_matrix(self, key_matrix, positions, save_dir, model_type):
        """保存密钥矩阵到文件"""
        save_subdir = os.path.join(save_dir, model_type)
        os.makedirs(save_subdir, exist_ok=True)
        
        # 保存密钥矩阵
        torch.save(key_matrix, os.path.join(save_subdir, 'key_matrix.pth'))
        
        # 保存位置信息
        with open(os.path.join(save_subdir, 'positions.json'), 'w') as f:
            json.dump(positions, f)
        
        # 保存信息
        info = {
            'watermark_size': self.watermark_size,
            'total_model_params': self.total_model_params,
            'model_type': self.model.__class__.__name__,
            'created_time': datetime.now().isoformat(),
            'seed': self.seed
        }
        with open(os.path.join(save_subdir, 'key_matrix_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"密钥矩阵已保存到: {save_subdir}")


def main():
    parser = argparse.ArgumentParser(description='生成水印密钥矩阵')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=['resnet', 'alexnet'], help='主任务模型类型')
    parser.add_argument('--save_dir', type=str, default='./save/key_matrix', 
                       help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--in_channels', type=int, default=1, help='模型输入通道数')
    parser.add_argument('--input_size', type=int, default=28, help='模型输入尺寸')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数')
    
    args = parser.parse_args()
    
    # 创建模型
    if args.model_type == 'resnet':
        model = resnet18(num_classes=args.num_classes, in_channels=args.in_channels, input_size=args.input_size)
    else:
        model = AlexNet(args.in_channels, args.num_classes, input_size=args.input_size, dropout_rate=0.5)
    
    print(f"使用模型: {args.model_type}")
    print(f"输入通道数: {args.in_channels}, 输入尺寸: {args.input_size}, 类别数: {args.num_classes}")
    
    # 生成密钥矩阵
    generator = KeyMatrixGenerator(model=model, seed=args.seed)
    key_matrix, positions = generator.generate_key_matrix()
    generator.save_key_matrix(key_matrix, positions, args.save_dir, args.model_type)
    
    # 验证
    total_ones = sum(tensor.sum().item() for tensor in key_matrix.values())
    print(f"密钥矩阵中1的数量: {int(total_ones)}")


if __name__ == '__main__':
    main()
