#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DermaMNIST数据集集成测试脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import get_data, get_data_no_fl
from models.alexnet import AlexNet
from utils.sampling import dermamnist_iid, dermamnist_beta
from medmnist import INFO
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def test_dermamnist_loading():
    """测试DermaMNIST数据加载"""
    print("=== 测试DermaMNIST数据加载 ===")
    
    try:
        # 测试联邦学习数据加载
        train_set, test_set, dict_users = get_data(
            dataset_name='dermamnist',
            data_root='./data',
            iid=True,
            client_num=5
        )
        info = INFO['dermamnist']
        num_classes = len(info['label'])
        print(f"✓ 联邦学习数据加载成功")
        print(f"  训练集大小: {len(train_set)}")
        print(f"  测试集大小: {len(test_set)}")
        print(f"  客户端数量: {len(dict_users)}")
        print(f"  类别数: {num_classes}")
        print(f"  通道数: {info['n_channels']}")
        print(f"  图像尺寸: 28x28")
        
        # 测试非联邦学习数据加载
        train_set_no_fl, test_set_no_fl = get_data_no_fl(
            dataset_name='dermamnist',
            data_root='./data'
        )
        print(f"✓ 非联邦学习数据加载成功")
        print(f"  训练集大小: {len(train_set_no_fl)}")
        print(f"  测试集大小: {len(test_set_no_fl)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_data_sampling():
    """测试数据采样"""
    print("\n=== 测试数据采样 ===")
    
    try:
        # 获取数据
        train_set, test_set, dict_users = get_data(
            dataset_name='dermamnist',
            data_root='./data',
            iid=True,
            client_num=5
        )
        info = INFO['dermamnist']
        num_classes = len(info['label'])
        # 测试IID采样
        iid_dict = dermamnist_iid(train_set, 5)
        print(f"✓ IID采样成功，客户端数量: {len(iid_dict)}")
        print(f"  类别数: {num_classes}")
        # 测试Non-IID采样
        non_iid_dict = dermamnist_beta(train_set, 0.1, 5)
        print(f"✓ Non-IID采样成功，客户端数量: {len(non_iid_dict)}")
        print(f"  类别数: {num_classes}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据采样失败: {e}")
        return False

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n=== 测试模型兼容性 ===")
    
    try:
        info = INFO['dermamnist']
        num_classes = len(info['label'])
        # 创建模型
        model = AlexNet(in_channels=3, num_classes=num_classes)
        print(f"✓ 模型创建成功")
        print(f"  输入通道数: 3")
        print(f"  输出类别数: {num_classes}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 28, 28)  # DermaMNIST图像尺寸
        output = model(dummy_input)
        print(f"✓ 前向传播成功")
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出形状: {output.shape}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数数量: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型兼容性测试失败: {e}")
        return False

def test_data_visualization():
    """测试数据可视化"""
    print("\n=== 测试数据可视化 ===")
    
    try:
        train_set, test_set, dict_users = get_data(
            dataset_name='dermamnist',
            data_root='./data',
            iid=True,
            client_num=5
        )
        info = INFO['dermamnist']
        num_classes = len(info['label'])
        # 创建数据加载器
        dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
        # 获取一批数据
        batch = next(iter(dataloader))
        images, labels = batch
        print(f"✓ 数据可视化测试成功")
        print(f"  批次大小: {images.shape}")
        print(f"  图像形状: {images[0].shape}")
        print(f"  标签形状: {labels.shape}")
        # 兼容标签类型
        label_min = int(labels.min().item()) if hasattr(labels.min(), 'item') else int(labels.min())
        label_max = int(labels.max().item()) if hasattr(labels.max(), 'item') else int(labels.max())
        print(f"  标签范围: {label_min} - {label_max}")
        print(f"  类别数: {num_classes}")
        # 显示图像
        plt.figure(figsize=(12, 6))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            img = images[i].permute(1, 2, 0)  # CHW -> HWC
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
            # 兼容标签类型
            label_val = int(labels[i].item()) if hasattr(labels[i], 'item') else int(labels[i])
            plt.imshow(img)
            plt.title(f'Label: {label_val}')
            plt.axis('off')
        plt.suptitle('DermaMNIST 样本图像')
        plt.tight_layout()
        plt.savefig('./save/image/dermamnist_samples.png', dpi=150, bbox_inches='tight')
        print(f"  样本图像已保存到: ./save/image/dermamnist_samples.png")
        return True
    except Exception as e:
        print(f"✗ 数据可视化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始DermaMNIST集成测试...\n")
    
    # 创建保存目录
    import os
    os.makedirs('./save/image', exist_ok=True)
    
    # 运行测试
    tests = [
        test_dermamnist_loading,
        test_data_sampling,
        test_model_compatibility,
        test_data_visualization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
    
    # 输出测试结果
    print("\n=== 测试结果汇总 ===")
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！DermaMNIST集成成功！")
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
    
    return passed == total

if __name__ == '__main__':
    main() 