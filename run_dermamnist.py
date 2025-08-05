#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DermaMNIST联邦学习运行脚本
"""

import os
import sys
import argparse
from main import main as fl_main
from utils.args import parser_args

def run_dermamnist_experiment():
    """运行DermaMNIST联邦学习实验"""
    
    # 创建参数解析器
    parser = parser_args()
    
    # 设置DermaMNIST特定的参数
    args = parser.parse_args([])  # 空列表表示使用默认参数
    
    # 修改为DermaMNIST配置
    args.dataset = 'dermamnist'
    args.num_classes = 7
    args.in_channels = 3
    args.client_num = 5  # 减少客户端数量，因为DermaMNIST数据量较小
    args.batch_size = 8  # 减小批次大小
    args.local_ep = 3    # 增加本地训练轮数
    args.epochs = 50     # 增加通信轮数
    args.lr = 0.01       # 学习率
    args.frac = 1.0      # 所有客户端参与
    args.iid = True      # 使用IID数据分布
    args.dp = False      # 暂时关闭差分隐私
    args.sigma = 0.1
    
    # 水印参数
    args.num_sign = 1
    args.weight_type = 'gamma'
    args.num_bit = 20
    args.loss_type = 'sign'
    args.loss_alpha = 0.2
    args.num_back = 1
    args.num_trigger = 40
    args.backdoor_indis = True
    
    print("=== DermaMNIST联邦学习实验配置 ===")
    print(f"数据集: {args.dataset}")
    print(f"类别数: {args.num_classes}")
    print(f"客户端数: {args.client_num}")
    print(f"批次大小: {args.batch_size}")
    print(f"本地训练轮数: {args.local_ep}")
    print(f"通信轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"数据分布: {'IID' if args.iid else 'Non-IID'}")
    print(f"差分隐私: {'开启' if args.dp else '关闭'}")
    print(f"水印类型: {args.weight_type}")
    print(f"水印位数: {args.num_bit}")
    print("=" * 40)
    
    # 运行实验
    try:
        fl_main(args)
        print("🎉 DermaMNIST联邦学习实验完成！")
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        raise

def run_dermamnist_no_fl():
    """运行DermaMNIST非联邦学习实验（对比）"""
    
    print("=== 运行DermaMNIST非联邦学习实验 ===")
    
    # 这里可以添加非联邦学习的对比实验
    # 使用no_fl目录下的脚本
    pass

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DermaMNIST联邦学习实验')
    parser.add_argument('--mode', type=str, default='fl', 
                       choices=['fl', 'no_fl'], 
                       help='运行模式: fl(联邦学习) 或 no_fl(非联邦学习)')
    
    args = parser.parse_args()
    
    if args.mode == 'fl':
        run_dermamnist_experiment()
    elif args.mode == 'no_fl':
        run_dermamnist_no_fl()
    else:
        print("无效的运行模式") 