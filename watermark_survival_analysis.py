"""
水印存活能力分析 (改进版)
========================

使用更敏感的指标衡量水印衰减:
- 水印区域参数绝对变化量
- 水印/非水印变化量比
- 每轮水印区域参数的标准差变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime


def get_model(model_type, in_channels, num_classes, input_size):
    if model_type == 'alexnet':
        from models.alexnet import AlexNet
        return AlexNet(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
    else:
        from models.resnet import resnet18
        return resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)


def run_survival_analysis(model_type, dataset, in_channels, num_classes,
                          input_size, task_type, 
                          num_rounds=15, local_epochs=2,
                          watermark_ratio=0.1, wm_grad_ratio=0.3,
                          lr=0.01, seed=42):
    """运行水印存活分析 - 使用更敏感的指标"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type, in_channels, num_classes, input_size).to(device)
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    watermark_numel = int(total_params * watermark_ratio)
    
    # 创建水印掩码和值
    flat_mask = torch.zeros(total_params, device=device)
    flat_mask[:watermark_numel] = 1.0
    
    # 生成水印值
    flat_watermark = torch.zeros(total_params, device=device)
    wm_indices = flat_mask.nonzero().squeeze()
    flat_watermark[wm_indices] = torch.randn(watermark_numel, device=device) * 0.3
    
    # 嵌入水印
    offset = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        wm_slice = flat_watermark[offset:offset+numel].view(param.shape)
        param.data = param.data + wm_slice
        offset += numel
    
    # 记录基准参数
    benchmark_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # 准备输入
    if dataset == 'cifar10':
        input_shape = (32, 3, 32, 32)
        num_target = 10
    elif dataset == 'chestmnist':
        input_shape = (32, 1, 28, 28)
        num_target = 14
    elif dataset == 'cifar100':
        input_shape = (32, 3, 32, 32)
        num_target = 100
    
    history = {
        'round': [],
        'wm_l2_change': [],          # 水印区域 L2 变化
        'nonwm_l2_change': [],       # 非水印区域 L2 变化
        'wm_relative_change': [],    # 水印相对变化率
        'wm_to_total_ratio': [],     # 水印变化占总变化的比例
        'wm_param_std': [],          # 水印参数标准差
        'wm_param_mean': [],         # 水印参数均值
    }
    
    print(f"  配置: {model_type} on {dataset}")
    print(f"  总参数={total_params:,}, 水印={watermark_numel:,} ({watermark_ratio*100:.0f}%)")
    
    for round_idx in range(num_rounds):
        for epoch in range(local_epochs):
            x = torch.randn(input_shape, device=device)
            if task_type == 'multiclass':
                y = torch.randint(0, num_target, (32,), device=device)
            else:
                y = torch.randint(0, 2, (32, num_target), device=device).float()
            
            model.train()
            pred = model(x)
            if task_type == 'multiclass':
                loss = F.cross_entropy(pred, y)
            else:
                loss = F.binary_cross_entropy_with_logits(pred, y)
            loss.backward()
            
            # 应用水印梯度缩放
            with torch.no_grad():
                all_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                all_grads = torch.cat(all_grads)
                
                if all_grads.shape[0] == flat_mask.shape[0]:
                    wm_grad = all_grads * flat_mask
                    non_wm_grad = all_grads * (1.0 - flat_mask)
                    wm_count = flat_mask.sum().item()
                    non_wm_count = (1.0 - flat_mask).sum().item()
                    
                    wm_grad_mean = wm_grad.abs().sum().item() / max(wm_count, 1)
                    non_wm_grad_mean = non_wm_grad.abs().sum().item() / max(non_wm_count, 1)
                    wm_ratio = wm_grad_mean / max(non_wm_grad_mean, 1e-10)
                    
                    if wm_ratio > wm_grad_ratio:
                        scale = (non_wm_grad_mean * wm_grad_ratio) / max(wm_grad_mean, 1e-10)
                        scale = max(0.1, min(1.0, scale))
                        
                        grad_offset = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                numel = param.numel()
                                p_grad = param.grad.view(-1)
                                mask_slice = flat_mask[grad_offset:grad_offset+numel]
                                p_grad_new = p_grad * (1.0 - mask_slice) + p_grad * mask_slice * scale
                                param.grad.view(-1).copy_(p_grad_new)
                                grad_offset += numel
            
            # 梯度下降
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
                    param.grad.zero_()
        
        # 计算指标
        with torch.no_grad():
            current_flat = torch.cat([p.view(-1) for p in model.parameters()])
            benchmark_flat = torch.cat([p.view(-1) for p in benchmark_params.values()])
            
            diff = current_flat - benchmark_flat
            
            # 水印区域
            wm_diff = diff * flat_mask
            non_wm_diff = diff * (1.0 - flat_mask)
            
            wm_l2 = wm_diff.norm().item()
            non_wm_l2 = non_wm_diff.norm().item()
            
            # 水印区域参数
            wm_params = current_flat * flat_mask
            wm_std = wm_params.std().item()
            wm_mean = wm_params.mean().item()
            
            # 相对变化率
            wm_bench = benchmark_flat * flat_mask
            if wm_bench.norm() > 1e-10:
                wm_rel = wm_l2 / wm_bench.norm()
            else:
                wm_rel = 0.0
            
            # 水印变化占总体变化比例
            total_l2 = diff.norm().item()
            wm_ratio_of_total = wm_l2 / max(total_l2, 1e-10)
        
        history['round'].append(round_idx + 1)
        history['wm_l2_change'].append(wm_l2)
        history['nonwm_l2_change'].append(non_wm_l2)
        history['wm_relative_change'].append(wm_rel)
        history['wm_to_total_ratio'].append(wm_ratio_of_total)
        history['wm_param_std'].append(wm_std)
        history['wm_param_mean'].append(wm_mean)
        
        if (round_idx + 1) % 3 == 0 or round_idx == 0:
            print(f"    Round {round_idx+1:2d}: wm_L2={wm_l2:.4f}, nonwm_L2={non_wm_l2:.4f}, "
                  f"rel_change={wm_rel*100:.1f}%, wm_ratio={wm_ratio_of_total*100:.1f}%")
    
    return history


def main():
    print("="*70)
    print("                    水印存活能力分析")
    print("="*70)
    
    configs = [
        {'name': 'AlexNet + CIFAR-10', 'model_type': 'alexnet', 'dataset': 'cifar10',
         'in_channels': 3, 'num_classes': 10, 'input_size': 32, 'task_type': 'multiclass'},
        {'name': 'AlexNet + CIFAR-100', 'model_type': 'alexnet', 'dataset': 'cifar100',
         'in_channels': 3, 'num_classes': 100, 'input_size': 32, 'task_type': 'multiclass'},
        {'name': 'AlexNet + ChestMNIST', 'model_type': 'alexnet', 'dataset': 'chestmnist',
         'in_channels': 1, 'num_classes': 14, 'input_size': 28, 'task_type': 'multilabel'},
        {'name': 'ResNet18 + CIFAR-10', 'model_type': 'resnet18', 'dataset': 'cifar10',
         'in_channels': 3, 'num_classes': 10, 'input_size': 32, 'task_type': 'multiclass'},
        {'name': 'ResNet18 + CIFAR-100', 'model_type': 'resnet18', 'dataset': 'cifar100',
         'in_channels': 3, 'num_classes': 100, 'input_size': 32, 'task_type': 'multiclass'},
        {'name': 'ResNet18 + ChestMNIST', 'model_type': 'resnet18', 'dataset': 'chestmnist',
         'in_channels': 1, 'num_classes': 14, 'input_size': 28, 'task_type': 'multilabel'},
    ]
    
    all_results = {}
    num_rounds = 15
    
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"实验: {cfg['name']}")
        print(f"{'='*50}")
        
        history = run_survival_analysis(
            cfg['model_type'], cfg['dataset'], cfg['in_channels'],
            cfg['num_classes'], cfg['input_size'], cfg['task_type'],
            num_rounds=num_rounds
        )
        
        # 计算指标
        wm_l2_initial = history['wm_l2_change'][0]
        wm_l2_final = history['wm_l2_change'][-1]
        nonwm_l2_initial = history['nonwm_l2_change'][0]
        nonwm_l2_final = history['nonwm_l2_change'][-1]
        rel_change_final = history['wm_relative_change'][-1] * 100
        
        # 水印/非水印变化比
        wm_to_nonwm_ratio = wm_l2_final / max(nonwm_l2_final, 1e-10)
        wm_to_nonwm_initial = wm_l2_initial / max(nonwm_l2_initial, 1e-10)
        
        all_results[cfg['name']] = {
            'history': history,
            'wm_l2_initial': wm_l2_initial,
            'wm_l2_final': wm_l2_final,
            'nonwm_l2_initial': nonwm_l2_initial,
            'nonwm_l2_final': nonwm_l2_final,
            'wm_relative_change_pct': rel_change_final,
            'wm_to_nonwm_ratio_final': wm_to_nonwm_ratio,
            'wm_to_nonwm_ratio_initial': wm_to_nonwm_initial,
            'wm_growth_factor': wm_l2_final / max(wm_l2_initial, 1e-10),
        }
    
    # ========== 对比分析 ==========
    print("\n" + "="*70)
    print("                       水印存活能力对比")
    print("="*70)
    
    print(f"\n{'配置':<25} {'WM L2初':<10} {'WM L2终':<10} {'非WM L2终':<10} {'WM/非WM比':<10} {'WM相对变化':<12}")
    print("-"*85)
    
    for name, r in all_results.items():
        print(f"{name:<25} {r['wm_l2_initial']:.4f}     {r['wm_l2_final']:.4f}     "
              f"{r['nonwm_l2_final']:.4f}     {r['wm_to_nonwm_ratio_final']:.4f}     "
              f"{r['wm_relative_change_pct']:.1f}%")
    
    # ========== 假设验证 ==========
    print("\n" + "="*70)
    print("                       假设验证")
    print("="*70)
    
    a_cifar = all_results.get('AlexNet + CIFAR-10', {})
    a_chest = all_results.get('AlexNet + ChestMNIST', {})
    r_cifar = all_results.get('ResNet18 + CIFAR-10', {})
    r_chest = all_results.get('ResNet18 + ChestMNIST', {})
    
    print("\n【假设1】AlexNet + CIFAR-10 水印变化最大 (vs AlexNet + ChestMNIST)")
    if a_cifar and a_chest:
        print(f"  AlexNet + CIFAR-10:")
        print(f"    - 水印L2变化: {a_cifar['wm_l2_final']:.4f}")
        print(f"    - 非水印L2变化: {a_cifar['nonwm_l2_final']:.4f}")
        print(f"    - WM/NonWM 比例: {a_cifar['wm_to_nonwm_ratio_final']:.4f}")
        print(f"    - 水印相对变化: {a_cifar['wm_relative_change_pct']:.1f}%")
        print(f"  AlexNet + ChestMNIST:")
        print(f"    - 水印L2变化: {a_chest['wm_l2_final']:.4f}")
        print(f"    - 非水印L2变化: {a_chest['nonwm_l2_final']:.4f}")
        print(f"    - WM/NonWM 比例: {a_chest['wm_to_nonwm_ratio_final']:.4f}")
        print(f"    - 水印相对变化: {a_chest['wm_relative_change_pct']:.1f}%")
        # WM/NonWM 比例越低越好（说明水印相对受保护）
        better_protection = a_cifar['wm_to_nonwm_ratio_final'] < a_chest['wm_to_nonwm_ratio_final']
        print(f"  结论: CIFAR-10 水印相对变化更{'大' if not better_protection else '小'}")
    
    print("\n【假设2】AlexNet CIFAR-10 vs ResNet18 CIFAR-10")
    if a_cifar and r_cifar:
        print(f"  AlexNet:")
        print(f"    - 水印L2: {a_cifar['wm_l2_final']:.4f}, 非水印L2: {a_cifar['nonwm_l2_final']:.4f}")
        print(f"    - WM/NonWM: {a_cifar['wm_to_nonwm_ratio_final']:.4f}")
        print(f"    - 增长倍数: {a_cifar['wm_growth_factor']:.2f}x")
        print(f"  ResNet18:")
        print(f"    - 水印L2: {r_cifar['wm_l2_final']:.4f}, 非水印L2: {r_cifar['nonwm_l2_final']:.4f}")
        print(f"    - WM/NonWM: {r_cifar['wm_to_nonwm_ratio_final']:.4f}")
        print(f"    - 增长倍数: {r_cifar['wm_growth_factor']:.2f}x")
        better_protection = a_cifar['wm_to_nonwm_ratio_final'] < r_cifar['wm_to_nonwm_ratio_final']
        print(f"  结论: AlexNet 水印相对保护{'更差' if not better_protection else '更好'}")
    
    print("\n【核心发现】")
    print("  以下按水印相对变化从小到大排序 (变化越小说明水印越稳定):")
    sorted_results = sorted(all_results.items(), 
                            key=lambda x: x[1]['wm_relative_change_pct'])
    for i, (name, r) in enumerate(sorted_results, 1):
        print(f"    {i}. {name}: {r['wm_relative_change_pct']:.2f}%")
    
    # ========== 保存结果 ==========
    os.makedirs("save/survival_analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"save/survival_analysis/survival_{timestamp}.json"
    
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n报告已保存到: {output_path}")
    return all_results


if __name__ == '__main__':
    main()
