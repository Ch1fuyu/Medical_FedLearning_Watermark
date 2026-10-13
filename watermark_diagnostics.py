"""
水印容量与梯度分布诊断工具
===========================

目标: 验证假设
1. AlexNet + CIFAR-10 容量不足,主任务"占满"参数空间
2. ResNet18 容量大,水印有生存空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple


def get_model(model_type, in_channels, num_classes, input_size):
    if model_type == 'alexnet':
        from models.alexnet import AlexNet
        return AlexNet(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
    else:
        from models.resnet import resnet18
        return resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)


def get_grad_stats(model, watermark_mask_flat, all_grads_flat):
    """计算梯度统计"""
    wm_grad = all_grads_flat * watermark_mask_flat
    non_wm_grad = all_grads_flat * (1.0 - watermark_mask_flat)
    wm_count = watermark_mask_flat.sum().item()
    non_wm_count = (1.0 - watermark_mask_flat).sum().item()
    
    wm_grad_mean = wm_grad.abs().sum().item() / max(wm_count, 1)
    non_wm_grad_mean = non_wm_grad.abs().sum().item() / max(non_wm_count, 1)
    ratio = wm_grad_mean / max(non_wm_grad_mean, 1e-10)
    
    # 活跃参数比例
    active_ratio = (all_grads_flat.abs() > 1e-6).float().mean().item()
    
    # 层利用率
    layer_util = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            layer_active = (param.grad.abs() > 1e-6).float().mean().item()
            layer_intensity = param.grad.abs().mean().item()
            layer_util[name] = {
                'active_ratio': layer_active,
                'grad_intensity': layer_intensity,
                'param_numel': param.numel(),
            }
    
    return {
        'total_grad_norm': all_grads_flat.norm().item(),
        'total_grad_mean': all_grads_flat.abs().mean().item(),
        'total_grad_std': all_grads_flat.std().item(),
        'active_grad_ratio': active_ratio,
        'watermark_grad_mean': wm_grad_mean,
        'non_watermark_grad_mean': non_wm_grad_mean,
        'wm_to_non_wm_ratio': ratio,
        'watermark_norm': wm_grad.norm().item(),
        'layer_utilization': layer_util,
    }


def run_one_trial(model_type, dataset, in_channels, num_classes, 
                  input_size, task_type, watermark_ratio=0.1, seed=42):
    """运行单次试验"""
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type, in_channels, num_classes, input_size).to(device)
    
    # 计算总参数
    total_params = sum(p.numel() for p in model.parameters())
    watermark_numel = int(total_params * watermark_ratio)
    watermark_mask = torch.zeros(total_params, device=device)
    watermark_mask[:watermark_numel] = 1.0
    
    # 准备输入
    batch_size = 32
    if dataset == 'cifar10':
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
    elif dataset == 'chestmnist':
        x = torch.randn(batch_size, 1, 28, 28, device=device)
        y = torch.randint(0, 2, (batch_size, 14), device=device).float()
    elif dataset == 'cifar100':
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 100, (batch_size,), device=device)
    
    # 前向反向
    model.train()
    pred = model(x)
    if task_type == 'multiclass':
        loss = F.cross_entropy(pred, y)
    else:
        loss = F.binary_cross_entropy_with_logits(pred, y)
    loss.backward()
    
    # 收集所有梯度
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads)
    
    # 同步掩码
    if watermark_mask.shape[0] != all_grads.shape[0]:
        # 调整掩码大小
        wm_resized = torch.zeros(all_grads.shape[0], device=device)
        wm_count = min(watermark_numel, all_grads.shape[0])
        wm_resized[:wm_count] = 1.0
        watermark_mask = wm_resized
    
    # 获取统计
    stats = get_grad_stats(model, watermark_mask, all_grads)
    stats['loss'] = loss.item()
    stats['total_params'] = total_params
    stats['watermark_numel'] = watermark_numel
    
    return stats


def main():
    print("="*70)
    print("                    水印容量与梯度分布诊断")
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
    num_trials = 3
    
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"实验: {cfg['name']}")
        print(f"{'='*50}")
        
        trial_results = []
        for trial in range(num_trials):
            stats = run_one_trial(
                cfg['model_type'], cfg['dataset'], cfg['in_channels'],
                cfg['num_classes'], cfg['input_size'], cfg['task_type'],
                seed=42 + trial * 10
            )
            trial_results.append(stats)
        
        # 聚合
        avg = {}
        for key in ['total_grad_norm', 'total_grad_mean', 'total_grad_std',
                    'active_grad_ratio', 'watermark_grad_mean', 'non_watermark_grad_mean',
                    'wm_to_non_wm_ratio', 'watermark_norm', 'loss', 'total_params',
                    'watermark_numel']:
            values = [r[key] for r in trial_results]
            avg[key] = float(np.mean(values))
            avg[f'{key}_std'] = float(np.std(values))
        
        # 聚合层利用率
        all_layers = set()
        for tr in trial_results:
            all_layers.update(tr['layer_utilization'].keys())
        
        layer_summary = {}
        for layer in all_layers:
            active_vals = [tr['layer_utilization'][layer]['active_ratio'] 
                          for tr in trial_results if layer in tr['layer_utilization']]
            intensity_vals = [tr['layer_utilization'][layer]['grad_intensity'] 
                            for tr in trial_results if layer in tr['layer_utilization']]
            layer_summary[layer] = {
                'active_ratio_mean': float(np.mean(active_vals)),
                'grad_intensity_mean': float(np.mean(intensity_vals)),
            }
        avg['layer_utilization'] = layer_summary
        
        all_results[cfg['name']] = avg
        
        # 打印
        print(f"  总参数量: {int(avg['total_params']):,}")
        print(f"  水印参数量: {int(avg['watermark_numel']):,} ({avg['watermark_numel']/avg['total_params']*100:.1f}%)")
        print(f"  损失值: {avg['loss']:.4f}")
        print(f"  --- 梯度诊断 ---")
        print(f"  总梯度范数: {avg['total_grad_norm']:.4f}")
        print(f"  总梯度均值: {avg['total_grad_mean']:.6f}")
        print(f"  活跃参数比例: {avg['active_grad_ratio']*100:.1f}%")
        print(f"  --- 水印 vs 非水印 ---")
        print(f"  水印梯度均值: {avg['watermark_grad_mean']:.6f}")
        print(f"  非水印梯度均值: {avg['non_watermark_grad_mean']:.6f}")
        print(f"  梯度比 (WM/Non-WM): {avg['wm_to_non_wm_ratio']:.4f}")
        print(f"  --- 容量利用率 ---")
        avg_active = np.mean([v['active_ratio_mean'] for v in layer_summary.values()])
        avg_intensity = np.mean([v['grad_intensity_mean'] for v in layer_summary.values()])
        print(f"  平均层活跃度: {avg_active*100:.1f}%")
        print(f"  平均梯度强度: {avg_intensity:.6f}")
    
    # ========== 对比分析 ==========
    print("\n" + "="*70)
    print("                       对比分析报告")
    print("="*70)
    
    print("\n【对比1】AlexNet 在不同任务上的表现")
    print(f"{'指标':<25} {'CIFAR-10':<15} {'CIFAR-100':<15} {'ChestMNIST':<15}")
    print("-"*75)

    a = all_results['AlexNet + CIFAR-10']
    a100 = all_results.get('AlexNet + CIFAR-100')
    b = all_results['AlexNet + ChestMNIST']

    for metric_name, key in [('总梯度范数', 'total_grad_norm'),
                              ('活跃参数比例', 'active_grad_ratio'),
                              ('水印梯度比', 'wm_to_non_wm_ratio'),
                              ('平均层活跃度', None)]:
        if key:
            v1, v2, v3 = a[key], a100[key], b[key]
            if metric_name == '活跃参数比例':
                print(f"{metric_name:<25} {v1*100:.1f}%         {v2*100:.1f}%         {v3*100:.1f}%")
            else:
                print(f"{metric_name:<25} {v1:.4f}         {v2:.4f}         {v3:.4f}")
        else:
            avg_a = np.mean([v['active_ratio_mean'] for v in a['layer_utilization'].values()])
            avg_a100 = np.mean([v['active_ratio_mean'] for v in a100['layer_utilization'].values()])
            avg_b = np.mean([v['active_ratio_mean'] for v in b['layer_utilization'].values()])
            print(f"{metric_name:<25} {avg_a*100:.1f}%         {avg_a100*100:.1f}%         {avg_b*100:.1f}%")
    
    print("\n【对比2】CIFAR-10 上不同架构的表现")
    print(f"{'指标':<25} {'AlexNet':<15} {'ResNet18':<15} {'AlexNet/ResNet18':<10}")
    print("-"*70)

    a = all_results['AlexNet + CIFAR-10']
    c = all_results['ResNet18 + CIFAR-10']

    for metric_name, key in [('总参数量', 'total_params'),
                              ('总梯度范数', 'total_grad_norm'),
                              ('活跃参数比例', 'active_grad_ratio'),
                              ('水印梯度比', 'wm_to_non_wm_ratio')]:
        v1, v2 = a[key], c[key]
        ratio = v1 / max(v2, 1e-10)
        if metric_name == '总参数量':
            print(f"{metric_name:<25} {int(v1):<15} {int(v2):<15} {ratio:.2f}x")
        elif metric_name == '活跃参数比例':
            print(f"{metric_name:<25} {v1*100:.1f}%          {v2*100:.1f}%          {ratio:.2f}x")
        else:
            print(f"{metric_name:<25} {v1:.4f}          {v2:.4f}          {ratio:.2f}x")

    print("\n【对比3】CIFAR-100 上不同架构的表现")
    print(f"{'指标':<25} {'AlexNet':<15} {'ResNet18':<15} {'AlexNet/ResNet18':<10}")
    print("-"*70)

    a100 = all_results['AlexNet + CIFAR-100']
    r100 = all_results['ResNet18 + CIFAR-100']

    for metric_name, key in [('总参数量', 'total_params'),
                              ('总梯度范数', 'total_grad_norm'),
                              ('活跃参数比例', 'active_grad_ratio'),
                              ('水印梯度比', 'wm_to_non_wm_ratio')]:
        v1, v2 = a100[key], r100[key]
        ratio = v1 / max(v2, 1e-10)
        if metric_name == '总参数量':
            print(f"{metric_name:<25} {int(v1):<15} {int(v2):<15} {ratio:.2f}x")
        elif metric_name == '活跃参数比例':
            print(f"{metric_name:<25} {v1*100:.1f}%          {v2*100:.1f}%          {ratio:.2f}x")
        else:
            print(f"{metric_name:<25} {v1:.4f}          {v2:.4f}          {ratio:.2f}x")
    
    # ========== 假设验证 ==========
    print("\n" + "="*70)
    print("                       假设验证")
    print("="*70)

    a_cifar = all_results['AlexNet + CIFAR-10']
    a_cifar100 = all_results['AlexNet + CIFAR-100']
    a_chest = all_results['AlexNet + ChestMNIST']
    r_cifar = all_results['ResNet18 + CIFAR-10']
    r_cifar100 = all_results['ResNet18 + CIFAR-100']
    r_chest = all_results['ResNet18 + ChestMNIST']

    print("\n假设1: AlexNet 处理复杂任务时主任务占据更多参数空间")
    print(f"  总梯度强度: CIFAR-10={a_cifar['total_grad_norm']:.4f}  CIFAR-100={a_cifar100['total_grad_norm']:.4f}  ChestMNIST={a_chest['total_grad_norm']:.4f}")
    print(f"  活跃参数比例: CIFAR-10={a_cifar['active_grad_ratio']*100:.1f}%  CIFAR-100={a_cifar100['active_grad_ratio']*100:.1f}%  ChestMNIST={a_chest['active_grad_ratio']*100:.1f}%")
    print(f"  解读: {'CIFAR 系列激活更多参数' if a_cifar['active_grad_ratio'] > a_chest['active_grad_ratio'] else '差异不显著'}")

    print("\n假设2: AlexNet 比 ResNet18 在复杂任务上容量更紧张")
    print(f"  参数量比例 (AlexNet/ResNet18): CIFAR-10={a_cifar['total_params']/r_cifar['total_params']:.2f}x  CIFAR-100={a_cifar100['total_params']/r_cifar100['total_params']:.2f}x")
    print(f"  梯度强度比例: CIFAR-10={a_cifar['total_grad_norm']/r_cifar['total_grad_norm']:.2f}x  CIFAR-100={a_cifar100['total_grad_norm']/r_cifar100['total_grad_norm']:.2f}x")
    unit_a = a_cifar['total_grad_norm']/a_cifar['total_params']
    unit_a100 = a_cifar100['total_grad_norm']/a_cifar100['total_params']
    unit_r = r_cifar['total_grad_norm']/r_cifar['total_params']
    unit_r100 = r_cifar100['total_grad_norm']/r_cifar100['total_params']
    print(f"  单位参数负载 (梯度/参数): CIFAR-10 AlexNet={unit_a:.2e} ResNet18={unit_r:.2e} | CIFAR-100 AlexNet={unit_a100:.2e} ResNet18={unit_r100:.2e}")
    print(f"  解读: AlexNet 在 CIFAR-10/100 上单位负载分别是 ResNet18 的 {unit_a/unit_r:.1f}x / {unit_a100/unit_r100:.1f}x")

    print("\n假设3: 任务复杂度越高，容量竞争越剧烈（梯度强度排序）")
    grad_list = [
        ('AlexNet + CIFAR-100', a_cifar100['total_grad_norm']),
        ('AlexNet + CIFAR-10', a_cifar['total_grad_norm']),
        ('AlexNet + ChestMNIST', a_chest['total_grad_norm']),
        ('ResNet18 + CIFAR-100', r_cifar100['total_grad_norm']),
        ('ResNet18 + CIFAR-10', r_cifar['total_grad_norm']),
        ('ResNet18 + ChestMNIST', r_chest['total_grad_norm']),
    ]
    grad_list.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'排名':<6} {'配置':<25} {'梯度范数':<12} {'相对基准':<10}")
    base_grad = r_chest['total_grad_norm']
    for rank, (name, grad) in enumerate(grad_list, 1):
        print(f"  {rank:<6} {name:<25} {grad:.4f}       {grad/base_grad:.2f}x")
    
    # ========== 保存结果 ==========
    os.makedirs("save/diagnostics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"save/diagnostics/diagnostics_{timestamp}.json"
    
    def make_serializable(obj):
        if isinstance(obj, np.floating):
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
