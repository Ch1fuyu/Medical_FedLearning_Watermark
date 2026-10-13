"""
水印梯度缩放参数自动调优脚本
=============================

使用方法:
    python tune_watermark_ratio.py --dataset cifar10 --model alexnet --ratios 0.2 0.3 0.4 0.5
    python tune_watermark_ratio.py --preset alexnet_cifar10  # 使用预设配置快速测试
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple

# 预设配置表
PRESETS = {
    # 架构-数据集: (推荐ratio范围, 检测阈值说明)
    'alexnet_chestmnist': {
        'ratios': [0.20, 0.25, 0.30, 0.35, 0.40],
        'description': 'AlexNet + ChestMNIST: 简单任务，0.25-0.35 效果最好'
    },
    'alexnet_cifar10': {
        'ratios': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        'description': 'AlexNet + CIFAR-10: 中等任务，需要更大ratio保持水印信号'
    },
    'resnet18_chestmnist': {
        'ratios': [0.25, 0.30, 0.35, 0.40, 0.45],
        'description': 'ResNet-18 + ChestMNIST: 残差连接，梯度更稳定'
    },
    'resnet18_cifar10': {
        'ratios': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        'description': 'ResNet-18 + CIFAR-10: 复杂任务，梯度方差大'
    },
}


def run_experiment(dataset: str, model: str, target_ratio: float, 
                   epochs: int = 20, client_num: int = 5,
                   leak_interval: int = 5) -> Dict:
    """
    运行单次实验并返回结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"tune_{model}_{dataset}_ratio{target_ratio}_{timestamp}"
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--model_name", model,
        "--watermark_grad_target_ratio", str(target_ratio),
        "--epochs", str(epochs),
        "--client_num", str(client_num),
        "--leak_interval", str(leak_interval),
        "--save_excel_dir", f"save/tuning/{exp_name}",
        "--log_file", f"save/tuning/{exp_name}.log",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Ratio: {target_ratio}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 解析输出，提取关键指标
        output = result.stdout + result.stderr
        
        # 尝试从输出或日志文件中提取结果
        metrics = {
            'target_ratio': target_ratio,
            'experiment': exp_name,
            'success': result.returncode == 0,
            'detection_accuracy': None,
            'final_accuracy': None,
            'watermark_ratio_actual': None,
        }
        
        # 解析日志文件
        log_file = f"save/tuning/{exp_name}.log"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 提取检测准确率
                    if '检测准确率' in line or 'detection accuracy' in line.lower():
                        try:
                            acc = float(''.join(filter(lambda x: x in '0123456789.', line.split(':')[-1])))
                            metrics['detection_accuracy'] = acc
                        except:
                            pass
                    
                    # 提取最终准确率
                    if 'Test Accuracy' in line or '测试准确率' in line:
                        try:
                            acc = float(''.join(filter(lambda x: x in '0123456789.', line.split(':')[-1])))
                            metrics['final_accuracy'] = acc
                        except:
                            pass
                    
                    # 提取实际水印梯度比率
                    if 'wm/non_wm ratio' in line:
                        try:
                            ratio = float(line.split('wm/non_wm ratio:')[1].split('|')[0].strip())
                            metrics['watermark_ratio_actual'] = ratio
                        except:
                            pass
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {
            'target_ratio': target_ratio,
            'experiment': exp_name,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'target_ratio': target_ratio,
            'experiment': exp_name,
            'success': False,
            'error': str(e)
        }


def analyze_results(results: List[Dict]) -> Dict:
    """
    分析调优结果，返回最优参数
    """
    # 过滤成功的实验
    valid_results = [r for r in results if r.get('success', False)]
    
    if not valid_results:
        return {'error': 'No valid results', 'recommendations': []}
    
    # 按检测准确率排序
    sorted_by_detection = sorted(
        valid_results,
        key=lambda x: (x.get('detection_accuracy') or 0, x.get('final_accuracy') or 0),
        reverse=True
    )
    
    # 综合评分：检测准确率权重更高
    for r in valid_results:
        det_acc = r.get('detection_accuracy') or 0
        fin_acc = r.get('final_accuracy') or 0
        r['composite_score'] = det_acc * 0.7 + fin_acc * 0.3
    
    sorted_by_composite = sorted(valid_results, key=lambda x: x['composite_score'], reverse=True)
    
    return {
        'best_by_detection': sorted_by_detection[0] if sorted_by_detection else None,
        'best_by_composite': sorted_by_composite[0] if sorted_by_composite else None,
        'all_results': valid_results,
        'recommendations': generate_recommendations(valid_results)
    }


def generate_recommendations(results: List[Dict]) -> List[str]:
    """生成调优建议"""
    recommendations = []
    
    # 分析检测准确率趋势
    ratios = [r['target_ratio'] for r in results]
    det_accs = [r.get('detection_accuracy', 0) for r in results]
    
    if len(ratios) >= 2:
        # 找到检测准确率最高的ratio
        best_idx = det_accs.index(max(det_accs))
        best_ratio = ratios[best_idx]
        
        recommendations.append(
            f"当前测试中，检测准确率最高的ratio是 {best_ratio}，准确率: {det_accs[best_idx]:.1f}%"
        )
        
        # 分析趋势
        if best_ratio == min(ratios):
            recommendations.append("提示: 最小的ratio效果最好，可能需要进一步降低ratio (如0.15-0.20)")
        elif best_ratio == max(ratios):
            recommendations.append("提示: 最大的ratio效果最好，可能需要进一步提高ratio (如0.50-0.60)")
        else:
            recommendations.append("提示: 存在最优区间，建议在该值附近进行更细粒度的搜索")
    
    return recommendations


def print_report(analysis: Dict):
    """打印最终报告"""
    print("\n" + "="*70)
    print("                    水印梯度缩放参数调优报告")
    print("="*70)
    
    if 'error' in analysis:
        print(f"\n❌ 调优失败: {analysis['error']}")
        return
    
    print("\n📊 实验结果汇总:")
    print("-"*70)
    print(f"{'Ratio':<10} {'检测准确率':<15} {'最终准确率':<15} {'实际wm ratio':<15} {'综合评分':<10}")
    print("-"*70)
    
    for r in sorted(analysis['all_results'], key=lambda x: x['target_ratio']):
        det = r.get('detection_accuracy')
        fin = r.get('final_accuracy')
        wm = r.get('watermark_ratio_actual')
        score = r.get('composite_score', 0)
        print(f"{r['target_ratio']:<10.2f} "
              f"{f'{det:.1f}%' if det else 'N/A':<15} "
              f"{f'{fin:.2f}%' if fin else 'N/A':<15} "
              f"{f'{wm:.3f}' if wm else 'N/A':<15} "
              f"{score:.2f}")
    
    if analysis['best_by_detection']:
        best = analysis['best_by_detection']
        print("\n🏆 最优参数 (按检测准确率):")
        print(f"   - target_ratio: {best['target_ratio']}")
        print(f"   - 检测准确率: {best.get('detection_accuracy', 'N/A')}")
        print(f"   - 最终准确率: {best.get('final_accuracy', 'N/A')}")
    
    if analysis['recommendations']:
        print("\n💡 调优建议:")
        for rec in analysis['recommendations']:
            print(f"   {rec}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='水印梯度缩放参数自动调优')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['chestmnist', 'cifar10', 'cifar100'],
                       help='数据集')
    parser.add_argument('--model', type=str, default='alexnet',
                       choices=['alexnet', 'resnet'],
                       help='模型架构')
    parser.add_argument('--ratios', type=float, nargs='+',
                       help='要测试的ratio值列表，如: 0.2 0.3 0.4 0.5')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                       help='使用预设配置')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--client_num', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--leak_interval', type=int, default=5,
                       help='泄漏检测间隔')
    
    args = parser.parse_args()
    
    # 处理预设配置
    if args.preset and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        args.ratios = preset['ratios']
        # 从预设名称解析dataset和model
        parts = args.preset.split('_')
        if len(parts) >= 2:
            args.model = parts[0] if 'resnet' not in parts[0] else 'resnet18'
            args.dataset = '_'.join(parts[1:]) if len(parts) > 2 else parts[-1]
        print(f"使用预设配置: {args.preset}")
        print(f"描述: {preset['description']}")
    
    # 默认ratio范围
    if not args.ratios:
        # 根据架构和数据集提供默认范围
        if args.model == 'alexnet' and args.dataset == 'chestmnist':
            args.ratios = [0.20, 0.25, 0.30, 0.35, 0.40]
        elif args.model == 'alexnet' and args.dataset == 'cifar10':
            args.ratios = [0.30, 0.35, 0.40, 0.45, 0.50]
        elif args.model == 'resnet' and args.dataset == 'chestmnist':
            args.ratios = [0.25, 0.30, 0.35, 0.40, 0.45]
        else:  # resnet + cifar10 或其他
            args.ratios = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    print(f"\n📋 测试配置:")
    print(f"   - 数据集: {args.dataset}")
    print(f"   - 模型: {args.model}")
    print(f"   - 测试的ratios: {args.ratios}")
    print(f"   - 训练轮数: {args.epochs}")
    
    # 创建输出目录
    os.makedirs("save/tuning", exist_ok=True)
    
    # 运行实验
    results = []
    for ratio in sorted(args.ratios):
        result = run_experiment(
            dataset=args.dataset,
            model=args.model,
            target_ratio=ratio,
            epochs=args.epochs,
            client_num=args.client_num,
            leak_interval=args.leak_interval
        )
        results.append(result)
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 打印报告
    print_report(analysis)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"save/tuning/report_{args.model}_{args.dataset}_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'model': args.model,
                'ratios': args.ratios,
                'epochs': args.epochs
            },
            'analysis': {
                k: v for k, v in analysis.items() 
                if k != 'all_results'
            },
            'raw_results': results
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
