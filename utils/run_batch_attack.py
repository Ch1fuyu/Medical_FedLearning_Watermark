#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量攻击实验脚本
对所有18个训练模型分别进行剪枝攻击和微调攻击实验

实验配置 (对应不同训练批次的模型文件):
- 数据集: chestmnist, cifar10, cifar100
- 模型: alexnet, resnet
- 客户端数: 5, 10, 20
- 模型路径: ./save/{model}/{dataset}/{pkl_filename}
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 实验配置列表 (根据 trace_results_*.json 对应的训练结果)
EXPERIMENTS = [
    # chestmnist + alexnet
    {"model": "alexnet", "dataset": "chestmnist", "clients": 5,
     "pkl": "202605141420_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_auc_0.7624_enhanced.pkl"},
    {"model": "alexnet", "dataset": "chestmnist", "clients": 10,
     "pkl": "202605141530_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_auc_0.7609_enhanced.pkl"},
    {"model": "alexnet", "dataset": "chestmnist", "clients": 20,
     "pkl": "202605141639_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_auc_0.7558_enhanced.pkl"},

    # chestmnist + resnet
    {"model": "resnet", "dataset": "chestmnist", "clients": 5,
     "pkl": "202605141829_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_auc_0.7635_enhanced.pkl"},
    {"model": "resnet", "dataset": "chestmnist", "clients": 10,
     "pkl": "202605142018_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_auc_0.7618_enhanced.pkl"},
    {"model": "resnet", "dataset": "chestmnist", "clients": 20,
     "pkl": "202605142206_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_auc_0.7600_enhanced.pkl"},

    # cifar10 + alexnet
    {"model": "alexnet", "dataset": "cifar10", "clients": 5,
     "pkl": "202605142301_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.9006_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar10", "clients": 10,
     "pkl": "202605142356_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.8984_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar10", "clients": 20,
     "pkl": "202605150049_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.8907_enhanced.pkl"},

    # cifar10 + resnet
    {"model": "resnet", "dataset": "cifar10", "clients": 5,
     "pkl": "202605150218_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.9215_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar10", "clients": 10,
     "pkl": "202605150349_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.9230_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar10", "clients": 20,
     "pkl": "202605150518_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.9100_enhanced.pkl"},

    # cifar100 + alexnet
    {"model": "alexnet", "dataset": "cifar100", "clients": 5,
     "pkl": "202605150614_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.6698_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar100", "clients": 10,
     "pkl": "202605150710_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.6661_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar100", "clients": 20,
     "pkl": "202605150804_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.6324_enhanced.pkl"},

    # cifar100 + resnet
    {"model": "resnet", "dataset": "cifar100", "clients": 5,
     "pkl": "202605150935_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.7262_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar100", "clients": 10,
     "pkl": "202605151106_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.7186_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar100", "clients": 20,
     "pkl": "202605151235_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.6867_enhanced.pkl"},
]


def get_model_path(exp_config):
    """根据实验配置获取模型文件路径"""
    pkl = exp_config["pkl"]
    model_type = exp_config["model"]
    dataset = exp_config["dataset"]
    return os.path.join(PROJECT_ROOT, "save", model_type, dataset, pkl)


def run_pruning_attack(model_path, model_type, client_num, autoencoder_dir):
    """运行剪枝攻击实验"""
    print(f"\n{'='*80}")
    print(f"运行剪枝攻击: {model_path}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        "pruning_attack.py",
        "--model_path", model_path,
        "--model_type", model_type,
        "--client_num", str(client_num),
        "--autoencoder_dir", autoencoder_dir,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True, "成功"
    except subprocess.CalledProcessError as e:
        return False, f"失败 (退出码: {e.returncode})"


def run_finetune_attack(model_path, model_type, client_num, dataset,
                        autoencoder_dir, finetune_epochs, learning_rate, batch_size, save_mode='paper'):
    """运行微调攻击实验"""
    print(f"\n{'='*80}")
    print(f"运行微调攻击: {model_path}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        "finetune_attack.py",
        "--model_path", model_path,
        "--model_type", model_type,
        "--client_num", str(client_num),
        "--dataset", dataset,
        "--autoencoder_dir", autoencoder_dir,
        "--finetune_epochs", str(finetune_epochs),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--save_mode", save_mode,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True, "成功"
    except subprocess.CalledProcessError as e:
        return False, f"失败 (退出码: {e.returncode})"


def run_all_pruning_experiments(autoencoder_dir=None, start_idx=0, end_idx=None):
    """运行所有剪枝攻击实验"""
    if autoencoder_dir is None:
        autoencoder_dir = os.path.join(PROJECT_ROOT, 'save', 'autoencoder')

    if end_idx is None:
        end_idx = len(EXPERIMENTS)

    print(f"\n{'#'*80}")
    print(f"# 开始运行剪枝攻击实验 (实验 {start_idx+1} ~ {end_idx})")
    print(f"# 总共 {end_idx - start_idx} 个实验")
    print(f"{'#'*80}")

    results = []

    for i, exp in enumerate(EXPERIMENTS[start_idx:end_idx], start=start_idx+1):
        model_path = get_model_path(exp)
        exp_name = f"{exp['model']}_{exp['dataset']}_{exp['clients']}clients"

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不存在: {model_path}")
            results.append({
                "index": i, "name": exp_name, "model": exp["model"],
                "dataset": exp["dataset"], "clients": exp["clients"],
                "pkl": exp["pkl"], "success": False, "result": "模型文件不存在"
            })
            continue

        success, result = run_pruning_attack(
            model_path=model_path,
            model_type=exp["model"],
            client_num=exp["clients"],
            autoencoder_dir=autoencoder_dir
        )

        results.append({
            "index": i, "name": exp_name, "model": exp["model"],
            "dataset": exp["dataset"], "clients": exp["clients"],
            "pkl": exp["pkl"], "success": success, "result": result
        })

    return results


def run_all_finetune_experiments(autoencoder_dir=None,
                                  finetune_epochs=50, learning_rate=0.001, batch_size=128,
                                  save_mode='paper', start_idx=0, end_idx=None):
    """运行所有微调攻击实验"""
    if autoencoder_dir is None:
        autoencoder_dir = os.path.join(PROJECT_ROOT, 'save', 'autoencoder')

    if end_idx is None:
        end_idx = len(EXPERIMENTS)

    print(f"\n{'#'*80}")
    print(f"# 开始运行微调攻击实验 (实验 {start_idx+1} ~ {end_idx})")
    print(f"# 总共 {end_idx - start_idx} 个实验")
    print(f"# 微调参数: epochs={finetune_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"# 保存模式: {save_mode}")
    print(f"{'#'*80}")

    results = []

    for i, exp in enumerate(EXPERIMENTS[start_idx:end_idx], start=start_idx+1):
        model_path = get_model_path(exp)
        exp_name = f"{exp['model']}_{exp['dataset']}_{exp['clients']}clients"

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不存在: {model_path}")
            results.append({
                "index": i, "name": exp_name, "model": exp["model"],
                "dataset": exp["dataset"], "clients": exp["clients"],
                "pkl": exp["pkl"], "success": False, "result": "模型文件不存在"
            })
            continue

        success, result = run_finetune_attack(
            model_path=model_path,
            model_type=exp["model"],
            client_num=exp["clients"],
            dataset=exp["dataset"],
            autoencoder_dir=autoencoder_dir,
            finetune_epochs=finetune_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            save_mode=save_mode
        )

        results.append({
            "index": i, "name": exp_name, "model": exp["model"],
            "dataset": exp["dataset"], "clients": exp["clients"],
            "pkl": exp["pkl"], "success": success, "result": result
        })

    return results


def print_summary(results, attack_type):
    """打印实验结果汇总"""
    print(f"\n{'='*80}")
    print(f"# {attack_type}攻击实验结果汇总")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)

    print(f"\n成功率: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

    print(f"\n{'序号':<4} {'实验名称':<35} {'状态':<15}")
    print("-" * 60)

    for r in results:
        status = "成功" if r["success"] else f"失败: {r['result']}"
        print(f"{r['index']:<4} {r['name']:<35} {status:<15}")

    return results


def save_results_to_file(results, attack_type, save_dir=None):
    """保存实验结果到文件"""
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'save', 'batch_results')
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(save_dir, f"batch_{attack_type}_{timestamp}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"批量{attack_type}攻击实验结果\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        for r in results:
            status = "成功" if r["success"] else f"失败: {r['result']}"
            f.write(f"[{r['index']}] {r['name']}\n")
            f.write(f"  模型: {r['model']}, 数据集: {r['dataset']}, 客户端: {r['clients']}\n")
            f.write(f"  文件: {r['pkl']}\n")
            f.write(f"  状态: {status}\n\n")

    print(f"\n结果已保存到: {filepath}")
    return filepath


def collect_paper_results(data_dir=None):
    """收集所有论文数据，生成汇总CSV"""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, 'save', 'finetune_attack')
    import pandas as pd
    import glob
    import re

    print(f"\n{'#'*80}")
    print("# 收集论文数据汇总")
    print(f"{'#'*80}")

    paper_files = glob.glob(os.path.join(data_dir, "*_paper.csv"))

    if not paper_files:
        print("  警告: 没有找到论文数据文件 (_paper.csv)")
        return

    all_data = []

    for filepath in paper_files:
        filename = os.path.basename(filepath)
        match = re.search(r'finetune_attack_\d+_\d+_(\w+)_paper\.csv', filename)
        dataset = match.group(1) if match else "unknown"

        try:
            df = pd.read_csv(filepath, comment='#', encoding='utf-8-sig')
            if len(df) > 0:
                df['source_file'] = filename
                df['dataset'] = dataset
                all_data.append(df)
        except Exception as e:
            print(f"  警告: 读取失败 {filename}: {e}")

    if not all_data:
        print("  警告: 没有成功读取任何数据")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(data_dir, f'paper_summary_{timestamp}.csv')
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n  论文数据汇总已保存: {output_path}")
    print(f"  共 {len(combined_df)} 条记录，来自 {len(all_data)} 个实验")
    print(f"\n  数据预览 (前20行):")
    print(combined_df.head(20).to_string(index=False))


def run_batch(args):
    """执行批量实验"""
    print(f"\n批量攻击实验配置:")
    print(f"  攻击类型: {args.attack_type}")
    print(f"  自编码器目录: {args.autoencoder_dir}")
    if args.attack_type in ['finetune', 'both']:
        print(f"  微调轮数: {args.finetune_epochs}")
        print(f"  学习率: {args.learning_rate}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  保存模式: {args.save_mode}")
    print(f"  实验范围: {args.start_idx} ~ {args.end_idx if args.end_idx else len(EXPERIMENTS)-1}")

    # 运行剪枝攻击
    if args.attack_type in ['pruning', 'both']:
        pruning_results = run_all_pruning_experiments(
            autoencoder_dir=args.autoencoder_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        print_summary(pruning_results, "剪枝")
        save_results_to_file(pruning_results, "pruning")

    # 运行微调攻击
    if args.attack_type in ['finetune', 'both']:
        finetune_results = run_all_finetune_experiments(
            autoencoder_dir=args.autoencoder_dir,
            finetune_epochs=args.finetune_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            save_mode=args.save_mode,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        print_summary(finetune_results, "微调")
        save_results_to_file(finetune_results, "finetune")

    # 汇总论文数据
    if args.attack_type in ['finetune', 'both']:
        collect_paper_results()

    print(f"\n{'='*80}")
    print("批量实验全部完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量攻击实验脚本')
    parser.add_argument('--attack_type', type=str, required=True,
                       choices=['pruning', 'finetune', 'both', 'collect'],
                       help='pruning(剪枝)/finetune(微调)/both(两者)/collect(收集论文数据)')
    parser.add_argument('--autoencoder_dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'save', 'autoencoder'),
                       help='自编码器目录')
    parser.add_argument('--finetune_epochs', type=int, default=50,
                       help='微调轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--save_mode', type=str, default='paper',
                       choices=['all', 'paper', 'full'],
                       help='保存模式: all(完整+论文), paper(仅论文数据,默认), full(仅完整结果)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='起始实验索引')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='结束实验索引')

    args = parser.parse_args()

    if args.attack_type == 'collect':
        collect_paper_results()
    else:
        run_batch(args)
