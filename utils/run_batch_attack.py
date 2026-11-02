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
import re
import subprocess
import argparse
from datetime import datetime

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 实验配置列表 (2026年6月 reg_ablation drift+margin 实验)
EXPERIMENTS = [
    # chestmnist + alexnet
    {"model": "alexnet", "dataset": "chestmnist", "clients": 5,
     "pkl": "202606081711_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_auc_0.7704_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "chestmnist", "clients": 10,
     "pkl": "202606081817_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_auc_0.7704_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "chestmnist", "clients": 20,
     "pkl": "202606081924_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_auc_0.7628_drift_margin_enhanced.pkl"},

    # chestmnist + resnet
    {"model": "resnet", "dataset": "chestmnist", "clients": 5,
     "pkl": "202606082158_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_auc_0.7753_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "chestmnist", "clients": 10,
     "pkl": "202606090028_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_auc_0.7707_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "chestmnist", "clients": 20,
     "pkl": "202606090301_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_auc_0.7693_drift_margin_enhanced.pkl"},

    # cifar10 + alexnet
    {"model": "alexnet", "dataset": "cifar10", "clients": 5,
     "pkl": "202606090359_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.8998_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar10", "clients": 10,
     "pkl": "202606090459_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.8986_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar10", "clients": 20,
     "pkl": "202606090559_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.8932_drift_margin_enhanced.pkl"},

    # cifar10 + resnet
    {"model": "resnet", "dataset": "cifar10", "clients": 5,
     "pkl": "202606090756_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.9387_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar10", "clients": 10,
     "pkl": "202606090954_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.9349_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar10", "clients": 20,
     "pkl": "202606091153_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.9250_drift_margin_enhanced.pkl"},

    # cifar100 + alexnet
    {"model": "alexnet", "dataset": "cifar100", "clients": 5,
     "pkl": "202606091253_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.6716_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar100", "clients": 10,
     "pkl": "202606091354_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.6573_drift_margin_enhanced.pkl"},
    {"model": "alexnet", "dataset": "cifar100", "clients": 20,
     "pkl": "202606091455_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.6304_drift_margin_enhanced.pkl"},

    # cifar100 + resnet
    {"model": "resnet", "dataset": "cifar100", "clients": 5,
     "pkl": "202606091653_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.7248_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar100", "clients": 10,
     "pkl": "202606091853_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_10_fra_1.0000_acc_0.7108_drift_margin_enhanced.pkl"},
    {"model": "resnet", "dataset": "cifar100", "clients": 20,
     "pkl": "202606092054_reg_ablation_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_20_fra_1.0000_acc_0.7013_drift_margin_enhanced.pkl"},
]


def get_model_path(exp_config):
    """根据实验配置获取模型文件路径"""
    pkl = exp_config["pkl"]
    model_type = exp_config["model"]
    dataset = exp_config["dataset"]
    return os.path.join(PROJECT_ROOT, "save", model_type, dataset, pkl)


def has_full_weights(model_path):
    """
    检测 checkpoint 是否包含完整网络权重。

    训练脚本 main.py 的 _optimize_model_storage(watermark_only=True) 会把模型
    保存成"仅水印参数"格式（只含 _watermark_only / watermark_values /
    client_watermark_values），这类文件不含任何网络权重，
    无法用于剪枝 / 微调攻击，必须跳过。

    这里直接扫描 pkl 二进制内容判断，避免依赖 torch（批量脚本本身无需 torch）。

    Args:
        model_path: 模型文件路径

    Returns:
        bool: True 表示含完整权重
    """
    try:
        with open(model_path, 'rb') as f:
            data = f.read()
    except OSError:
        return False

    if b'_watermark_only' not in data:
        # 没有水印标记，视为完整权重
        return True

    # 含水印标记：必须同时含至少一个网络权重键
    weight_keys = (b'features.0.conv.weight', b'conv1.weight', b'fc.weight',
                   b'classifier.weight')
    return any(k in data for k in weight_keys)


def resolve_model_path(exp_config, verbose=True):
    """
    解析模型文件路径。

    优先使用配置中硬编码的 pkl 文件名；若该文件不存在或只含水印参数
    （训练产物被清理 / 被覆盖 / 保存格式变更），则在
    save/{model}/{dataset}/ 下按客户端数自动挑选最合适的候选模型。

    挑选规则:
      1. 必须包含 '_cn_{clients}_'
      2. 必须包含完整网络权重（排除"仅水印参数"格式的 checkpoint）
      3. 优先水印模型 ('wm_enhanced')
      4. 训练轮数越多越好 (ep_150 > ep_100 > ...)
      5. 同条件下取时间戳最新的一个
    """
    model_path = get_model_path(exp_config)
    if os.path.exists(model_path) and has_full_weights(model_path):
        return model_path, None

    model_type = exp_config["model"]
    dataset = exp_config["dataset"]
    clients = exp_config["clients"]
    model_dir = os.path.join(PROJECT_ROOT, "save", model_type, dataset)

    if not os.path.isdir(model_dir):
        return model_path, f"模型目录不存在: {model_dir}"

    cn_tag = f"_cn_{clients}_"
    candidates = [f for f in os.listdir(model_dir)
                  if f.endswith(".pkl") and cn_tag in f]

    if not candidates:
        return model_path, (f"在 {model_dir} 下找不到客户端数为 {clients} 的模型 "
                            f"(匹配 '{cn_tag}')")

    # 过滤掉只含水印参数、无法用于攻击的 checkpoint
    usable = [f for f in candidates if has_full_weights(os.path.join(model_dir, f))]
    if not usable:
        return model_path, (
            f"{model_dir} 下客户端数为 {clients} 的模型均只含水印参数、"
            f"不含网络权重，无法用于攻击实验（共 {len(candidates)} 个候选）")

    def sort_key(name):
        is_wm = 1 if "wm_enhanced" in name else 0
        ep_match = re.search(r"_ep_(\d+)_", name)
        epochs = int(ep_match.group(1)) if ep_match else -1
        # 训练脚本的时间戳格式为 %Y%m%d%H%M (12位)
        ts_match = re.match(r"(\d{12})", name)
        ts = ts_match.group(1) if ts_match else ""
        return (is_wm, epochs, ts)

    best = max(usable, key=sort_key)
    resolved = os.path.join(model_dir, best)

    note = (f"配置中的模型不可用（不存在或仅含水印参数），已自动回退到: {best}\n"
            f"    原配置: {exp_config['pkl']}\n"
            f"    当前使用: {resolved}")
    if verbose:
        print(f"  ⚠️ {note}")
    return resolved, note


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
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
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
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
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
        model_path, note = resolve_model_path(exp)
        exp_name = f"{exp['model']}_{exp['dataset']}_{exp['clients']}clients"

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不可用: {model_path}")
            print(f"        原因: {note}")
            results.append({
                "index": i, "name": exp_name, "model": exp["model"],
                "dataset": exp["dataset"], "clients": exp["clients"],
                "pkl": exp["pkl"], "success": False,
                "result": f"模型文件不可用: {note or '文件不存在'}"
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
            "pkl": exp["pkl"], "success": success, "result": result,
            "used_pkl": os.path.basename(model_path)
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
        model_path, note = resolve_model_path(exp)
        exp_name = f"{exp['model']}_{exp['dataset']}_{exp['clients']}clients"

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        if not os.path.exists(model_path):
            print(f"  警告: 模型文件不可用: {model_path}")
            print(f"        原因: {note}")
            results.append({
                "index": i, "name": exp_name, "model": exp["model"],
                "dataset": exp["dataset"], "clients": exp["clients"],
                "pkl": exp["pkl"], "success": False,
                "result": f"模型文件不可用: {note or '文件不存在'}"
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
            "pkl": exp["pkl"], "success": success, "result": result,
            "used_pkl": os.path.basename(model_path)
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


def save_results_to_file(results, attack_type, save_dir=None, extra_config=None):
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

        if extra_config:
            f.write("实验配置:\n")
            for key, value in extra_config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n{'='*80}\n\n")

        for r in results:
            status = "成功" if r["success"] else f"失败: {r['result']}"
            f.write(f"[{r['index']}] {r['name']}\n")
            f.write(f"  模型类型: {r['model']}\n")
            f.write(f"  数据集: {r['dataset']}\n")
            f.write(f"  客户端数: {r['clients']}\n")
            f.write(f"  配置模型文件: {r['pkl']}\n")
            if r.get("used_pkl") and r["used_pkl"] != r["pkl"]:
                f.write(f"  实际使用文件: {r['used_pkl']} (自动回退)\n")
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
        # 文件名格式: finetune_attack_{YYYYMMDDHHMMSS}_{dataset}_paper.csv
        match = re.search(r'finetune_attack_\d{14}_(\w+)_paper\.csv', filename)
        if match:
            dataset = match.group(1)
        else:
            # 回退: 从CSV头注释 "# Finetune Attack Results - DATASET" 解析
            dataset = "unknown"
            try:
                with open(filepath, 'r', encoding='utf-8-sig') as fh:
                    for line in fh:
                        if line.startswith('#'):
                            m2 = re.search(r'-\s*([A-Za-z0-9]+)\s*$', line.strip())
                            if m2:
                                dataset = m2.group(1).lower()
                                break
                        else:
                            break
            except Exception:
                pass

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
        save_results_to_file(pruning_results, "pruning", extra_config={
            "攻击类型": "剪枝攻击",
            "模型路径模式": "save/{model}/{dataset}/{pkl}",
            "自编码器目录": args.autoencoder_dir,
            "其他参数": "默认",
        })

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
        save_results_to_file(finetune_results, "finetune", extra_config={
            "攻击类型": "微调攻击",
            "模型路径模式": "save/{model}/{dataset}/{pkl}",
            "自编码器目录": args.autoencoder_dir,
            "微调轮数": args.finetune_epochs,
            "学习率": args.learning_rate,
            "批次大小": args.batch_size,
            "保存模式": args.save_mode,
        })

    # 汇总论文数据
    if args.attack_type in ['finetune', 'both']:
        collect_paper_results()

    print(f"\n{'='*80}")
    print("批量实验全部完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量攻击实验脚本')
    parser.add_argument('--attack_type', type=str, default='both',
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
