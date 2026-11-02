#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重训脚本 - 重新生成 utils/run_batch_attack.py 中 EXPERIMENTS 配置的模型

背景
----
main.py 的 _optimize_model_storage() 曾默认 watermark_only=True，
只会把水印区域参数落盘，checkpoint 内不含网络权重，
导致剪枝 / 微调攻击无法加载模型（commit 0a901ec 引入）。

该问题已修复（main.py 现在默认保存完整权重），但**已经训练出来的旧模型
不会自动升级**——必须重新训练才能得到含完整权重的 checkpoint。

本脚本按 run_batch_attack.py 的 EXPERIMENTS 配置，逐个重训对应模型。
训练完成后，把新生成的文件名回填到 run_batch_attack.py 的 EXPERIMENTS 中即可。

使用方法
--------
    # 预检：只检查哪些模型缺失/不可用，不训练
    python utils/retrain_batch_models.py --check

    # 训练全部缺失的模型
    python utils/retrain_batch_models.py

    # 只训练指定索引范围
    python utils/retrain_batch_models.py --start_idx 0 --end_idx 3

    # 训练后自动回填 EXPERIMENTS 配置
    python utils/retrain_batch_models.py --update-config
"""

import os
import re
import sys
import subprocess
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 复用 run_batch_attack.py 的实验配置，避免两处配置不一致
from utils.run_batch_attack import (  # noqa: E402
    EXPERIMENTS,
    get_model_path,
    has_full_weights,
    PROJECT_ROOT as _PR,
)

# 重训使用的训练脚本与参数
TRAIN_SCRIPT = "main.py"
# 与 EXPERIMENTS 中模型命名一致的训练配置：
#   文件名含 reg_ablation 且带 drift_margin 后缀 -> 由 reg_ablation_experiment.py 产出
TRAIN_SCRIPT_REG = "reg_ablation_experiment.py"

BASE_ARGS = [
    "--optim", "adam",
    "--lr", "0.001",
    "--epochs", "150",
    "--local_ep", "2",
    "--batch_size", "128",
    "--frac", "1.0",
]


def is_reg_ablation(pkl_name):
    """判断该配置对应的模型应由 reg_ablation_experiment.py 产出"""
    return "reg_ablation" in pkl_name and "drift_margin" in pkl_name


def build_train_cmd(exp):
    """构造训练命令"""
    script = TRAIN_SCRIPT_REG if is_reg_ablation(exp["pkl"]) else TRAIN_SCRIPT
    cmd = [sys.executable, script]
    cmd += BASE_ARGS
    cmd += [
        "--dataset", exp["dataset"],
        "--model_name", exp["model"],
        "--client_num", str(exp["clients"]),
    ]
    if script == TRAIN_SCRIPT_REG:
        # 漂移 + 裕量（与 drift_margin 后缀对应）
        cmd += ["--use_drift_reg", "True", "--use_margin_reg", "True"]
    return cmd


def list_dir_models(exp):
    """列出该配置目录下已有的模型文件"""
    model_dir = os.path.join(PROJECT_ROOT, "save", exp["model"], exp["dataset"])
    if not os.path.isdir(model_dir):
        return []
    cn_tag = f"_cn_{exp['clients']}_"
    return sorted(f for f in os.listdir(model_dir)
                  if f.endswith(".pkl") and cn_tag in f)


def newest_full_model(exp):
    """找出该配置下最新的、含完整权重的模型"""
    model_dir = os.path.join(PROJECT_ROOT, "save", exp["model"], exp["dataset"])
    if not os.path.isdir(model_dir):
        return None
    cn_tag = f"_cn_{exp['clients']}_"
    cands = [f for f in os.listdir(model_dir)
             if f.endswith(".pkl") and cn_tag in f
             and has_full_weights(os.path.join(model_dir, f))]
    if not cands:
        return None

    def sort_key(name):
        ep = re.search(r"_ep_(\d+)_", name)
        ts = re.match(r"(\d{12})", name)
        return (int(ep.group(1)) if ep else -1,
                ts.group(1) if ts else "")

    return max(cands, key=sort_key)


def check_only(start_idx=0, end_idx=None):
    """预检：报告每个实验的模型状态"""
    if end_idx is None:
        end_idx = len(EXPERIMENTS)

    print(f"\n{'='*90}")
    print(f"批量攻击模型预检 (实验 {start_idx+1} ~ {end_idx})")
    print(f"{'='*90}")

    ok, need = 0, []
    for i, exp in enumerate(EXPERIMENTS[start_idx:end_idx], start=start_idx+1):
        path = get_model_path(exp)
        name = f"{exp['model']}/{exp['dataset']}/K={exp['clients']}"
        if os.path.exists(path) and has_full_weights(path):
            print(f"  [OK ] [{i:2}] {name}")
            ok += 1
        else:
            reason = "文件不存在" if not os.path.exists(path) else "仅含水印参数、缺网络权重"
            print(f"  [需重训] [{i:2}] {name}  ({reason})")
            need.append((i, exp))

    print(f"\n{'='*90}")
    print(f"可直接使用: {ok}/{end_idx - start_idx}    需要重训: {len(need)}")
    print(f"{'='*90}")
    return need


def train_one(exp, verbose=True):
    """训练单个模型，返回生成的文件名（失败返回 None）"""
    cmd = build_train_cmd(exp)
    name = f"{exp['model']}/{exp['dataset']}/K={exp['clients']}"

    if verbose:
        print(f"\n{'-'*90}")
        print(f"训练 {name}")
        print(f"命令: {' '.join(cmd)}")
        print(f"{'-'*90}")

    # 记录训练前的文件列表，用于定位新生成的产物
    before = set(list_dir_models(exp))

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败 (退出码 {e.returncode}): {name}")
        return None

    after = set(list_dir_models(exp))
    new_files = sorted(after - before)
    if not new_files:
        # 时间戳精度导致文件名相同，取最新的一个
        candidates = [f for f in after if has_full_weights(
            os.path.join(PROJECT_ROOT, "save", exp["model"], exp["dataset"], f))]
        if not candidates:
            print(f"❌ 未找到含完整权重的产物: {name}")
            return None
        new_files = [max(candidates, key=lambda n: n)]

    for f in new_files:
        full = os.path.join(PROJECT_ROOT, "save", exp["model"], exp["dataset"], f)
        flag = "含完整权重 ✓" if has_full_weights(full) else "仅含水印参数 ✗"
        print(f"  产物: {f}  [{flag}]")

    # 返回最新的完整权重文件
    fulls = [f for f in new_files if has_full_weights(
        os.path.join(PROJECT_ROOT, "save", exp["model"], exp["dataset"], f))]
    return max(fulls) if fulls else None


def update_config(mapping, config_path=None):
    """
    回填 EXPERIMENTS 配置中的 pkl 文件名。

    mapping: {实验索引(0-based): 新文件名}
    """
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "run_batch_attack.py")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    updated = 0
    for idx, new_name in mapping.items():
        exp = EXPERIMENTS[idx]
        old = exp["pkl"]
        if old == new_name:
            continue
        # 替换该实验条目中的 pkl 值
        pattern = re.compile(
            r'("model"\s*:\s*"' + re.escape(exp["model"]) + r'"\s*,\s*'
            r'"dataset"\s*:\s*"' + re.escape(exp["dataset"]) + r'"\s*,\s*'
            r'"clients"\s*:\s*' + str(exp["clients"]) + r'\s*,\s*'
            r'"pkl"\s*:\s*")([^"]*)(")'
        )
        new_content, n = pattern.subn(
            lambda m: m.group(1) + new_name + m.group(3), content)
        if n:
            content = new_content
            updated += 1
            print(f"  [{idx+1}] {exp['model']}/{exp['dataset']}/K={exp['clients']}")
            print(f"      {old}")
            print(f"   -> {new_name}")
        else:
            print(f"  [{idx+1}] 未找到匹配的条目，需手动更新: {old}")

    if updated:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ 已更新 {updated} 条配置: {config_path}")
    else:
        print("\n⚠️  没有更新任何配置")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description='重训 run_batch_attack.py 配置的模型（保存完整权重）')
    parser.add_argument('--check', action='store_true',
                        help='仅预检，不训练')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始实验索引 (0-based)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束实验索引 (0-based, 不含)')
    parser.add_argument('--update-config', action='store_true',
                        help='训练完成后自动回填 run_batch_attack.py 的 EXPERIMENTS')
    parser.add_argument('--retrain-all', action='store_true',
                        help='强制重训全部（默认只训练缺失/不可用的）')
    args = parser.parse_args()

    if args.end_idx is None:
        args.end_idx = len(EXPERIMENTS)

    need = check_only(args.start_idx, args.end_idx)

    if args.check:
        print("\n如需重训，请去掉 --check 参数运行。")
        return

    if not need and not args.retrain_all:
        print("\n所有模型均可用，无需重训。")
        return

    targets = need if not args.retrain_all else [
        (i, EXPERIMENTS[i - 1]) for i in range(args.start_idx + 1, args.end_idx + 1)
    ]

    print(f"\n将重训 {len(targets)} 个模型")
    response = input("确认开始? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return

    mapping = {}
    for i, exp in targets:
        result = train_one(exp)
        if result:
            mapping[exp["pkl"]] = result

    print(f"\n{'='*90}")
    print(f"重训完成: {len(mapping)}/{len(targets)} 成功")
    print(f"{'='*90}")

    if mapping and args.update_config:
        print("\n回填 EXPERIMENTS 配置...")
        update_config(mapping)

    if mapping:
        print("\n下一步：")
        print("  1. 若未加 --update-config，请手动把上面的新文件名填入")
        print("     utils/run_batch_attack.py 的 EXPERIMENTS")
        print("  2. 重跑: python utils/run_batch_attack.py")


if __name__ == '__main__':
    main()
