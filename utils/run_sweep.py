"""
自动化批量实验脚本 —— 顺序执行多次 ``python main.py``。

为什么需要 ``--no_config_center``
--------------------------------
``utils/args.py`` 里的 :data:`EXPERIMENT_CONFIG` 是"给手改的"实验配置中心，
优先级是 **命令行 > 配置中心 > argparse 默认值**。

批量脚本会把参数逐个传给 ``main.py`` 子进程。如果不显式关闭配置中心，
配置块里的 ``'collusion_only': True`` / ``'epochs': 3`` / ``'client_num': 10``
等设置会污染**每一次**运行，导致本脚本的遍历维度失效
（例如想扫 5/10/20 个客户端，结果每次都被压回 10）。

因此本脚本**始终**给每个子进程加 ``--no_config_center``：
所有参数只来自本文件里的配置 + argparse 默认值。

两种实验模式
------------
``MODE = 'baseline'``（普通水印基线）
    扫数据集 × 模型 × 客户端数。``client_num`` 直接生效，
    密钥矩阵从 ``save/key_matrix/<model>/client<N>`` 读取。

``MODE = 'collusion'``（抗合谋）
    扫数据集 × 模型 × **合谋规模 K**。
    注意：抗合谋模式下 ``client_num`` 由**抗合谋配置表**权威决定
    （``config/collusion_presets*.json``），命令行给的 ``--client_num``
    会被覆盖 —— 这是设计如此，因为密钥矩阵目录与 (K, 模型) 必须配套。
    所以抗合谋模式的遍历维度是 K，而不是 client_num。

用法
----
1. 直接改下面的 ``MODE`` / ``BASE_CONFIG`` / 遍历维度，然后：

       python utils/run_sweep.py

2. 只看会跑哪些实验（不实际执行）：

       python utils/run_sweep.py --dry_run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 获取项目根目录（run_sweep.py 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAIN_PY_PATH = PROJECT_ROOT / 'main.py'

# ======================================================================
#  在这里定义你的实验配置
# ======================================================================
# 每个字典代表一次实验，按顺序执行。
# 只需填写需要变化的参数，未填的用 argparse 默认值
# （配置中心已被 --no_config_center 关闭）。
#
# 两种模式：
#   'baseline' : 普通水印基线，扫 数据集 × 模型 × 客户端数
#   'collusion': 抗合谋，扫 数据集 × 模型 × 合谋规模 K
# ======================================================================

#: 实验模式：'baseline' | 'collusion'
MODE = 'baseline'

# ---- 所有实验共享的固定配置 ----
BASE_CONFIG = {
    '--optim': 'adam',
    '--lr': '0.001',
    '--watermark_grad_target_ratio': '0.3',
    '--epochs': '10',
    '--local_ep': '1',
    '--batch_size': '128',
}

# ---- 遍历维度 ----
DATASETS = ['cifar10', 'cifar100']
MODELS = ['alexnet', 'resnet']

#: 'baseline' 模式：客户端数（直接传给 --client_num）
CLIENT_NUMS = ['5', '10', '20']

#: 'collusion' 模式：合谋规模 K（传给 --collusion_size）
COLLUSION_SIZES = ['2', '3', '4']


def build_experiments(mode):
    """根据模式生成实验列表。

    Args:
        mode: 'baseline' | 'collusion'。

    Returns:
        实验字典列表（每个含 'name' 与命令行参数）。
    """
    if mode == 'baseline':
        return [
            {
                'name': f'{ds}_{model}_{k}clients',
                **BASE_CONFIG,
                '--dataset': ds,
                '--model_name': model,
                '--client_num': k,
            }
            for ds in DATASETS
            for model in MODELS
            for k in CLIENT_NUMS
        ]

    if mode == 'collusion':
        # 抗合谋：--collusion_only 一键切到抗合谋评测模式
        # (关闭单客户端泄漏、开启合谋检测、切换密钥矩阵目录)
        #
        # 注意：这里**不传** --client_num —— 抗合谋模式下客户端数由
        # 抗合谋配置表 (config/collusion_presets*.json) 决定，
        # 命令行给了也会被覆盖。遍历维度因此是合谋规模 K。
        return [
            {
                'name': f'{ds}_{model}_K{k}',
                **BASE_CONFIG,
                '--dataset': ds,
                '--model_name': model,
                '--collusion_only': '',
                '--collusion_size': k,
                '--collusion_leak_stage': 'post_training',
            }
            for ds in DATASETS
            for model in MODELS
            for k in COLLUSION_SIZES
        ]

    raise ValueError(f"未知 MODE: {mode!r}（支持 'baseline' / 'collusion'）")


# ======================================================================
#  以下为执行逻辑，通常不需要修改
# ======================================================================

# GPU 配置
GPU = '0'

# 日志输出目录
LOG_DIR = PROJECT_ROOT / 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


def build_command(exp_config):
    """
    构建命令行参数列表

    Args:
        exp_config: 实验配置字典（只包含变化的参数）

    Returns:
        cmd: 命令列表
    """
    cmd = [sys.executable, str(MAIN_PY_PATH)]

    # 关键：关闭实验配置中心，避免配置块污染每一次运行，
    # 使本脚本的遍历维度失效。
    cmd.append('--no_config_center')

    # 添加实验特定参数
    for key, value in exp_config.items():
        if key == 'name':  # 跳过名称字段
            continue
        cmd.append(key)
        # 值为空字符串的开关（如 --collusion_only）只传键名
        if value != '':
            cmd.append(str(value))

    return cmd


def run_experiment(cmd, exp_config, experiment_idx, total, log_file):
    """
    运行单个实验
    """
    exp_name = exp_config.get('name', f'实验 {experiment_idx}')

    print("\n" + "=" * 70)
    print(f"[{exp_name}] ({experiment_idx}/{total})")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 70)

    try:
        # 打开日志文件记录输出
        with open(str(log_file), 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"[{exp_name}] ({experiment_idx}/{total}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 70 + "\n")

        # 运行命令（在项目根目录执行）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            cwd=str(PROJECT_ROOT)
        )

        # 实时输出并写入日志
        for line in process.stdout:
            print(line, end='')
            with open(str(log_file), 'a', encoding='utf-8') as f:
                f.write(line)

        process.wait()

        success = process.returncode == 0

        with open(str(log_file), 'a', encoding='utf-8') as f:
            f.write(f"\n[{exp_name}] 完成 - 返回码: {process.returncode}\n")

        return success

    except Exception as e:
        print(f"运行出错: {e}")
        with open(str(log_file), 'a', encoding='utf-8') as f:
            f.write(f"\n[{exp_name}] 错误: {e}\n")
        return False


def main():
    cli = argparse.ArgumentParser(
        description='批量实验脚本（关闭 utils/args.py 的实验配置中心）')
    cli.add_argument('--dry_run', action='store_true',
                     help='只打印将要执行的命令，不实际运行')
    cli.add_argument('--mode', choices=['baseline', 'collusion'], default=None,
                     help=f'实验模式（默认用脚本里的 MODE={MODE!r}）')
    cli.add_argument('--yes', '-y', action='store_true',
                     help='跳过确认，直接开始运行')
    args = cli.parse_args()

    mode = args.mode or MODE
    experiments = build_experiments(mode)

    print("=" * 70)
    print("联邦学习自动化实验工具")
    print("=" * 70)
    print(f"模式: {mode}")

    # 检查是否有实验配置
    if not experiments:
        print("\n错误: 请在脚本中定义至少一个实验配置")
        return

    total = len(experiments)

    print(f"\n将运行 {total} 个实验:")
    print("-" * 50)
    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get('name', f'实验 {i}')
        params = {k: v for k, v in exp.items() if k != 'name'}
        print(f"  {i}. [{exp_name}]")
        for k, v in params.items():
            print(f"       {k} {v}" if v != '' else f"       {k}")
    print("-" * 50)
    print(f"未列出的参数使用 argparse 默认值")
    print(f"（实验配置中心 EXPERIMENT_CONFIG 已通过 --no_config_center 关闭）")
    if mode == 'collusion':
        print("-" * 50)
        print("注意: 抗合谋模式下 client_num 由抗合谋配置表决定"
              "(config/collusion_presets*.json)，")
        print("      命令行传 --client_num 也会被覆盖，因此遍历维度是合谋规模 K。")
    print("-" * 50)

    if args.dry_run:
        print("\n--dry_run: 只显示命令，不执行。")
        for i, exp in enumerate(experiments, 1):
            cmd = build_command(exp)
            print(f"  {i}. {' '.join(cmd)}")
        return

    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'sweep_{mode}_{timestamp}.log'

    # 确认是否继续
    if not args.yes:
        response = input(f"\n确认开始运行? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return

    print(f"\n日志将保存到: {log_file}")
    print("开始运行...\n")

    # 统计结果
    success_count = 0
    fail_count = 0

    for i, exp in enumerate(experiments, 1):
        cmd = build_command(exp)
        success = run_experiment(cmd, exp, i, total, log_file)

        if success:
            success_count += 1
        else:
            fail_count += 1

        print(f"\n进度: {i}/{total} | 成功: {success_count} | 失败: {fail_count}")

    # 总结
    print("\n" + "=" * 70)
    print("实验完成!")
    print(f"总计: {total} | 成功: {success_count} | 失败: {fail_count}")
    print(f"日志文件: {log_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
