"""
自动化运行脚本 - 手动配置参数运行 main.py

使用方法:
    python run_sweep.py
    
只需在 EXPERIMENTS 列表中定义你要变化的参数，其他使用 args.py 的默认值
"""

import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

# 获取项目根目录（run_sweep.py 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAIN_PY_PATH = PROJECT_ROOT / 'main.py'

# ========================= 在这里定义你的实验配置 =========================
# 每个字典代表一次实验，按顺序执行
# 只需填写需要变化的参数，其他自动使用 args.py 的默认值
#
# 优化器和学习率建议:
# - CIFAR-10/100 + AlexNet: Adam, lr=0.003 (小学习率适合复杂模型)
# - CIFAR-10/100 + ResNet: SGD, lr=0.01 (SGD在大数据集上泛化更好)
EXPERIMENTS = [
    # 实验配置：CIFAR × 2网络 × 3客户端数 = 12个实验

    # cifar10 + alexnet
    {
        'name': 'cifar10_alexnet_5clients',
        '--dataset': 'cifar10',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '5',
    },
    {
        'name': 'cifar10_alexnet_10clients',
        '--dataset': 'cifar10',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '10',
    },
    {
        'name': 'cifar10_alexnet_20clients',
        '--dataset': 'cifar10',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '20',
    },

    # cifar10 + resnet
    {
        'name': 'cifar10_resnet_5clients',
        '--dataset': 'cifar10',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '5',
    },
    {
        'name': 'cifar10_resnet_10clients',
        '--dataset': 'cifar10',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '10',
    },
    {
        'name': 'cifar10_resnet_20clients',
        '--dataset': 'cifar10',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '20',
    },

    # cifar100 + alexnet
    {
        'name': 'cifar100_alexnet_5clients',
        '--dataset': 'cifar100',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '5',
    },
    {
        'name': 'cifar100_alexnet_10clients',
        '--dataset': 'cifar100',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '10',
    },
    {
        'name': 'cifar100_alexnet_20clients',
        '--dataset': 'cifar100',
        '--model_name': 'alexnet',
        '--optim': 'adam',
        '--lr': '0.003',
        '--client_num': '20',
    },

    # cifar100 + resnet
    {
        'name': 'cifar100_resnet_5clients',
        '--dataset': 'cifar100',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '5',
    },
    {
        'name': 'cifar100_resnet_10clients',
        '--dataset': 'cifar100',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '10',
    },
    {
        'name': 'cifar100_resnet_20clients',
        '--dataset': 'cifar100',
        '--model_name': 'resnet',
        '--optim': 'sgd',
        '--lr': '0.01',
        '--client_num': '20',
    },
]
# =======================================================================

# GPU 配置
GPU = '0'

# 日志输出目录
LOG_DIR = PROJECT_ROOT / 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
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
    
    # 添加实验特定参数
    for key, value in exp_config.items():
        if key == 'name':  # 跳过名称字段
            continue
        cmd.append(key)
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
    print("=" * 70)
    print("联邦学习自动化实验工具")
    print("=" * 70)
    
    # 检查是否有实验配置
    if not EXPERIMENTS:
        print("\n错误: 请在脚本中定义至少一个实验配置 (EXPERIMENTS 列表)")
        print("\n示例配置格式:")
        print("""
EXPERIMENTS = [
    {
        'name': 'chestmnist_resnet',
        '--dataset': 'chestmnist',
        '--model_name': 'resnet',
    },
]
        """)
        return
    
    total = len(EXPERIMENTS)
    
    print(f"\n将运行 {total} 个实验:")
    print("-" * 50)
    for i, exp in enumerate(EXPERIMENTS, 1):
        exp_name = exp.get('name', f'实验 {i}')
        params = {k: v for k, v in exp.items() if k != 'name'}
        print(f"  {i}. [{exp_name}]")
        for k, v in params.items():
            print(f"       {k} {v}")
    print("-" * 50)
    print(f"其他参数将使用 utils/args.py 的默认值")
    print("-" * 50)
    
    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'sweep_{timestamp}.log'
    
    # 确认是否继续
    response = input(f"\n确认开始运行? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    print(f"\n日志将保存到: {log_file}")
    print("开始运行...\n")
    
    # 统计结果
    success_count = 0
    fail_count = 0
    
    for i, exp in enumerate(EXPERIMENTS, 1):
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
