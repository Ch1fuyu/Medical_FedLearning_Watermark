"""
自动化运行脚本 - 手动配置参数运行 main.py

使用方法:
    python utils/run_sweep.py

只需在下面的遍历维度里定义你要变化的参数，其他使用 args.py 的默认值

遍历维度
--------
每个维度（DATASETS / MODELS / CLIENT_NUMS / GRAD_TARGET_RATIOS）都会与其他维度
**笛卡尔积**组合，生成全部实验。要临时关掉某个维度，把对应列表设为 ``None``。

    DATASETS      = ['cifar10', 'cifar100']   -> None 表示不遍历
    MODELS        = ['alexnet', 'resnet']      -> None 表示不遍历
    CLIENT_NUMS   = ['5', '10', '20']          -> None 表示不遍历
    GRAD_TARGET_RATIOS = [...]                 -> None 表示不遍历

watermark_grad_target_ratio 遍历
--------------------------------
``--watermark_grad_target_ratio`` 控制**水印区梯度被压制到什么程度**：

- ``target_ratio = 0``：**完全关闭梯度缩放**。``models/losses/multi_loss.py``
  里判断 ``target_ratio > 0`` 不成立，capping 整段被跳过，水印区梯度
  原样保留（只受优化器本身影响）。
- ``target_ratio > 0``：当水印区/非水印区梯度比例超过该阈值时，把水印区
  梯度等比缩放，压到 ``target_ratio`` 倍（缩放系数下限 0.1，见 ``multi_loss.py``）。

因此 ratio 越大 = 水印区被压得越少 = 水印信号保留越多，但主任务精度
受任务梯度推动得也越多。这个维度扫的就是这条**精度-鲁棒性权衡曲线**。

注意 clamp 下限
--------------
缩放系数被 ``clamp(min=0.1, max=1.0)`` 限制，因此：

- ``target_ratio >= 0.1`` 时：能精确达成目标比例；
- ``target_ratio = 0.1`` 时：恰好在边界上，原始比例 R 若 > 1.0 则
  实际达成 0.1*R（略高于 0.1）。

解读结果时请对照 Excel 里的 ``wm_grad_ratio_mean``（capping 前）与
``wm_grad_ratio_after_cap_mean``（capping 后）两列。

输出目录
--------
每个实验写到**自己的目录**，互不覆盖，事后可直接按目录名区分 ratio：

    save/excel/sweep/<实验名>/metrics_<model>_<dataset>_<时间戳>.xlsx
    logs/sweep_<实验名>.logs

其中 ``<实验名>`` 形如 ``ratio_off_cifar10_alexnet_5clients``
（``ratio_off`` 表示 ratio=0，即完全关闭梯度缩放）。

（Excel 文件名本身只含 model/dataset/时间戳，**不含 ratio**，
所以必须靠独立目录来区分。）
"""

import argparse
import itertools
import os
import subprocess
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
# 当前调查: watermark_grad_target_ratio 对水印鲁棒性的影响
# 固定 optim=adam / lr=0.001，遍历 水印梯度比例 × 数据集 × 模型 × 客户端数

# ---- 所有实验共享的固定配置 ----
BASE_CONFIG = {
    '--optim': 'adam',
    '--lr': '0.001',
}

# ---- 遍历维度: 水印梯度比例 × 数据集 × 模型 × 客户端数量 ----
# 注意: BASE_CONFIG 里不再写 '--watermark_grad_target_ratio'，
# 该参数由下面的 GRAD_TARGET_RATIOS 维度统一控制。
#
# GRAD_TARGET_RATIOS 的取值含义:
#   '0'   -> 完全关闭梯度缩放（capping 整段跳过）
#   '0.1' -> 强压制（水印区梯度压到非水印区的 10%）
#   '0.3' -> 默认值
#   '0.5' -> 弱压制
#   '0.8' -> 极弱压制（几乎只压制超出部分）
GRAD_TARGET_RATIOS = ['0', '0.1', '0.3', '0.5', '0.8']

DATASETS = ['cifar10', 'cifar100', 'chestmnist']
MODELS = ['alexnet', 'resnet']
CLIENT_NUMS = ['5', '10', '20']


def _ratio_tag(ratio: str) -> str:
    """把 ratio 转成适合放进目录名/文件名的短标签。

    ``'0'`` -> ``off``（完全关闭梯度缩放，便于一眼识别）
    ``'0.1'`` -> ``r0.1``
    """
    if float(ratio) == 0.0:
        return 'off'
    return f'r{ratio}'


def build_experiments():
    """按各遍历维度的笛卡尔积生成实验列表。

    维度列表为 None 时，该维度不参与遍历（只取唯一值）。

    Returns:
        实验字典列表（每个含 'name' 与命令行参数）。
    """
    def _expand(dim, fallback):
        """None -> 单元素列表（不遍历）；否则原样返回。"""
        return fallback if dim is None else list(dim)

    ratios = _expand(GRAD_TARGET_RATIOS, [None])
    datasets = _expand(DATASETS, [None])
    models = _expand(MODELS, [None])
    client_nums = _expand(CLIENT_NUMS, [None])

    experiments = []
    for ratio, ds, model, cn in itertools.product(
            ratios, datasets, models, client_nums):
        name_parts = []
        params = {}

        if ratio is not None:
            params['--watermark_grad_target_ratio'] = str(ratio)
            name_parts.append(f'ratio_{_ratio_tag(ratio)}')
        if ds is not None:
            params['--dataset'] = str(ds)
            name_parts.append(str(ds))
        if model is not None:
            params['--model_name'] = str(model)
            name_parts.append(str(model))
        if cn is not None:
            params['--client_num'] = str(cn)
            name_parts.append(f'{cn}clients')

        name = '_'.join(name_parts) or 'exp'

        # 每个实验写到自己的目录，避免多轮 ratio 扫描的结果互相覆盖，
        # 也让事后能一眼看出某个 Excel 属于哪个 ratio。
        # （Excel 文件名本身只含 model/dataset/时间戳，不含 ratio。）
        params['--save_excel_dir'] = f'save/excel/sweep/{name}'
        params['--log_file'] = f'logs/sweep_{name}.logs'

        experiments.append({
            'name': name,
            **BASE_CONFIG,
            **params,
        })

    return experiments


#: 兼容旧写法：模块级直接暴露实验列表
EXPERIMENTS = build_experiments()
# =======================================================================

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
    cli = argparse.ArgumentParser(
        description='批量实验脚本')
    cli.add_argument('--dry_run', action='store_true',
                     help='只打印将要执行的命令，不实际运行')
    cli.add_argument('--yes', '-y', action='store_true',
                     help='跳过确认，直接开始运行')
    cli_args = cli.parse_args()

    print("=" * 70)
    print("联邦学习自动化实验工具")
    print("=" * 70)

    experiments = build_experiments()

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
            print(f"       {k} {v}")
    print("-" * 50)
    print(f"其他参数将使用 utils/args.py 的默认值")
    print("-" * 50)

    if cli_args.dry_run:
        print("\n--dry_run: 只显示命令，不执行。")
        for i, exp in enumerate(experiments, 1):
            cmd = build_command(exp)
            print(f"  {i}. {' '.join(cmd)}")
        return

    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'sweep_{timestamp}.log'

    # 确认是否继续
    if not cli_args.yes:
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
