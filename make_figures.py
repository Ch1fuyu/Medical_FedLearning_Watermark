# -*- coding: utf-8 -*-
"""
一键出图脚本 (论文 Figure 1~5)
============================

在实验机��上跑完 ``main.py`` 之后, 只需执行本脚本即可从原始 JSON 结果
直接生成论文用的 5 张图, 无需手工分两步跑整合与绘图。

    python make_figures.py

流程:
    save/trace/trace_results_*.json
    save/detection/watermark_deviation_detection_*.json
                ↓  1) utils.aggregate_results  (整合成 CSV)
    save/aggregate/*.csv
                ↓  2) utils.plot_paper_figures  (出图)
    save/figures/fig1~fig5.png

常用参数:
    --datasets chestmnist          只画指定数据集 (默认: 全部)
    --models alexnet,resnet        只画指定模型 (默认: 全部)
    --client-nums 5,10,20          只画指定客户端数 (默认: 全部)
    --top-n 6                      每个目录只取最新 6 个结果文件 (默认: 0 = 全部)
    --out-dir save/figures         图片输出目录 (默认: save/figures)

示例:
    # 只画 ChestMNIST 的 6 个实验 (AlexNet/ResNet x K=5/10/20)
    python make_figures.py --datasets chestmnist --client-nums 5,10,20

    # 先做一次全量出图
    python make_figures.py
"""

import argparse
import os
import sys

# --------------------------------------------------------------------------- #
# 重要: 必须在 import numpy 之前设置
#   本机 anaconda 环境自带的 MKL 在 AVX2 路径上会触发 "Windows fatal exception
#   0xc06d007e" (缺失 DLL), 表现为 numpy.dot / matplotlib 绘制时进程直接崩溃
#   且无 Python 异常。禁用 MKL 多线程调度即可绕开。
#   如在别的机器上运行, 该设置无副作用, 可安全保留。
# --------------------------------------------------------------------------- #
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# 全局常量
# --------------------------------------------------------------------------- #
DEFAULT_TRACE_DIR = os.path.join(PROJECT_ROOT, 'save', 'trace')
DEFAULT_DETECTION_DIR = os.path.join(PROJECT_ROOT, 'save', 'detection')
DEFAULT_AGG_DIR = os.path.join(PROJECT_ROOT, 'save', 'aggregate')
DEFAULT_FIG_DIR = os.path.join(PROJECT_ROOT, 'save', 'figures')


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def _split_lower(s):
    """'chestmnist, cifar10' -> {'chestmnist', 'cifar10'}; 空串 -> None"""
    return {x.strip().lower() for x in s.split(',') if x.strip()} or None


def _split_int(s):
    """'5,10,20' -> {5, 10, 20}; 空串 -> None"""
    vals = set()
    for x in s.split(','):
        x = x.strip()
        if x:
            vals.add(int(x))
    return vals or None


def _count_result_files(trace_dir, detection_dir):
    """统计两个目录里可用的结果文件数, 返回 (trace 数, detection 数)"""
    import glob
    n_trace = len(glob.glob(os.path.join(trace_dir, 'trace_results_*.json')))
    n_det = len(glob.glob(os.path.join(detection_dir,
                                       'watermark_deviation_detection_*.json')))
    return n_trace, n_det


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='一键整合实验结果并生成论文 Figure 1~5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--trace-dir', default=DEFAULT_TRACE_DIR,
                        help='trace 结果目录 (默认 save/trace)')
    parser.add_argument('--detection-dir', default=DEFAULT_DETECTION_DIR,
                        help='detection 结果目录 (默认 save/detection)')
    parser.add_argument('--agg-dir', default=DEFAULT_AGG_DIR,
                        help='中间 CSV 输出目录 (默认 save/aggregate)')
    parser.add_argument('--out-dir', default=DEFAULT_FIG_DIR,
                        help='图片输出目录 (默认 save/figures)')
    parser.add_argument('--datasets', default='',
                        help='只画指定数据集, 逗号分隔 (如 chestmnist)。默认全部')
    parser.add_argument('--models', default='',
                        help='只画指定模型, 逗号分隔 (如 alexnet,resnet)。默认全部')
    parser.add_argument('--client-nums', default='',
                        help='只画指定客户端数, 逗号分隔 (如 5,10,20)。默认全部')
    parser.add_argument('--top-n', type=int, default=0,
                        help='每个目录只取最新 N 个结果文件 (默认 0 = 全部)')
    parser.add_argument('--keep-duplicates', action='store_true',
                        help='不去重: 同一 (模型, 数据集, K) 的多次实验全部保留 '
                             '(默认每个配置只保留最新一次)')
    parser.add_argument('--skip-aggregate', action='store_true',
                        help='跳过整合步骤, 直接用已有 save/aggregate 的 CSV 画图')
    args = parser.parse_args(argv)

    print('=' * 70)
    print('  论文图表一键生成 (Figure 1~5)')
    print('=' * 70)

    # ----------------------------------------------------------------- #
    # 0) 检查输入文件
    # ----------------------------------------------------------------- #
    n_trace, n_det = _count_result_files(args.trace_dir, args.detection_dir)
    print(f'[INFO] trace 目录    : {args.trace_dir}  ({n_trace} 个文件)')
    print(f'[INFO] detection 目录: {args.detection_dir}  ({n_det} 个文件)')

    if n_trace == 0 or n_det == 0:
        print('\n[ERROR] 没有找到实验结果文件!')
        print('        请先在��台机器上跑完 main.py, 确认下面两个目录有结果:')
        print(f'          {args.trace_dir}')
        print(f'          {args.detection_dir}')
        return 1

    if not args.skip_aggregate:
        # ------------------------------------------------------------- #
        # 1) 整合
        # ------------------------------------------------------------- #
        print('\n' + '-' * 70)
        print('[STEP 1/2] 整合实验结果 (utils.aggregate_results)')
        print('-' * 70)

        from utils import aggregate_results as agg

        agg.aggregate(
            trace_dir=args.trace_dir,
            detection_dir=args.detection_dir,
            top_n=args.top_n,
            out_dir=args.agg_dir,
            datasets=_split_lower(args.datasets),
            models=_split_lower(args.models),
            client_nums=_split_int(args.client_nums),
            dedupe_latest=not args.keep_duplicates,
            verbose=True,
        )
    else:
        print('\n[STEP 1/2] 跳过整合 (--skip-aggregate), 使用已有 CSV')

    # ----------------------------------------------------------------- #
    # 2) 出图
    # ----------------------------------------------------------------- #
    print('\n' + '-' * 70)
    print('[STEP 2/2] 生成论文图表 (utils.plot_paper_figures)')
    print('-' * 70)

    from utils import plot_paper_figures as plot

    os.makedirs(args.out_dir, exist_ok=True)
    rc = plot.main([
        '--data-dir', args.agg_dir,
        '--out-dir', args.out_dir,
    ])
    if rc not in (0, None):
        print(f'[ERROR] 出图失败, 返回码 {rc}')
        return rc

    # ----------------------------------------------------------------- #
    # 3) 汇总
    # ----------------------------------------------------------------- #
    print('\n' + '=' * 70)
    print('  [OK] 全部完成! 图片已输出到:')
    print(f'         {args.out_dir}')
    print('=' * 70)

    try:
        import pandas as pd
        summary_path = os.path.join(args.agg_dir, 'summary_by_experiment.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            print(f'\n[INFO] 本次纳入的实验 ({len(df)} 组):')
            for _, row in df.iterrows():
                print(f'  - {row.get("model")} / {row.get("dataset")} '
                      f'/ K={row.get("client_num")}')
    except Exception as exc:  # noqa: BLE001
        print(f'[WARN] 汇总打印失败: {exc}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
