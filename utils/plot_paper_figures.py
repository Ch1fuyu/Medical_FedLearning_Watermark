# -*- coding: utf-8 -*-
"""
论文核心图表生成脚本 (4 张图, 两种攻击模式分开绘制)
====================================================

两种攻击模式:
    Mode A  水印区域扰动 (Watermark-region perturbation)
             对定制模型的水印区域注入高斯噪声后追踪, 使用 trace_results_*.json
             追踪任务 = 在 K 个客户端中定位泄漏源 (similarity ranking)
    Mode B  本地微调 (Local fine-tuning)
             追踪客户端本地训练后的模型, 使用 watermark_deviation_detection_*.json
             检测任务 = 判定该轮是否存在可疑恶意客户端

输出 (每张图只含一种模式):
    Figure 1  fig1_perturbation_accuracy.png   水印区域扰动的追踪准确率
    Figure 2  fig2_finetune_accuracy.png       本地微调的检测准确率
    Figure 3  fig3_perturbation_similarity.png  水印区域扰动的 top-1/top-2 相似度与 gap
    Figure 4  fig4_finetune_similarity.png      本地微调的 top-1/top-2 相似度与 gap

依赖 ``python -m utils.aggregate_results`` 生成的:
    - summary_by_experiment.csv   每个实验一行 (两种模式的准确率)
    - rounds_merged.csv           逐轮水印区域扰动的 top1/top2/margin
    - detection_margins.csv       逐轮本地微调的相似度与余量

用法:
    python -m utils.aggregate_results          # 先整合数据
    python -m utils.plot_paper_figures         # 再画图
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

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# 全局常量
# --------------------------------------------------------------------------- #

DATASET_ORDER = ['chestmnist', 'cifar10', 'cifar100']
MODEL_ORDER = ['alexnet', 'resnet']
K_ORDER = [5, 10, 20]

MODEL_LABEL = {'alexnet': 'AlexNet', 'resnet': 'ResNet18'}
DATASET_LABEL = {'chestmnist': 'ChestMNIST', 'cifar10': 'CIFAR-10',
                 'cifar100': 'CIFAR-100'}

# --------------------------------------------------------------------------- #
# 配色方案
# --------------------------------------------------------------------------- #
# 两个泄漏场景各用一个色系, 读者可凭颜色直接区分:
#   水印区域扰动 (pre) = 青绿系
#   本地微调     (post) = 玫瑰土系
# 同场景内的图共用同一色系 (fig1↔fig5, fig2↔fig4), 整体取淡雅的低饱和度色调。

PRE_COLOR = '#5BA79C'     # 水印区域扰动 · 主色 (中青绿)
POST_COLOR = '#C98C87'    # 本地微调     · 主色 (柔和玫瑰土)
SECOND_COLOR = '#9AA3AE'  # 次高相似度 (中性灰蓝)

# 三档高斯噪声强度 (main.py: noise_scale = 0.01 / 0.05 / 0.10)
# 与扰动场景同属青绿色系, 由浅到深表示噪声增强; 与 fig1 的柱子一一对应。
NOISE_LEVELS = [
    (1, 'weak',   0.01, '#9CCFC5'),
    (2, 'medium', 0.05, PRE_COLOR),
    (3, 'strong', 0.10, '#2F7369'),
]

# 两种模式的定义: (key, 中文名, 英文名, 颜色, 准确率列, 任务名)
#   pre  = 水印区域扰动 (Watermark-region perturbation)
#          泄漏的是下发的定制模型, 其水印区域被注入高斯噪声后追踪
#   post = 本地微调 (Local fine-tuning)
#          泄漏的是客户端本地训练后的模型, 追踪其水印偏移
PRE_MODE = ('pre', '水印区域扰动', 'Watermark-region perturbation', PRE_COLOR,
            'trace_accuracy', 'Tracing (source identification)')
POST_MODE = ('post', '本地微调', 'Local fine-tuning', POST_COLOR,
             'detection_accuracy', 'Detection (malicious client)')


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #

def setup_matplotlib():
    """统一视觉风格"""
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['savefig.dpi'] = 200
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.facecolor'] = 'white'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.25
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['legend.frameon'] = False
    return plt


def exp_label(model, dataset, k):
    return f'{MODEL_LABEL.get(model, model)}\n{DATASET_LABEL.get(dataset, dataset)}\nK={k}'


def readable_text_color(bg_hex):
    """
    根据背景色亮度挑选前景文字色 (黑或白), 保证淡雅配色下文字仍可读。

    依据 WCAG 相对亮度公式; 返回 '#FFFFFF' 或 '#2B2B2B'。
    """
    h = bg_hex.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    # 与白色对比度 < 4.5 时改用深色文字
    return '#FFFFFF' if (1.05 / (lum + 0.05)) >= 4.5 else '#2B2B2B'


def bar_value_label(ax, x, val, color, fontsize=7.0):
    """
    在柱子上标注百分比。

    淡雅配色下纯白字可能看不清, 故按柱子底色自动选白字或深字。
    柱子接近 y 轴上限时改为标在柱顶上方 —— 此时文字落在白色背景上,
    必须用深色字, 否则深色柱的白色标签会"消失"在白底里。
    """
    ymax = ax.get_ylim()[1]
    if val >= ymax * 0.92:
        ax.text(x, val + ymax * 0.012, f'{val:.0f}%', ha='center', va='bottom',
                fontsize=fontsize, color='#2B2B2B',
                rotation=90, zorder=7)
    else:
        ax.text(x, val - ymax * 0.085, f'{val:.0f}%', ha='center', va='bottom',
                fontsize=fontsize, color=readable_text_color(color),
                rotation=90, zorder=7)


def savefig_retry(fig, out_path, attempts=5):
    """
    保存图片, 失败时重试。

    Windows 上偶发 OSError: [Errno 22] Invalid argument — 通常是目标 PNG
    仍被上一个进程 (或图片预览/编辑器) 占用, 稍等即可写入。
    """
    import time
    for i in range(attempts):
        try:
            fig.savefig(out_path)
            return
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f'[WARN] 保存 {os.path.basename(out_path)} 失败 '
                  f'({e}), {i + 1}/{attempts - 1} 次重试...')
            time.sleep(2.0)


def load_data(data_dir):
    """读取所有需要的 CSV"""
    files = {
        'summary': 'summary_by_experiment.csv',
        'rounds': 'rounds_merged.csv',
        'detmargin': 'detection_margins.csv',
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f'[WARN] 缺少文件: {path}, 请先运行 python -m utils.aggregate_results')
            continue
        data[key] = pd.read_csv(path)
    return data


def experiment_order(summary):
    """按 dataset -> model -> K 的固定顺序排列实验"""
    s = summary.copy()
    s['_ds'] = s['dataset'].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
    s['_md'] = s['model'].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    s['_k'] = s['client_num'].astype(int).map(
        {k: i for i, k in enumerate(K_ORDER)}).fillna(99)
    s = s.sort_values(['_ds', '_md', '_k']).drop(
        columns=['_ds', '_md', '_k']).reset_index(drop=True)
    return s


# --------------------------------------------------------------------------- #
# Figure 1 / 2: 单模式的准确率柱状图
# --------------------------------------------------------------------------- #

def fig_mode_accuracy(summary, mode, out_path):
    """
    单种泄漏模式下的准确率柱状图 (18 个配置)。

    mode: PRE_MODE 或 POST_MODE 元组
    """
    plt = setup_matplotlib()
    _, cn, en, color, acc_col, task = mode

    s = experiment_order(summary)

    n = len(s)
    group_w = 1.0 / n
    bar_w = group_w * 0.62

    fig, ax = plt.subplots(figsize=(16, 5.4), layout='constrained')
    ax.set_ylim(0, 108)

    x_centers = []
    for i, (_, row) in enumerate(s.iterrows()):
        c = i * group_w + group_w / 2
        x_centers.append(c)

        val = row.get(acc_col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.bar(c, val, bar_w, color=color, edgecolor='white', linewidth=0.6,
               zorder=3)
        bar_value_label(ax, c, val, color, fontsize=7.0)

    # dataset 分组背景与分隔线
    bounds = []
    for ds in DATASET_ORDER:
        idxs = [i for i, (_, r) in enumerate(s.iterrows()) if r['dataset'] == ds]
        if idxs:
            bounds.append((ds, idxs[0] * group_w, (idxs[-1] + 1) * group_w))
    for idx, (ds, lo, hi) in enumerate(bounds):
        if idx % 2 == 1:
            ax.axvspan(lo, hi, color='#F4F4F4', zorder=0)
        ax.axvline(hi, color='#DDDDDD', linewidth=1.0, zorder=1)
        ax.text((lo + hi) / 2, 103.5, DATASET_LABEL.get(ds, ds),
                ha='center', va='bottom', fontsize=13.5, fontweight='bold')

    ax.set_ylim(0, 108)
    ax.set_ylabel(f'{task}\nAccuracy (%)', fontweight='bold')
    ax.set_xlabel('Configuration (model × dataset × K)', fontsize=10)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([exp_label(r['model'], r['dataset'], int(r['client_num']))
                        for _, r in s.iterrows()], fontsize=7.4)
    ax.set_xlim(0, n * group_w)

    savefig_retry(fig, out_path)
    plt.close(fig)
    print(f'[OK] {os.path.basename(out_path)}')


# --------------------------------------------------------------------------- #
# Figure 1: 聚焦配置的准确率 (三档噪声分组)
# --------------------------------------------------------------------------- #

def fig_focus_accuracy(summary, out_path):
    """
    聚焦配置 (AlexNet / ChestMNIST, K = 5 / 10 / 20) 的追踪准确率。

    每组 = 一个 K, 组内三根柱子 = 三档高斯噪声。
    仅适用于水印区域扰动实验 —— 本地微调攻击没有噪声维度。
    """
    plt = setup_matplotlib()
    _, cn, en, color, _, task = PRE_MODE

    s = summary[(summary['model'] == FOCUS_MODEL)
                & (summary['dataset'] == FOCUS_DATASET)]
    if s.empty:
        print('[WARN] 未找到 AlexNet/ChestMNIST 的准确率数据, 跳过 Figure 1')
        return

    ks = [k for k in FOCUS_K_LIST if k in set(s['client_num'].astype(int))]
    if not ks:
        print('[WARN] 未找到 K=5/10/20 的准确率数据, 跳过 Figure 1')
        return

    n_grp, n_bar = len(ks), len(NOISE_LEVELS)
    group_w = 1.0 / n_grp
    bar_w = group_w / n_bar * 0.72

    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    ax.set_ylim(0, 108)

    for gi, k in enumerate(ks):
        row = s[s['client_num'] == k].iloc[0]
        for bi, (lvl, lname, lscale, lcolor) in enumerate(NOISE_LEVELS):
            col = f'noise{lvl}_accuracy'
            if col not in summary.columns:
                print(f'[WARN] 缺少 {col}, 跳过 Figure 1')
                return
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            c = gi * group_w + (bi + 0.5) * (group_w / n_bar)
            ax.bar(c, val, bar_w, color=lcolor, edgecolor='white',
                   linewidth=0.7, zorder=3)
            bar_value_label(ax, c, val, lcolor, fontsize=8)

    # x 轴: 每组一个 K
    ax.set_xticks([gi * group_w + group_w / 2 for gi in range(n_grp)])
    ax.set_xticklabels([f'K = {k}' for k in ks], fontsize=12)
    ax.set_xlim(0, n_grp * group_w)
    ax.set_ylim(0, 108)
    ax.set_ylabel(f'{task}\nAccuracy (%)', fontweight='bold')
    ax.set_xlabel('Number of clients (K)', fontsize=11)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6, zorder=0)

    # 图例: 噪声档位
    handles = [
        Patch(facecolor=lcolor, label=f'{lname.capitalize()} (σ = {lscale})')
        for lvl, lname, lscale, lcolor in NOISE_LEVELS
    ]
    ax.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
              frameon=False, bbox_to_anchor=(0.5, -0.28))

    # 不用大标题; 配置信息以小字标注在图下方
    fig.text(0.0, -0.055,
             f'{en} — {MODEL_LABEL.get(FOCUS_MODEL, FOCUS_MODEL)} · '
             f'{DATASET_LABEL.get(FOCUS_DATASET, FOCUS_DATASET)}',
             ha='left', va='top', fontsize=8.5, color='#666666')

    savefig_retry(fig, out_path)
    plt.close(fig)
    print(f'[OK] {os.path.basename(out_path)}')


# --------------------------------------------------------------------------- #
# Figure 3 / 4: 单模式的 top-1 / top-2 相似度与 gap
# --------------------------------------------------------------------------- #

def fig_mode_similarity(data, mode, out_path):
    """
    单种泄漏模式下的逐轮相似度曲线: 3 x 6 网格 (行 = dataset, 列 = model x K)。

    每个子图: 泄漏者 top-1 相似度 vs 次高相似度, 阴影为判别 gap。
    """
    plt = setup_matplotlib()
    _, cn, en, color, _, _ = mode

    n_rows, n_cols = len(DATASET_ORDER), len(MODEL_ORDER) * len(K_ORDER)
    fig = plt.figure(figsize=(3.6 * n_cols, 3.1 * n_rows + 0.5), layout='constrained')
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[1] * n_rows + [0.11],
                          hspace=0.10, wspace=0.12)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for ri in range(n_rows):
        for ci in range(n_cols):
            kw = {}
            if ri > 0 or ci > 0:
                kw = dict(sharex=axes[0][0], sharey=axes[0][0])
            axes[ri, ci] = fig.add_subplot(gs[ri, ci], **kw)

    gap_all = []
    for ri, ds in enumerate(DATASET_ORDER):
        for ci, (md, k) in enumerate([(m, kk) for m in MODEL_ORDER for kk in K_ORDER]):
            ax = axes[ri, ci]

            if mode[0] == 'pre':
                sub = data['rounds']
                sub = sub[(sub['model'] == md) & (sub['dataset'] == ds)
                          & (sub['client_num'] == k)].sort_values('round')
                if sub.empty:
                    ax.set_visible(False)
                    continue
                x = sub['round'].values
                top1 = sub['trace_similarity_top1'].values
                top2 = sub['trace_similarity_second'].values
            else:
                sub = data['detmargin']
                sub = sub[(sub['model'] == md) & (sub['dataset'] == ds)
                          & (sub['client_num'] == k)].sort_values('round')
                if sub.empty:
                    ax.set_visible(False)
                    continue
                x = sub['round'].values
                top1 = sub['leaker_similarity'].values
                top2 = sub['max_other_similarity'].values

            ax.plot(x, top1, linewidth=1.7, color=color, label='Leaker (top-1)')
            ax.plot(x, top2, linewidth=1.2, color=SECOND_COLOR,
                    label='2nd-highest')
            ax.fill_between(x, top2, top1, color=color, alpha=0.16)
            ax.axhline(1.0, color='#CCCCCC', linestyle=':', linewidth=1.0)

            gap_all.extend((top1 - top2).tolist())

            # 列头只在第一行标注
            if ri == 0:
                ax.set_title(f'{MODEL_LABEL.get(md, md)} · K={k}', fontsize=11)
            # 行首标注 dataset
            if ci == 0:
                ax.set_ylabel(f'{DATASET_LABEL.get(ds, ds)}\nCosine similarity',
                              fontweight='bold', fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel('Federated round', fontweight='bold', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_ylim(-0.05, 1.06)

    # 图例放在独占的底部网格行内
    handles = [
        Line2D([0], [0], color=color, linewidth=1.8,
               label=f'{en}: leaker (top-1) similarity'),
        Line2D([0], [0], color=SECOND_COLOR, linewidth=1.4,
               label='2nd-highest similarity'),
        Patch(facecolor=color, alpha=0.16, label='Discriminative gap'),
    ]
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.set_axis_off()
    legend_ax.legend(handles=handles, loc='center', ncol=3, fontsize=11)

    # gap 统计: 作为图例行右侧的说明文字 (不用标题)
    if gap_all:
        g = np.asarray(gap_all, dtype=float)
        legend_ax.text(0.995, 0.5,
                       f'Gap: mean {g.mean():.4f}  |  min {g.min():.4f}  |  '
                       f'max {g.max():.4f}',
                       transform=legend_ax.transAxes, ha='right', va='center',
                       fontsize=11, fontweight='bold', color=color)

    savefig_retry(fig, out_path)
    plt.close(fig)
    print(f'[OK] {os.path.basename(out_path)}')


# --------------------------------------------------------------------------- #
# Figure 5: 三档高斯噪声强度 (仅水印区域扰动实验)
# --------------------------------------------------------------------------- #

# 聚焦配置: AlexNet + ChestMNIST, K = 5 / 10 / 20
FOCUS_MODEL = 'alexnet'
FOCUS_DATASET = 'chestmnist'
FOCUS_K_LIST = [5, 10, 20]


def fig_noise_levels(data, out_path):
    """
    三档高斯噪声强度 (weak/medium/strong) 分面展示。

    仅适用于水印区域扰动实验 —— 本地微调攻击没有噪声维度。

    聚焦配置: AlexNet / ChestMNIST, K = 5 / 10 / 20
    布局: 3 行 (K) x 3 列 (噪声档位), 每格画泄漏者 top-1 相似度 vs 次高相似度,
          阴影为判别 gap。
    """
    plt = setup_matplotlib()
    _, cn, en, color, _, _ = PRE_MODE

    rounds = data.get('rounds')
    if rounds is None:
        print('[WARN] 缺少 rounds_merged.csv, 跳过 Figure 5')
        return

    base = rounds[(rounds['model'] == FOCUS_MODEL)
                  & (rounds['dataset'] == FOCUS_DATASET)]
    if base.empty:
        print('[WARN] 未找到 AlexNet/ChestMNIST 的追踪数据, 跳过 Figure 5')
        return

    n_row, n_col = len(FOCUS_K_LIST), len(NOISE_LEVELS)
    fig = plt.figure(figsize=(4.6 * n_col, 3.1 * n_row + 0.9), layout='constrained')
    gs = fig.add_gridspec(n_row, n_col, hspace=0.30, wspace=0.18)

    for ri, k in enumerate(FOCUS_K_LIST):
        sub = base[base['client_num'] == k].sort_values('round')
        if sub.empty:
            continue
        x = sub['round'].values

        for ci, (lvl, lname, lscale, lcolor) in enumerate(NOISE_LEVELS):
            c_top1 = f'noise{lvl}_similarity_top1'
            c_top2 = f'noise{lvl}_similarity_second'
            c_gap = f'noise{lvl}_margin'
            if c_top1 not in sub.columns:
                print(f'[WARN] 缺少 {c_top1}, 跳过 Figure 5')
                return

            s1 = sub[c_top1].values
            s2 = sub[c_top2].values if c_top2 in sub.columns else None

            ax = fig.add_subplot(gs[ri, ci])
            valid = ~np.isnan(s1)
            ax.plot(x[valid], s1[valid], linewidth=1.6, color=lcolor,
                    label='Leaker (top-1)')
            if s2 is not None:
                v2 = ~np.isnan(s2)
                ax.plot(x[v2], s2[v2], linewidth=1.0, color=SECOND_COLOR,
                        label='2nd-highest')
                m = valid & v2
                if m.any():
                    ax.fill_between(x[m], s2[m], s1[m], color=lcolor, alpha=0.16)
            ax.axhline(1.0, color='#CCCCCC', linestyle=':', linewidth=1.0)

            # 列头 (仅第一行) + 行标 (仅第一列)
            if ri == 0:
                ax.set_title(f'{lname.capitalize()} noise  (σ = {lscale})',
                             fontsize=12.5, fontweight='bold', color=lcolor)
            if ci == 0:
                ax.set_ylabel(f'K = {k}\nCosine similarity',
                              fontweight='bold', fontsize=10)
            if ri == n_row - 1:
                ax.set_xlabel('Federated round', fontweight='bold', fontsize=10)
            ax.set_ylim(-0.05, 1.06)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, linewidth=0.6)

            # 角标: 该 (K, 噪声档位) 的 gap 统计
            gap = sub[c_gap].values if c_gap in sub.columns else None
            if gap is not None and not np.all(np.isnan(gap)):
                ax.text(0.985, 0.06,
                        f'gap: mean {np.nanmean(gap):.4f}\nmin {np.nanmin(gap):.4f}',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=8.5, fontweight='bold', color=lcolor)

    # 总图例 + 配置说明
    # gap 阴影在每个子图里用的是该噪声档位自己的颜色, 因此图例也要逐档列出,
    # 否则读者只能看到一种颜色, 不知道三档的 gap 用的是三套配色。
    handles = [
        Line2D([0], [0], color=SECOND_COLOR, linewidth=1.4,
               label='2nd-highest similarity'),
    ]
    for lvl, lname, lscale, lcolor in NOISE_LEVELS:
        handles.append(
            Patch(facecolor=lcolor, alpha=0.16,
                  label=f'Discriminative gap ({lname}, σ = {lscale})'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.8, handlelength=1.6)
    # 不用大标题; 配置信息以小字标注在图例下方
    fig.text(0.5, -0.075,
             f'{en} — {MODEL_LABEL.get(FOCUS_MODEL, FOCUS_MODEL)} · '
             f'{DATASET_LABEL.get(FOCUS_DATASET, FOCUS_DATASET)}',
             ha='center', va='top', fontsize=8.5, color='#666666')

    savefig_retry(fig, out_path)
    plt.close(fig)
    print(f'[OK] {os.path.basename(out_path)}')

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='生成论文核心图表 (两种泄漏模式分开绘制, 共 4 张)')
    parser.add_argument('--data-dir', default=os.path.join('save', 'aggregate'),
                        help='aggregate_results.py 的输出目录')
    parser.add_argument('--out-dir', default=os.path.join('save', 'figures'),
                        help='图片输出目录')
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(root, args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(data_dir)
    if 'summary' not in data:
        print('[ERROR] 未找到 summary_by_experiment.csv, '
              '请先运行 python -m utils.aggregate_results')
        return 1

    # Figure 1: 聚焦配置的准确率 (三档噪声分组)
    fig_focus_accuracy(data['summary'],
                       os.path.join(out_dir, 'fig1_perturbation_accuracy.png'))

    # Figure 2: 本地微调准确率 (全配置)
    fig_mode_accuracy(data['summary'], POST_MODE,
                      os.path.join(out_dir, 'fig2_finetune_accuracy.png'))

    # Figure 3 / 4: 相似度与 gap (每种模式一张)
    if 'rounds' in data and 'detmargin' in data:
        fig_mode_similarity(data, PRE_MODE,
                            os.path.join(out_dir, 'fig3_perturbation_similarity.png'))
        fig_mode_similarity(data, POST_MODE,
                            os.path.join(out_dir, 'fig4_finetune_similarity.png'))
    else:
        print('[WARN] 缺少 rounds_merged.csv / detection_margins.csv, 跳过 Figure 3/4')

    # Figure 5: 三档高斯噪声强度 (仅水印区域扰动实验; 本地微调无噪声维度)
    fig_noise_levels(data, os.path.join(out_dir, 'fig5_noise_levels.png'))

    print(f'\n[OK] 图表已输出到: {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
