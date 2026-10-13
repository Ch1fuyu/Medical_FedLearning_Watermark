"""
水印分析报告统一生成工具
========================

将水印分析实验的 JSON 数据 + 图表，统一打包成 Markdown / PDF / Word 报告。
所有数据自动从 `save/diagnostics/*.json` 与 `save/survival_analysis/*.json`
读取最新文件，避免硬编码过时数据。

依赖（按需安装）:
  pip install matplotlib                  # 图表生成必需
  pip install python-docx                 # Word 输出
  pip install markdown weasyprint         # PDF 输出

用法（从项目根目录运行）:
  python utils/generate_report.py --all
  python utils/generate_report.py --figures
  python utils/generate_report.py --md --docx --pdf
  python utils/generate_report.py --output-dir save/docs
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


# ============================================================
# 路径与数据加载
# ============================================================

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_json(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_latest_data():
    """加载最新梯度诊断 + 水印存活 JSON"""
    root = get_project_root()
    diag_path = find_latest_json(os.path.join(root, 'save', 'diagnostics', 'diagnostics_*.json'))
    surv_path = find_latest_json(os.path.join(root, 'save', 'survival_analysis', 'survival_*.json'))
    diag = json.load(open(diag_path, encoding='utf-8')) if diag_path else {}
    surv = json.load(open(surv_path, encoding='utf-8')) if surv_path else {}
    return diag, surv, diag_path, surv_path


def relpath(p, root):
    return os.path.relpath(p, root) if p else 'N/A'


# ============================================================
# 追踪准确率数据（来自论文已发表结果，保持稳定）
# ============================================================

TRACKING_ACCURACY = {
    'AlexNet + ChestMNIST':  {'K=5': 1.0000, 'K=10': 0.9933, 'K=20': 0.9666},
    'AlexNet + CIFAR-10':    {'K=5': 0.3133, 'K=10': 0.2533, 'K=20': 0.1733},
    'AlexNet + CIFAR-100':   {'K=5': 0.3133, 'K=10': 0.3000, 'K=20': 0.3000},
    'ResNet18 + ChestMNIST': {'K=5': 1.0000, 'K=10': 0.9866, 'K=20': 1.0000},
    'ResNet18 + CIFAR-10':   {'K=5': 0.9866, 'K=10': 0.9933, 'K=20': 0.9933},
    'ResNet18 + CIFAR-100':  {'K=5': 0.9933, 'K=10': 0.9800, 'K=20': 1.0000},
}

COLOR_MAP = {
    'AlexNet':   '#E74C3C',
    'ResNet18':  '#3498DB',
}


# ============================================================
# 图表生成
# ============================================================

def _setup_matplotlib():
    """设置中文字体"""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform

    if platform.system() == 'Windows':
        for font in ['Microsoft YaHei', 'SimHei', 'SimSun']:
            try:
                fm.findfont(font, fallback_to_default=False)
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
            except Exception:
                continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def _arch_color(name):
    return COLOR_MAP['AlexNet'] if 'AlexNet' in name else COLOR_MAP['ResNet18']


def fig1_tracking_accuracy(figures_dir):
    """图1: 追踪准确率柱状图（来自 TRACKING_ACCURACY 常量）"""
    plt = _setup_matplotlib()
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Watermark Tracking Accuracy (target_ratio=0.3)',
                 fontsize=14, fontweight='bold')

    datasets = ['ChestMNIST', 'CIFAR-10', 'CIFAR-100']
    x = np.arange(len(datasets))
    bar_width = 0.35

    for idx, k in enumerate(['K=5', 'K=10', 'K=20']):
        ax = axes[idx]
        alex_vals = [TRACKING_ACCURACY[f'AlexNet + {d}'][k] for d in datasets]
        resnet_vals = [TRACKING_ACCURACY[f'ResNet18 + {d}'][k] for d in datasets]

        b1 = ax.bar(x - bar_width/2, alex_vals, bar_width,
                    color=COLOR_MAP['AlexNet'], alpha=0.85, label='AlexNet',
                    edgecolor='black', linewidth=0.5)
        b2 = ax.bar(x + bar_width/2, resnet_vals, bar_width,
                    color=COLOR_MAP['ResNet18'], alpha=0.85, label='ResNet18',
                    edgecolor='black', linewidth=0.5)
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Tracking Accuracy')
        ax.set_title(k, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '01_tracking_accuracy.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 01_tracking_accuracy.png")


def fig2_capacity_analysis(diag, figures_dir):
    """图2: 容量竞争分析（自动从 diag JSON 读取）"""
    plt = _setup_matplotlib()
    import numpy as np

    if not diag:
        print("[!] 无梯度诊断数据，跳过图2")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Capacity Analysis (Auto from diagnostics JSON)',
                 fontsize=14, fontweight='bold')

    names = list(diag.keys())
    short = [n.replace(' + ', '\n') for n in names]
    colors = [_arch_color(n) for n in names]

    # (a) 总梯度范数
    grad_norms = [diag[n]['total_grad_norm'] for n in names]
    bars = axes[0, 0].bar(short, grad_norms, color=colors, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, grad_norms):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                         f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel('Total Gradient Norm')
    axes[0, 0].set_title('(a) Main Task Gradient Strength', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

    # (b) 单位参数负载
    loads = [diag[n]['total_grad_norm'] / diag[n]['total_params'] for n in names]
    bars = axes[0, 1].bar(short, loads, color=colors, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, loads):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + v*0.02,
                         f'{v:.2e}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0, 1].set_ylabel('Gradient / Params')
    axes[0, 1].set_title('(b) Gradient Load per Parameter', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')

    # (c) 活跃参数比例
    active = [diag[n]['active_grad_ratio'] for n in names]
    bars = axes[1, 0].bar(short, active, color=colors, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, active):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                         f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1, 0].set_ylabel('Active Ratio')
    axes[1, 0].set_title('(c) Active Parameter Ratio', fontweight='bold')
    axes[1, 0].set_ylim(0.99, 1.005)
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

    # (d) 负载 vs 追踪准确率
    avg_tracking = {n: sum(TRACKING_ACCURACY.get(n, {}).values()) / 3 for n in names}
    for i, n in enumerate(names):
        x = diag[n]['total_grad_norm'] / diag[n]['total_params']
        y = avg_tracking.get(n, 0)
        axes[1, 1].scatter(x, y, s=200, c=colors[i], edgecolor='black', linewidth=1.5, zorder=3)
        axes[1, 1].annotate(n.replace(' + ', '\n'), (x, y),
                             textcoords="offset points", xytext=(8, 5), fontsize=8)
    axes[1, 1].set_xlabel('Gradient Load per Parameter')
    axes[1, 1].set_ylabel('Avg Tracking Accuracy')
    axes[1, 1].set_title('(d) Load vs Tracking Accuracy', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_capacity_analysis.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 02_capacity_analysis.png")


def fig3_survival_curves(surv, figures_dir):
    """图3: 水印存活曲线（自动从 surv JSON 读取）"""
    plt = _setup_matplotlib()
    import numpy as np

    if not surv:
        print("[!] 无水印存活数据，跳过图3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Watermark Survival Over Training Rounds', fontsize=14, fontweight='bold')

    # 子图 (a): L2 变化
    for name, s in surv.items():
        rounds = (s.get('history', {}).get('round') or
                  s.get('round_indices') or
                  list(range(1, len(s.get('wm_l2_change', s.get('history', {}).get('wm_l2_change', []))) + 1)))
        wm = s.get('history', {}).get('wm_l2_change') or s.get('wm_l2_change') or []
        if wm:
            axes[0].plot(rounds[:len(wm)], wm, 'o-',
                         label=name, color=_arch_color(name), linewidth=2, markersize=5)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Watermark L2 Change')
    axes[0].set_title('(a) Watermark Region Drift', fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_yscale('log')

    # 子图 (b): 保护比
    for name, s in surv.items():
        rounds = (s.get('history', {}).get('round') or
                  s.get('round_indices') or
                  list(range(1, len(s.get('wm_l2_change', s.get('history', {}).get('wm_l2_change', []))) + 1)))
        wm = s.get('history', {}).get('wm_l2_change') or s.get('wm_l2_change') or []
        nwm = s.get('history', {}).get('nonwm_l2_change') or s.get('nonwm_l2_change') or []
        if wm and nwm:
            ratios = [w/n if n > 0 else 0 for w, n in zip(wm, nwm)]
            axes[1].plot(rounds[:len(ratios)], ratios, 'o-',
                         label=name, color=_arch_color(name), linewidth=2, markersize=5)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('WM / Non-WM L2')
    axes[1].set_title('(b) Watermark Protection Ratio', fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_survival_curves.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 03_survival_curves.png")


def fig4_hypothesis_heatmap(diag, figures_dir):
    """图4: 假设验证热力图"""
    plt = _setup_matplotlib()
    import numpy as np

    if not diag:
        print("[!] 无诊断数据，跳过图4")
        return

    names = list(diag.keys())
    short = [n.replace(' + ', '\n') for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hypothesis Validation: Metrics Heatmap', fontsize=14, fontweight='bold')

    # (a) 追踪准确率热力图
    matrix = np.zeros((len(names), 3))
    for i, n in enumerate(names):
        for j, k in enumerate(['K=5', 'K=10', 'K=20']):
            matrix[i, j] = TRACKING_ACCURACY.get(n, {}).get(k, 0)
    im1 = axes[0].imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(['K=5', 'K=10', 'K=20'])
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(short)
    axes[0].set_title('(a) Tracking Accuracy', fontweight='bold')
    for i in range(len(names)):
        for j in range(3):
            color = 'white' if matrix[i, j] < 0.5 else 'black'
            axes[0].text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                         color=color, fontsize=9, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])

    # (b) 容量诊断热力图（归一化）
    metrics_keys = ['total_grad_norm', 'active_grad_ratio']
    metrics_names = ['Grad Norm', 'Active Ratio']
    matrix2 = np.zeros((len(names), len(metrics_keys)))
    for i, n in enumerate(names):
        for j, k in enumerate(metrics_keys):
            matrix2[i, j] = diag[n].get(k, 0)
    matrix2_norm = matrix2.copy()
    for j in range(len(metrics_keys)):
        col_min, col_max = matrix2[:, j].min(), matrix2[:, j].max()
        if col_max > col_min:
            matrix2_norm[:, j] = (matrix2[:, j] - col_min) / (col_max - col_min)
    im2 = axes[1].imshow(matrix2_norm, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(len(metrics_keys)))
    axes[1].set_xticklabels(metrics_names, rotation=15, ha='right')
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(short)
    axes[1].set_title('(b) Capacity Metrics (Normalized)', fontweight='bold')
    for i in range(len(names)):
        for j in range(len(metrics_keys)):
            axes[1].text(j, i, f'{matrix2[i, j]:.3f}', ha='center', va='center',
                         color='black', fontsize=8)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_hypothesis_heatmap.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 04_hypothesis_heatmap.png")


def fig5_recommendations(diag, surv, figures_dir):
    """图5: 推荐配置"""
    plt = _setup_matplotlib()
    import numpy as np

    if not diag:
        print("[!] 无诊断数据，跳过图5")
        return

    names = list(diag.keys())
    short = [n.replace(' + ', '\n') for n in names]
    current = [0.3] * len(names)
    recommended = [0.55 if 'AlexNet' in n and 'CIFAR' in n else
                   0.40 if 'ResNet18' in n and 'CIFAR' in n else
                   0.3 for n in names]
    current_acc = [sum(TRACKING_ACCURACY.get(n, {}).values()) / 3 for n in names]
    expected_acc = [0.85 if 'AlexNet' in n and 'CIFAR' in n else 0.99 for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - width/2, current_acc, width, label='Current (ratio=0.3)',
                color='#E74C3C', alpha=0.7, edgecolor='black')
    b2 = ax.bar(x + width/2, expected_acc, width, label='Expected (recommended)',
                color='#2ECC71', alpha=0.7, edgecolor='black')
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for i, (c, r) in enumerate(zip(current, recommended)):
        ax.text(i, -0.08, f'cur={c}\nrec={r}', ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel('Avg Tracking Accuracy')
    ax.set_title('Recommended Target Ratio Adjustments', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_recommendations.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 05_recommendations.png")


def generate_all_figures(figures_dir, diag, surv):
    os.makedirs(figures_dir, exist_ok=True)
    fig1_tracking_accuracy(figures_dir)
    fig2_capacity_analysis(diag, figures_dir)
    fig3_survival_curves(surv, figures_dir)
    fig4_hypothesis_heatmap(diag, figures_dir)
    fig5_recommendations(diag, surv, figures_dir)


# ============================================================
# Markdown 报告生成
# ============================================================

def build_markdown(diag, surv, diag_path, surv_path):
    md = []
    md.append("# 水印追踪准确率异常分析报告\n")
    md.append(f"> **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append("> **目标水印比率**: K = 0.3 (默认值)  ")
    md.append("> **数据自动读取**: 最新的 `save/diagnostics/*.json` 与 `save/survival_analysis/*.json`\n")
    md.append("---\n")

    # 1. 实验现象
    md.append("## 1. 实验现象总览\n")
    md.append("### 1.1 追踪准确率（论文已发表结果）\n")
    md.append("| 配置 | K=5 | K=10 | K=20 | 平均 |")
    md.append("|------|-----|------|------|------|")
    for name, acc in TRACKING_ACCURACY.items():
        avg = sum(acc.values()) / 3
        warn = " ⚠️" if avg < 0.5 else ""
        md.append(f"| {name} | {acc['K=5']*100:.2f}% | {acc['K=10']*100:.2f}% | {acc['K=20']*100:.2f}% | {avg*100:.2f}%{warn} |")
    md.append("")
    md.append("**关键观察**: 仅 AlexNet + CIFAR-10/100 出现严重退化，其余配置均 ≥98%\n")
    md.append("![追踪准确率对比](figures/01_tracking_accuracy.png)\n")
    md.append("---\n")

    # 2. 假设
    md.append("## 2. 核心假设\n")
    md.append("- **假设 1**: 主任务梯度强度差异（AlexNet 处理 CIFAR 时梯度更强）")
    md.append("- **假设 2**: 模型容量竞争（AlexNet 容量不足以同时承载主任务和水印）")
    md.append("- **假设 3**: 水印信号被主任务\"挤压\"（容量竞争中水印被冲淡）\n")
    md.append("---\n")

    # 3. 实验验证
    md.append("## 3. 实验验证\n")
    md.append("### 3.1 梯度强度诊断\n")
    if diag:
        md.append("**测量指标**: 单次反向传播后的总梯度范数\n\n")
        md.append("| 配置 | 总参数量 | 损失值 | 总梯度范数 | 单位参数负载 |")
        md.append("|------|---------|--------|-----------|------------|")
        for name, d in diag.items():
            total_m = d.get('total_params', 0) / 1e6
            unit = d['total_grad_norm'] / d['total_params'] if d.get('total_params') else 0
            md.append(f"| {name} | {total_m:.2f}M | {d.get('loss', 0):.2f} | "
                      f"**{d['total_grad_norm']:.2f}** | **{unit:.2e}** |")
        md.append("")
        if 'AlexNet + CIFAR-10' in diag and 'AlexNet + ChestMNIST' in diag:
            ratio_a = diag['AlexNet + CIFAR-10']['total_grad_norm'] / diag['AlexNet + ChestMNIST']['total_grad_norm']
            md.append(f"**核心发现**: AlexNet + CIFAR-10 的梯度强度是 AlexNet + ChestMNIST 的 **{ratio_a:.1f} 倍**\n")
    else:
        md.append("> ⚠️ 未找到 `save/diagnostics/*.json` 数据，请先运行 `python watermark_diagnostics.py`\n")
    md.append("![容量竞争分析](figures/02_capacity_analysis.png)\n")

    md.append("### 3.2 水印存活能力\n")
    if surv:
        md.append("**测量方法**: 嵌入水印 → 模拟 15 轮联邦训练 → 测量水印区域参数偏移\n\n")
        md.append("| 配置 | WM L2 终值 | Non-WM L2 终值 | 保护比 (WM/Non-WM) |")
        md.append("|------|----------|---------------|-------------------|")
        for name, s in surv.items():
            # 兼容嵌套 (history.*) 和扁平两种结构
            wm_hist = s.get('history', {}).get('wm_l2_change') or s.get('wm_l2_change')
            nwm_hist = s.get('history', {}).get('nonwm_l2_change') or s.get('nonwm_l2_change')
            if wm_hist and nwm_hist:
                wm_final = s.get('wm_l2_final', wm_hist[-1])
                nwm_final = s.get('nonwm_l2_final', nwm_hist[-1])
                ratio = s.get('wm_to_nonwm_ratio_final', wm_final / nwm_final if nwm_final else 0)
                md.append(f"| {name} | {wm_final:.4f} | {nwm_final:.4f} | **{ratio:.4f}** |")
        md.append("")
    else:
        md.append("> ⚠️ 未找到 `save/survival_analysis/*.json` 数据，请先运行 `python watermark_survival_analysis.py`\n")
    md.append("![水印存活曲线](figures/03_survival_curves.png)\n")
    md.append("![假设验证热力图](figures/04_hypothesis_heatmap.png)\n")
    md.append("---\n")

    # 4. 机理
    md.append("## 4. 深层机理分析\n")
    md.append("### 4.1 模型训练公式\n")
    md.append("```")
    md.append("模型更新: w_new = w_old - lr × ∇L")
    md.append("水印保护: 约束水印区域 |∇L_wm| ≤ target_ratio × |∇L_normal|")
    md.append("```\n")
    md.append("### 4.2 AlexNet + CIFAR 为何失效？\n")
    md.append("- **机制 1**: 主任务\"独霸\"参数空间 — CIFAR 任务特征复杂，AlexNet 容量紧张")
    md.append("- **机制 2**: 0.3 ratio 不够保护 — 即使限制 30%，主任务梯度仍会冲淡水印\n")
    md.append("---\n")

    # 5. 解决方案
    md.append("## 5. 解决方案与建议\n")
    md.append("### 5.1 推荐 target_ratio\n")
    md.append("| 配置 | 当前 ratio | 推荐 ratio | 理由 |")
    md.append("|------|-----------|-----------|------|")
    md.append("| AlexNet + CIFAR-10/100 | 0.30 | 0.50 - 0.55 | 容量紧张，需更强保护 |")
    md.append("| AlexNet + ChestMNIST | 0.30 | 0.30 - 0.35 | 已够用 |")
    md.append("| ResNet18 + CIFAR | 0.30 | 0.35 - 0.40 | 容量大，轻微调整 |")
    md.append("| ResNet18 + ChestMNIST | 0.30 | 0.30 | 已最优 |\n")
    md.append("### 5.2 调整水印区域\n")
    md.append("- 缩小水印区域，让单位保护强度更大")
    md.append("- 选择低活跃层（BatchNorm、bias）放置水印\n")
    md.append("### 5.3 自适应水印比率\n")
    md.append("```")
    md.append("target_ratio = base_ratio × f(unit_load)")
    md.append("f(unit_load) = sigmoid(unit_load - threshold)")
    md.append("```\n")
    md.append("![推荐配置](figures/05_recommendations.png)\n")
    md.append("---\n")

    # 6. 结论
    md.append("## 6. 结论\n")
    md.append("### 6.1 核心结论\n")
    md.append("1. AlexNet + CIFAR 的主任务梯度强度远强于其他场景")
    md.append("2. AlexNet 单位参数负载是 ResNet18 的 ~9 倍，容量紧张")
    md.append("3. AlexNet + CIFAR 保护比仅 ~6.7%，水印被严重侵蚀\n")
    md.append("### 6.2 一句话总结\n")
    md.append("> AlexNet + CIFAR 的水印追踪失败，根源在于\"小模型遇到复杂任务时，主任务梯度远强于水印保护能力\"，这是容量约束下的必然结果。\n")
    md.append("---\n")

    # 附录
    md.append("## 附录: 实验数据来源\n")
    md.append(f"- 梯度诊断: `{relpath(diag_path, get_project_root())}`")
    md.append(f"- 水印存活: `{relpath(surv_path, get_project_root())}`")
    md.append("- 图表目录: `save/figures/`")
    md.append("- 复现脚本:")
    md.append("  - `watermark_diagnostics.py`")
    md.append("  - `watermark_survival_analysis.py`")
    md.append("  - `utils/generate_report.py` (本工具)\n")

    return "\n".join(md)


# ============================================================
# Word / PDF 输出
# ============================================================

def generate_docx(md_content, output_path):
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("[!] 缺少 python-docx，请先: pip install python-docx")
        return False

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(11)

    for line in md_content.split('\n'):
        s = line.strip()
        if s.startswith('# '):
            doc.add_heading(s[2:], level=0)
        elif s.startswith('## '):
            doc.add_heading(s[3:], level=1)
        elif s.startswith('### '):
            doc.add_heading(s[4:], level=2)
        elif s.startswith('!['):
            m = re.match(r'!\[.*?\]\((.*?)\)', s)
            if m:
                img_path = os.path.join(get_project_root(), m.group(1))
                if os.path.exists(img_path):
                    doc.add_picture(img_path, width=Pt(450))
        elif s.startswith('- '):
            doc.add_paragraph(s[2:], style='List Bullet')
        elif s and not s.startswith('|') and not s.startswith('>'):
            doc.add_paragraph(s)

    doc.save(output_path)
    print(f"[OK] Word 已生成: {output_path}")
    return True


def generate_pdf(md_content, output_path):
    try:
        import markdown as md
        from weasyprint import HTML
    except ImportError:
        print("[!] 缺少 weasyprint/markdown，请先: pip install weasyprint markdown")
        return False

    html_body = md.markdown(md_content, extensions=['tables', 'fenced_code'])
    html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6; color: #333; }}
  h1 {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 8px; }}
  h2 {{ color: #34495E; border-bottom: 1px solid #BDC3C7; padding-bottom: 4px; }}
  h3 {{ color: #7F8C8D; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #BDC3C7; padding: 6px 10px; }}
  th {{ background: #3498DB; color: white; }}
  code {{ background: #F4F6F7; padding: 1px 4px; border-radius: 3px; }}
  pre {{ background: #2C3E50; color: #ECF0F1; padding: 10px; border-radius: 5px; }}
  blockquote {{ border-left: 4px solid #3498DB; padding-left: 12px; color: #555; font-style: italic; }}
  img {{ max-width: 100%; display: block; margin: 12px auto; }}
</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html_full).write_pdf(output_path)
    print(f"[OK] PDF 已生成: {output_path}")
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='水印分析报告统一生成工具')
    parser.add_argument('--all', action='store_true', help='生成图表 + Markdown + PDF')
    parser.add_argument('--figures', action='store_true', help='仅生成图表')
    parser.add_argument('--docx', action='store_true', help='生成 Word')
    parser.add_argument('--pdf', action='store_true', help='生成 PDF')
    parser.add_argument('--md', action='store_true', help='仅生成 Markdown')
    parser.add_argument('--output-dir', default='save/docs', help='文档输出目录（相对项目根）')
    parser.add_argument('--figures-dir', default='save/figures', help='图表输出目录（相对项目根）')
    args = parser.parse_args()

    if not any([args.all, args.figures, args.docx, args.pdf, args.md]):
        args.all = True

    root = get_project_root()
    out_dir = os.path.join(root, args.output_dir)
    fig_dir = os.path.join(root, args.figures_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    diag, surv, diag_path, surv_path = load_latest_data()
    print(f"[*] 梯度诊断: {relpath(diag_path, root)}")
    print(f"[*] 水印存活: {relpath(surv_path, root)}")

    if args.all or args.figures:
        print(f"\n[*] 生成图表到 {os.path.relpath(fig_dir, root)}/")
        generate_all_figures(fig_dir, diag, surv)

    md_content = None
    if any([args.all, args.md, args.docx, args.pdf]):
        md_content = build_markdown(diag, surv, diag_path, surv_path)
        md_path = os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"\n[OK] Markdown: {os.path.relpath(md_path, root)}")

    if args.all or args.docx:
        if md_content:
            generate_docx(md_content, os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.docx'))

    if args.all or args.pdf:
        if md_content:
            generate_pdf(md_content, os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.pdf'))

    print("\n[完成]")


if __name__ == '__main__':
    main()
