# -*- coding: utf-8 -*-
"""
watermark_grad_target_ratio 敏感性扫描脚本
=============================================

目的
----
排查 ``utils/args.py`` 中 ``--watermark_grad_target_ratio`` 对**水印鲁棒性**的影响：
在多个 ratio 取值下重复联邦学习训练，统计水印相关的核心指标，找出
「追踪/检测准确率」与「主任务精度」之间的权衡曲线。

设计要点
--------
1. **可复现**：每个 ratio 使用相同随机种子（``config.globals.set_seed``），
   数据划分、客户端采样、泄漏客户端选择保持一致，使不同 ratio 之间可比。
2. **隔离输出**：每次实验写入独立的 ``save/ratio_sweep/<tag>/`` 目录，
   不污染 ``save/detection`` / ``save/trace`` 既有结果。
3. **指标来源**：直接复用 ``main.py`` 已有的落盘产物
   （``save/detection/*.json`` 的检测准确率、``save/trace/*.json`` 的追踪准确率、
   ``save/excel/*.xlsx`` 的训练指标），避免重复实现统计逻辑。
4. **梯度观测**：从训练日志中解析 ``wm/non_wm ratio``，确认 capping 实际生效情况
   （修复前该值会因 clamp 下限而"卡住"，修复后应贴合目标值）。

用法
----
    # 扫描默认 ratio 列表（单配置）
    python ratio_sweep_experiment.py --dataset cifar10 --model_name alexnet

    # 指定 ratio 列表
    python ratio_sweep_experiment.py --dataset cifar10 --model_name alexnet \
        --ratios 0.0 0.1 0.2 0.3 0.5

    # 只跑不重训、仅从已有结果汇总
    python ratio_sweep_experiment.py --dataset cifar10 --model_name alexnet --analyze-only
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

# Windows 控制台默认 GBK 编码，打印 emoji / 特殊符号会抛 UnicodeEncodeError。
# 统一把 stdout/stderr 切到 UTF-8（errors='replace' 保证不会因个别字符再次中断）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# 训练日志中 MultiLoss 打印的行: wm/non_wm ratio:X
_WM_RATIO_RE = re.compile(r'wm/non_wm ratio:([0-9.]+)')
# capping 之后的实际比例: wm/non_wm ratio(after cap):X
_WM_RATIO_AFTER_CAP_RE = re.compile(r'wm/non_wm ratio\(after cap\):([0-9.]+)')
# 是否真的执行了 capping
_CAP_APPLIED_RE = re.compile(r'cap_applied:(True|False)')
# 训练日志中检测准确率行
_DETECT_ACC_RE = re.compile(r'检测准确率:\s*([0-9.]+)%')
# 训练日志中追踪准确率行
_TRACE_ACC_RE = re.compile(r'追踪准确率:\s*([0-9.]+)%')


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _parse_wm_ratios(log_path):
    """
    从训练日志中解析梯度比例。

    返回 (capping 前比例列表, capping 后比例列表, capping 触发次数)
    """
    before, after = [], []
    cap_applied_count = 0
    if not os.path.exists(log_path):
        return before, after, cap_applied_count
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _WM_RATIO_AFTER_CAP_RE.search(line)
            if m:
                try:
                    after.append(float(m.group(1)))
                except ValueError:
                    pass
                continue
            m = _WM_RATIO_RE.search(line)
            if m:
                try:
                    before.append(float(m.group(1)))
                except ValueError:
                    pass
            m = _CAP_APPLIED_RE.search(line)
            if m:
                if m.group(1) == 'True':
                    cap_applied_count += 1
    return before, after, cap_applied_count


def _parse_acc_from_log(log_path):
    """从训练日志中解析检测准确率与追踪准确率"""
    detect_acc, trace_acc = None, None
    if not os.path.exists(log_path):
        return detect_acc, trace_acc
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    m = _DETECT_ACC_RE.search(content)
    if m:
        detect_acc = float(m.group(1))
    m = _TRACE_ACC_RE.search(content)
    if m:
        trace_acc = float(m.group(1))
    return detect_acc, trace_acc


def _latest_detection_json(out_dir, model, dataset, new_files=None):
    """
    查找本次实验产生的 detection 结果。

    注意：``main.py`` 中的 ``save/detection`` 与 ``save/trace`` 路径是**硬编码**的
    （相对于进程 cwd），无法通过 ``--save_excel_dir`` 之类的参数重定向。
    因此这里采用"运行前快照 -> 运行后取差集"的方式定位本次实验的新产物。
    """
    if new_files is not None:
        return new_files.get('detection')

    pattern = os.path.join('save', 'detection', 'watermark_deviation_detection_*.json')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _latest_trace_json(out_dir, model, dataset, new_files=None):
    """查找本次实验产生的 trace 结果（同样依赖快照差集）"""
    if new_files is not None:
        return new_files.get('trace')

    pattern = os.path.join('save', 'trace', 'trace_results_*.json')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _snapshot_result_dirs():
    """快照 save/trace 与 save/detection 下的文件集合"""
    snap = {'trace': set(), 'detection': set()}
    for key, sub in (('trace', 'trace'), ('detection', 'detection')):
        d = os.path.join('save', sub)
        if os.path.isdir(d):
            snap[key] = set(glob.glob(os.path.join(d, '*.json')))
    return snap


def _diff_result_dirs(before):
    """对比快照，返回本次运行新增的产物路径"""
    after = _snapshot_result_dirs()
    out = {}
    for key in ('trace', 'detection'):
        new = sorted(after[key] - before[key])
        out[key] = new[-1] if new else None
    return out


def _latest_excel(out_dir):
    pattern = os.path.join(out_dir, 'excel', '*.xlsx')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _read_excel_metrics(path):
    """
    读取 Excel 中的最佳精度指标。

    注意：Excel 里 ``val_acc_label`` / ``best_val_acc_so_far`` 存的是**比例**（0~1），
    这里统一转成百分比（0~100），方便与检测准确率（百分比）直接比较。
    """
    if not path:
        return {}
    try:
        import pandas as pd
        df = pd.read_excel(path)
        if df.empty:
            return {}
        out = {}
        if 'best_val_acc_so_far' in df.columns:
            out['best_val_acc'] = float(df['best_val_acc_so_far'].max()) * 100
        if 'best_val_auc_so_far' in df.columns:
            out['best_val_auc'] = float(df['best_val_auc_so_far'].max()) * 100
        if 'val_acc_label' in df.columns:
            out['final_val_acc'] = float(df['val_acc_label'].iloc[-1]) * 100
        if 'val_auc' in df.columns:
            out['final_val_auc'] = float(df['val_auc'].iloc[-1]) * 100
        # 泄漏追踪逐轮正确率
        for level in (1, 2, 3):
            col = f'leakage_noise{level}_correct'
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals):
                    out[f'noise{level}_trace_acc'] = float(vals.mean() * 100)
        return out
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# 单次实验
# --------------------------------------------------------------------------- #

def run_one_ratio(args, ratio, tag):
    """在指定 ratio 下运行一次联邦学习训练，返回该次实验的指标字典"""
    ratio_tag = f'ratio{ratio:.2f}'.replace('.', 'p')
    out_dir = os.path.join('save', 'ratio_sweep', tag, ratio_tag)
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, 'excel'))
    _ensure_dir(os.path.join(out_dir, 'detection'))
    _ensure_dir(os.path.join(out_dir, 'trace'))

    log_file = os.path.join(out_dir, 'train.log')

    cmd = [
        sys.executable, 'main.py',
        '--dataset', args.dataset,
        '--model_name', args.model_name,
        '--client_num', str(args.client_num),
        '--epochs', str(args.epochs),
        '--local_ep', str(args.local_ep),
        '--lr', str(args.lr),
        '--optim', args.optim,
        '--watermark_grad_target_ratio', str(ratio),
        '--leak_interval', str(args.leak_interval),
        '--save_excel_dir', os.path.join(out_dir, 'excel'),
        '--save_model_dir', os.path.join(out_dir, 'models'),
        '--log_file', log_file,
    ]

    print('=' * 78)
    print(f'[ratio={ratio}] {args.dataset} / {args.model_name} / K={args.client_num}')
    print('=' * 78)
    print('命令:', ' '.join(cmd))

    start = datetime.now()
    env = dict(os.environ)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Windows 控制台默认 GBK，main.py 中大量 print 含 emoji/中文，
    # 直接运行会抛 UnicodeEncodeError，这里强制子进程使用 UTF-8。
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    # 快照 save/trace 与 save/detection，用于定位本次实验新产生的结果文件
    snapshot_before = _snapshot_result_dirs()

    returncode = -1
    with open(log_file, 'w', encoding='utf-8', errors='replace') as lf:
        lf.write(f"# command: {' '.join(cmd)}\n")
        lf.write(f"# started: {start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n# TIMEOUT after {args.timeout}s\n")
            returncode = -9

    duration = (datetime.now() - start).total_seconds()

    # ---- 收集指标 ----
    wm_ratios, wm_ratios_after_cap, cap_applied_count = _parse_wm_ratios(log_file)
    log_detect_acc, log_trace_acc = _parse_acc_from_log(log_file)

    new_files = _diff_result_dirs(snapshot_before)
    det_path = _latest_detection_json(out_dir, args.model_name, args.dataset, new_files)
    trace_path = _latest_trace_json(out_dir, args.model_name, args.dataset, new_files)
    excel_path = _latest_excel(out_dir)

    # 把本次新产生的 JSON 复制一份到隔离目录，便于追溯
    for key, path in (('detection', det_path), ('trace', trace_path)):
        if path and os.path.exists(path):
            try:
                shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
            except Exception:
                pass

    detection_accuracy = None
    noise_acc = {}
    if det_path:
        with open(det_path, 'r', encoding='utf-8') as f:
            det = json.load(f)
        detection_accuracy = det.get('detection_accuracy')
        for r in det.get('rounds', []):
            pass

    trace_accuracy = None
    if trace_path:
        with open(trace_path, 'r', encoding='utf-8') as f:
            tr = json.load(f)
        # 不同版本落盘字段名不一致，逐一兼容
        for key in ('accuracy', 'overall_accuracy', 'trace_accuracy'):
            if tr.get(key) is not None:
                trace_accuracy = float(tr[key])
                break
        nls = tr.get('noise_level_stats') or {}
        for level, stat in nls.items():
            if isinstance(stat, dict):
                acc = stat.get('accuracy') or stat.get('trace_accuracy')
                if acc is not None:
                    noise_acc[f'noise{level}_acc'] = float(acc)

    excel_metrics = _read_excel_metrics(excel_path)

    result = {
        'target_ratio': ratio,
        'tag': tag,
        'returncode': returncode,
        'success': returncode == 0,
        'duration_seconds': round(duration, 1),
        'out_dir': out_dir,
        'log_file': log_file,
        'detection_accuracy': detection_accuracy,
        'trace_accuracy': trace_accuracy,
        'log_detection_accuracy': log_detect_acc,
        'log_trace_accuracy': log_trace_acc,
        'wm_ratio_observed_mean': (sum(wm_ratios) / len(wm_ratios)) if wm_ratios else None,
        'wm_ratio_observed_max': max(wm_ratios) if wm_ratios else None,
        'wm_ratio_samples': len(wm_ratios),
        'wm_ratio_after_cap_mean': (sum(wm_ratios_after_cap) / len(wm_ratios_after_cap)) if wm_ratios_after_cap else None,
        'wm_ratio_after_cap_max': max(wm_ratios_after_cap) if wm_ratios_after_cap else None,
        'cap_applied_count': cap_applied_count,
        'detection_file': det_path,
        'trace_file': trace_path,
        'excel_file': excel_path,
    }
    result.update(excel_metrics)
    result.update(noise_acc)

    # 综合鲁棒性分数：检测准确率为主，主任务精度为辅
    det = detection_accuracy if detection_accuracy is not None else (log_detect_acc or 0.0)
    acc = excel_metrics.get('best_val_acc') or 0.0
    result['robustness_score'] = round(det * 0.7 + acc * 0.3, 3)

    return result


# --------------------------------------------------------------------------- #
# 汇总与报告
# --------------------------------------------------------------------------- #

def print_report(results, tag):
    print()
    print('=' * 92)
    print(f'watermark_grad_target_ratio 敏感性汇总  [{tag}]')
    print('=' * 92)

    header = (
        f"{'ratio':>6} | {'检测准确率%':>10} | {'追踪准确率%':>10} | "
        f"{'最佳精度%':>9} | {'capping前':>9} | {'capping后':>9} | {'鲁棒性分':>8} | {'耗时s':>6}"
    )
    print(header)
    print('-' * 92)

    for r in sorted(results, key=lambda x: x['target_ratio']):
        det = r.get('detection_accuracy')
        det = f"{det:.2f}" if det is not None else 'N/A'
        tr = r.get('trace_accuracy')
        tr = f"{tr:.2f}" if tr is not None else 'N/A'
        acc = r.get('best_val_acc')
        acc = f"{acc:.2f}" if acc is not None else 'N/A'
        obs = r.get('wm_ratio_observed_mean')
        obs = f"{obs:.3f}" if obs is not None else 'N/A'
        after = r.get('wm_ratio_after_cap_mean')
        after = f"{after:.3f}" if after is not None else 'N/A'
        score = f"{r.get('robustness_score', 0):.2f}"
        dur = f"{r.get('duration_seconds', 0):.0f}"
        print(f"{r['target_ratio']:>6.2f} | {det:>10} | {tr:>10} | {acc:>9} | "
              f"{obs:>9} | {after:>9} | {score:>8} | {dur:>6}")

    print('-' * 92)

    ok = [r for r in results if r.get('success')]
    if ok:
        best = max(ok, key=lambda x: x.get('robustness_score', 0))
        print(f"🏆 鲁棒性最优 ratio = {best['target_ratio']} "
              f"(检测准确率 {best.get('detection_accuracy')}%, "
              f"鲁棒性分 {best.get('robustness_score')})")
    else:
        print('⚠️  没有成功的实验')

    # capping 生效性检查：capping 之后的实际比例应贴合目标值
    print()
    print('capping 生效性检查（capping 后的实际 wm/non_wm 比例 vs 目标值）:')
    for r in sorted(results, key=lambda x: x['target_ratio']):
        after = r.get('wm_ratio_after_cap_mean')
        if after is None:
            print(f"  ratio={r['target_ratio']:<5.2f} 无数据")
            continue
        tgt = r['target_ratio']
        n_cap = r.get('cap_applied_count', 0)
        if tgt <= 0:
            verdict = '已关闭 capping（梯度不缩放）'
        elif n_cap == 0:
            verdict = '未触发 capping（原比例本就低于目标）'
        elif abs(after - tgt) <= max(0.05, tgt * 0.2):
            verdict = f'OK（贴合目标，触发 {n_cap} 次）'
        else:
            verdict = f'⚠️ 偏离目标（触发 {n_cap} 次）'
        print(f"  ratio={tgt:<5.2f} capping后均值={after:<8.3f} {verdict}")

    print('=' * 92)


def save_summary(results, tag):
    out_dir = _ensure_dir(os.path.join('save', 'ratio_sweep', tag))
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    path = os.path.join(out_dir, f'summary_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'tag': tag,
            'generated_at': ts,
            'results': results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f'📁 汇总已保存: {path}')
    return path


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='watermark_grad_target_ratio 对水印鲁棒性影响的扫描实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['chestmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--model_name', type=str, default='alexnet',
                        choices=['alexnet', 'resnet'])
    parser.add_argument('--client_num', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30,
                        help='通信轮数（排查阶段建议先小值验证）')
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--leak_interval', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ratios', type=float, nargs='+', default=None,
                        help=f'要扫描的 ratio 列表，默认 {DEFAULT_RATIOS}')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='单次实验超时秒数')
    parser.add_argument('--analyze-only', action='store_true',
                        help='不重训，仅从已有结果目录汇总')
    parser.add_argument('--tag', type=str, default=None,
                        help='实验标签，默认由 dataset/model/K 自动生成')
    args = parser.parse_args()

    ratios = args.ratios if args.ratios else DEFAULT_RATIOS
    tag = args.tag or f'{args.dataset}_{args.model_name}_K{args.client_num}_ep{args.epochs}'

    print('=' * 78)
    print('watermark_grad_target_ratio 敏感性扫描')
    print('=' * 78)
    print(f'数据集      : {args.dataset}')
    print(f'模型        : {args.model_name}')
    print(f'客户端数 K  : {args.client_num}')
    print(f'通信轮数    : {args.epochs}')
    print(f'扫描 ratio  : {ratios}')
    print(f'输出根目录  : save/ratio_sweep/{tag}')
    print('=' * 78)

    results = []

    if args.analyze_only:
        base = os.path.join('save', 'ratio_sweep', tag)
        for ratio in ratios:
            ratio_tag = f'ratio{ratio:.2f}'.replace('.', 'p')
            out_dir = os.path.join(base, ratio_tag)
            if not os.path.isdir(out_dir):
                continue
            log_file = os.path.join(out_dir, 'train.log')
            wm_ratios, wm_ratios_after_cap, cap_applied_count = _parse_wm_ratios(log_file)
            log_detect, log_trace = _parse_acc_from_log(log_file)
            # 优先使用隔离目录中已归档的副本
            det_copies = sorted(glob.glob(os.path.join(out_dir, 'watermark_deviation_detection_*.json')))
            trace_copies = sorted(glob.glob(os.path.join(out_dir, 'trace_results_*.json')))
            det_path = det_copies[-1] if det_copies else _latest_detection_json(out_dir, args.model_name, args.dataset)
            trace_path = trace_copies[-1] if trace_copies else _latest_trace_json(out_dir, args.model_name, args.dataset)
            excel_path = _latest_excel(out_dir)

            detection_accuracy = None
            if det_path:
                with open(det_path, 'r', encoding='utf-8') as f:
                    detection_accuracy = json.load(f).get('detection_accuracy')
            trace_accuracy = None
            if trace_path:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    tr = json.load(f)
                for key in ('accuracy', 'overall_accuracy', 'trace_accuracy'):
                    if tr.get(key) is not None:
                        trace_accuracy = float(tr[key])
                        break

            excel_metrics = _read_excel_metrics(excel_path)
            r = {
                'target_ratio': ratio, 'tag': tag, 'success': True,
                'out_dir': out_dir, 'log_file': log_file,
                'detection_accuracy': detection_accuracy,
                'trace_accuracy': trace_accuracy,
                'log_detection_accuracy': log_detect,
                'log_trace_accuracy': log_trace,
                'wm_ratio_observed_mean': (sum(wm_ratios) / len(wm_ratios)) if wm_ratios else None,
                'wm_ratio_observed_max': max(wm_ratios) if wm_ratios else None,
                'wm_ratio_samples': len(wm_ratios),
                'wm_ratio_after_cap_mean': (sum(wm_ratios_after_cap) / len(wm_ratios_after_cap)) if wm_ratios_after_cap else None,
                'wm_ratio_after_cap_max': max(wm_ratios_after_cap) if wm_ratios_after_cap else None,
                'cap_applied_count': cap_applied_count,
            }
            r.update(excel_metrics)
            det = detection_accuracy if detection_accuracy is not None else (log_detect or 0.0)
            acc = excel_metrics.get('best_val_acc') or 0.0
            r['robustness_score'] = round(det * 0.7 + acc * 0.3, 3)
            results.append(r)
    else:
        for i, ratio in enumerate(sorted(ratios), 1):
            print(f'\n>>> 实验 {i}/{len(ratios)}: ratio = {ratio}')
            try:
                r = run_one_ratio(args, ratio, tag)
            except Exception as exc:  # noqa: BLE001
                print(f'❌ 实验异常: {exc}')
                r = {'target_ratio': ratio, 'tag': tag, 'success': False,
                     'error': str(exc)}
            results.append(r)
            # 每跑完一个就落盘，避免中断丢失结果
            save_summary(results, tag)

    if results:
        print_report(results, tag)
        save_summary(results, tag)
    else:
        print('没有可汇总的结果')


if __name__ == '__main__':
    main()
