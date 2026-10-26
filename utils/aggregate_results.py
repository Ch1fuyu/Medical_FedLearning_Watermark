# -*- coding: utf-8 -*-
"""
联邦学习水印实验结果整合工具
================================

将分散在 ``save/trace`` 与 ``save/detection`` 两个目录下的 JSON 结果文件
(两类泄露场景: 训练前模型泄漏追踪 / 训练后水印偏差检测) 合并为便于统计的
结构化数据集。

输入文件结构
------------
1) ``save/trace/trace_results_{model}_{dataset}_{ts}.json``
   - ``total_leakages`` / ``overall_correct`` / ``overall_accuracy``
   - ``noise_level_stats``: 三档高斯噪声 (weak/medium/strong) 下的追踪准确率
   - ``details``: 每轮记录, 含 ``attack_results`` (各噪声等级) 与
     ``similarity_rankings`` (按相似度降序的 [client_id, similarity])

2) ``save/detection/watermark_deviation_detection_{model}_{dataset}_{ts}.json``
   - ``experiment_info``: model / dataset / client_num / epochs / leak_interval
   - ``total_detections`` / ``correct_detections`` / ``detection_accuracy``
   - ``threshold_stats``: 全局 mean / std / threshold_3sigma
   - ``rounds``: 每轮的阈值统计与逐客户端评估 (similarity / watermark_mean /
     watermark_std / similarity_rank / is_suspicious)

输出 (默认写入 ``save/aggregate``)
----------------------------------
- ``aggregate_summary.json``     合并后的完整数据 (含元信息)
- ``summary_by_experiment.csv``  每个实验一行的汇总表
- ``rounds_merged.csv``          逐轮合并明细 (追踪 + 检测)
- ``leakage_events.csv``         逐 (轮次 x 噪声等级) 的泄漏事件明细
- ``client_assessments.csv``     逐 (轮次 x 客户端) 的检测评估明细
- ``aggregate_results.xlsx``     以上全部表格 (多 sheet), 便于直接透视统计

用法
----
    python -m utils.aggregate_results                 # 默认取最新 18 个文件
    python -m utils.aggregate_results --top-n 30      # 取最新 30 个
    python -m utils.aggregate_results --out-dir save/aggregate_v2
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TRACE_DIR = os.path.join(PROJECT_ROOT, 'save', 'trace')
DEFAULT_DETECTION_DIR = os.path.join(PROJECT_ROOT, 'save', 'detection')
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, 'save', 'aggregate')

TRACE_PREFIX = 'trace_results_'
DETECTION_PREFIX = 'watermark_deviation_detection_'

NOISE_NAME_MAP = {1: 'weak', 2: 'medium', 3: 'strong'}
NOISE_SCALE_MAP = {1: 0.01, 2: 0.05, 3: 0.10}

TRACE_FILENAME_RE = re.compile(
    r'^trace_results_(?P<model>.+)_(?P<dataset>.+)_(?P<ts>\d{14})\.json$'
)
DETECTION_FILENAME_RE = re.compile(
    r'^watermark_deviation_detection_(?P<model>.+)_(?P<dataset>.+)_(?P<ts>\d{14})\.json$'
)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def parse_ts(ts_str):
    """'20261021225148' -> datetime"""
    from datetime import datetime
    try:
        return datetime.strptime(ts_str, '%Y%m%d%H%M%S')
    except ValueError:
        return datetime.min


def safe_float(value):
    """numpy / None / NaN -> python float or None"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def load_json(path):
    """安全读取 JSON, 失败返回 (None, error_message)"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as exc:  # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'


def list_trace_files(trace_dir):
    """按修改时间倒序返回 trace 文件路径列表 (只保留能解析出文件名元信息的)"""
    files = []
    for path in glob.glob(os.path.join(trace_dir, f'{TRACE_PREFIX}*.json')):
        m = TRACE_FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        files.append({
            'path': path,
            'name': os.path.basename(path),
            'model': m.group('model'),
            'dataset': m.group('dataset'),
            'timestamp_str': m.group('ts'),
            'timestamp': parse_ts(m.group('ts')),
            'mtime': os.path.getmtime(path),
        })
    files.sort(key=lambda x: (x['timestamp'], x['mtime']), reverse=True)
    return files


def list_detection_files(detection_dir):
    """同上, 返回 detection 文件列表"""
    files = []
    for path in glob.glob(os.path.join(detection_dir, f'{DETECTION_PREFIX}*.json')):
        m = DETECTION_FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        files.append({
            'path': path,
            'name': os.path.basename(path),
            'model': m.group('model'),
            'dataset': m.group('dataset'),
            'timestamp_str': m.group('ts'),
            'timestamp': parse_ts(m.group('ts')),
            'mtime': os.path.getmtime(path),
        })
    files.sort(key=lambda x: (x['timestamp'], x['mtime']), reverse=True)
    return files


def infer_client_num(trace_data):
    """从 similarity_rankings 长度推断客户端数量"""
    details = trace_data.get('details') or []
    for d in details:
        rankings = d.get('similarity_rankings') or []
        if rankings:
            return len(rankings)
    return None


# --------------------------------------------------------------------------- #
# 单文件解析
# --------------------------------------------------------------------------- #

def parse_trace_file(meta, data):
    """把一个 trace JSON 转成规范化 dict"""
    details = data.get('details') or []
    client_num = infer_client_num(data)

    rounds = []
    for det in details:
        rankings = det.get('similarity_rankings') or []
        actual = det.get('actual_leaked_client')

        sim_of_actual = None
        for cid, sim in rankings:
            if cid == actual:
                sim_of_actual = safe_float(sim)
                break
        top1_sim = safe_float(rankings[0][1]) if rankings else None

        # Top-1 永远是泄漏者, 因此 "Top1 - 实际" 的 margin 恒为 0。
        # 有意义的 margin 应是 "实际泄漏者相似度 - 次高(非泄漏者)相似度"。
        other_sims = [safe_float(s) for cid, s in rankings if cid != actual]
        other_sims = [s for s in other_sims if s is not None]
        second_sim = max(other_sims) if other_sims else None
        actual_rank = None
        for idx, (cid, _sim) in enumerate(rankings, 1):
            if cid == actual:
                actual_rank = idx
                break

        rounds.append({
            'round': det.get('round'),
            'actual_leaked_client': actual,
            'detected_leaked_client': det.get('detected_leaked_client'),
            'is_correct': det.get('is_correct'),
            'similarity_actual': sim_of_actual,
            'similarity_top1': top1_sim,
            'similarity_second': second_sim,
            'actual_rank': actual_rank,
            'similarity_margin': (None if sim_of_actual is None or second_sim is None
                                  else round(sim_of_actual - second_sim, 8)),
            'similarity_rankings': [[int(c), safe_float(s)] for c, s in rankings],
            'attack_results': det.get('attack_results') or {},
        })

    level_stats = data.get('noise_level_stats') or {}

    return {
        'model': meta['model'],
        'dataset': meta['dataset'],
        'client_num': client_num,
        'trace_file': meta['name'],
        'trace_path': meta['path'],
        'timestamp': meta['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp_str': meta['timestamp_str'],
        'total_leakages': data.get('total_leakages'),
        'overall_correct': data.get('overall_correct'),
        'overall_accuracy': safe_float(data.get('overall_accuracy')),
        'attack_mode': data.get('attack_mode'),
        'description': data.get('description'),
        'noise_level_stats': level_stats,
        'rounds': rounds,
    }


def parse_detection_file(meta, data):
    """把一个 detection JSON 转成规范化 dict"""
    info = data.get('experiment_info') or {}

    rounds = []
    for r in data.get('rounds') or []:
        assessments = r.get('client_assessments') or {}
        for cid, v in assessments.items():
            pass  # 逐客户端明细在下面单独展开

        rounds.append({
            'round': r.get('round'),
            'detected_malicious_client': r.get('detected_malicious_client'),
            'actual_leaked_client': r.get('actual_leaked_client'),
            'is_correct': r.get('is_correct'),
            'threshold_mean': safe_float((r.get('threshold_info') or {}).get('mean')),
            'threshold_std': safe_float((r.get('threshold_info') or {}).get('std')),
            'threshold_2sigma': safe_float((r.get('threshold_info') or {}).get('threshold_2sigma')),
            'client_assessments': assessments,
        })

    thr = data.get('threshold_stats') or {}

    return {
        'model': info.get('model') or meta['model'],
        'dataset': info.get('dataset') or meta['dataset'],
        'client_num': info.get('client_num'),
        'epochs': info.get('epochs'),
        'leak_interval': info.get('leak_interval'),
        'detection_file': meta['name'],
        'detection_path': meta['path'],
        'timestamp': meta['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp_str': meta['timestamp_str'],
        'total_detections': data.get('total_detections'),
        'correct_detections': data.get('correct_detections'),
        'detection_accuracy': safe_float(data.get('detection_accuracy')),
        'threshold_mean': safe_float(thr.get('mean')),
        'threshold_std': safe_float(thr.get('std')),
        'threshold_3sigma': safe_float(thr.get('threshold_3sigma')),
        'rounds': rounds,
    }


# --------------------------------------------------------------------------- #
# 配对: 把 trace 与 detection 文件按 (model, dataset, client_num) 关联
# --------------------------------------------------------------------------- #

def pair_files(trace_files, detection_files, trace_parsed, det_parsed):
    """
    优先使用 (model, dataset, client_num) 做主键配对;
    若 client_num 无法确定, 回退到 (model, dataset) + 时间戳最近。
    返回 (pairs, unmatched_trace, unmatched_detection)
    """
    det_groups = defaultdict(list)
    for meta in detection_files:
        parsed = det_parsed.get(meta['path'])
        if parsed is None:
            continue
        key = (parsed['model'], parsed['dataset'], parsed['client_num'])
        det_groups[key].append((meta, parsed))
    for lst in det_groups.values():
        lst.sort(key=lambda x: x[0]['timestamp'], reverse=True)

    pairs, unmatched_trace = [], []
    used_det = set()

    for meta in trace_files:
        parsed = trace_parsed.get(meta['path'])
        if parsed is None:
            continue
        key = (parsed['model'], parsed['dataset'], parsed['client_num'])
        bucket = det_groups.get(key)
        if not bucket:
            unmatched_trace.append((meta, parsed))
            continue
        # 取同 key 中时间最接近且未被占用的 detection
        best = None
        for dm, dp in bucket:
            if dm['path'] in used_det:
                continue
            diff = abs((dm['timestamp'] - meta['timestamp']).total_seconds())
            if best is None or diff < best[0]:
                best = (diff, dm, dp)
        if best is None:
            unmatched_trace.append((meta, parsed))
            continue
        _, dm, dp = best
        used_det.add(dm['path'])
        pairs.append({'trace': parsed, 'detection': dp})

    unmatched_detection = [
        (m, det_parsed[m['path']])
        for m in detection_files
        if m['path'] in det_parsed and m['path'] not in used_det
    ]
    return pairs, unmatched_trace, unmatched_detection


# --------------------------------------------------------------------------- #
# 表格构建
# --------------------------------------------------------------------------- #

def build_summary_rows(pairs):
    """每个实验一行"""
    rows = []
    for p in pairs:
        t, d = p['trace'], p['detection']
        row = {
            'model': t['model'],
            'dataset': t['dataset'],
            'client_num': t['client_num'] or d['client_num'],
            'epochs': d['epochs'],
            'leak_interval': d['leak_interval'],
            'timestamp': t['timestamp'],
            'trace_file': t['trace_file'],
            'detection_file': d['detection_file'],
            'n_rounds_trace': t['total_leakages'],
            'n_rounds_detection': d['total_detections'],
            'trace_accuracy': t['overall_accuracy'],
            'detection_accuracy': d['detection_accuracy'],
            'threshold_mean': d['threshold_mean'],
            'threshold_std': d['threshold_std'],
            'threshold_3sigma': d['threshold_3sigma'],
        }
        for lvl in (1, 2, 3):
            st = (t['noise_level_stats'] or {}).get(str(lvl), {})
            row[f'noise{lvl}_name'] = st.get('noise_name', NOISE_NAME_MAP.get(lvl))
            row[f'noise{lvl}_scale'] = st.get('noise_scale', NOISE_SCALE_MAP.get(lvl))
            row[f'noise{lvl}_accuracy'] = safe_float(st.get('accuracy'))
            row[f'noise{lvl}_correct'] = st.get('correct')
            row[f'noise{lvl}_total'] = st.get('total')
        rows.append(row)
    return rows


def build_rounds_rows(pairs):
    """逐轮合并: 追踪结果 + 检测结果"""
    rows = []
    for p in pairs:
        t, d = p['trace'], p['detection']
        det_by_round = {r['round']: r for r in d['rounds']}
        for tr in t['rounds']:
            dr = det_by_round.get(tr['round'], {})
            row = {
                'model': t['model'],
                'dataset': t['dataset'],
                'client_num': t['client_num'] or d['client_num'],
                'round': tr['round'],
                'trace_actual_leaked_client': tr['actual_leaked_client'],
                'trace_detected_leaked_client': tr['detected_leaked_client'],
                'trace_is_correct': tr['is_correct'],
                'trace_similarity_actual': tr['similarity_actual'],
                'trace_similarity_top1': tr['similarity_top1'],
                'trace_similarity_second': tr['similarity_second'],
                'trace_actual_rank': tr['actual_rank'],
                'trace_similarity_margin': tr['similarity_margin'],
                'detect_actual_leaked_client': dr.get('actual_leaked_client'),
                'detect_detected_malicious_client': dr.get('detected_malicious_client'),
                'detect_is_correct': dr.get('is_correct'),
                'detect_threshold_mean': dr.get('threshold_mean'),
                'detect_threshold_std': dr.get('threshold_std'),
                'detect_threshold_2sigma': dr.get('threshold_2sigma'),
            }
            for lvl in (1, 2, 3):
                ar = (tr['attack_results'] or {}).get(str(lvl))
                if ar:
                    row[f'noise{lvl}_detected_leaker'] = ar.get('detected_leaker')
                    row[f'noise{lvl}_is_correct'] = ar.get('is_correct')
                    row[f'noise{lvl}_similarity_actual'] = None
                    for cid, sim in ar.get('similarity_rankings') or []:
                        if cid == tr['actual_leaked_client']:
                            row[f'noise{lvl}_similarity_actual'] = safe_float(sim)
                            break
                    row[f'noise{lvl}_similarity_top1'] = (
                        safe_float(ar['similarity_rankings'][0][1])
                        if ar.get('similarity_rankings') else None
                    )
                    # 次高相似度与 margin (泄漏者 vs 次高)
                    _rk = ar.get('similarity_rankings') or []
                    _others = [safe_float(s) for c, s in _rk if c != tr['actual_leaked_client']]
                    _others = [s for s in _others if s is not None]
                    row[f'noise{lvl}_similarity_second'] = max(_others) if _others else None
                    row[f'noise{lvl}_margin'] = (
                        None if row[f'noise{lvl}_similarity_actual'] is None
                        or row[f'noise{lvl}_similarity_second'] is None
                        else round(row[f'noise{lvl}_similarity_actual']
                                   - row[f'noise{lvl}_similarity_second'], 8)
                    )
                else:
                    row[f'noise{lvl}_detected_leaker'] = None
                    row[f'noise{lvl}_is_correct'] = None
                    row[f'noise{lvl}_similarity_actual'] = None
                    row[f'noise{lvl}_similarity_top1'] = None
                    row[f'noise{lvl}_similarity_second'] = None
                    row[f'noise{lvl}_margin'] = None
            rows.append(row)
    return rows


def build_leakage_event_rows(pairs):
    """逐 (轮次 x 噪声等级) 的泄漏事件"""
    rows = []
    for p in pairs:
        t, d = p['trace'], p['detection']
        for tr in t['rounds']:
            levels = tr['attack_results'] or {}
            if not levels:
                levels = {'2': {
                    'detected_leaker': tr['detected_leaked_client'],
                    'is_correct': tr['is_correct'],
                    'similarity_rankings': tr['similarity_rankings'],
                }}
            for lvl, ar in levels.items():
                rankings = ar.get('similarity_rankings') or []
                sim_actual = None
                for cid, sim in rankings:
                    if cid == tr['actual_leaked_client']:
                        sim_actual = safe_float(sim)
                        break
                top1 = safe_float(rankings[0][1]) if rankings else None
                others = [safe_float(s) for c, s in rankings if c != tr['actual_leaked_client']]
                others = [s for s in others if s is not None]
                second = max(others) if others else None
                rows.append({
                    'model': t['model'],
                    'dataset': t['dataset'],
                    'client_num': t['client_num'] or d['client_num'],
                    'round': tr['round'],
                    'noise_level': int(lvl) if str(lvl).isdigit() else lvl,
                    'noise_name': ar.get('noise_name') or NOISE_NAME_MAP.get(int(lvl) if str(lvl).isdigit() else lvl),
                    'noise_scale': ar.get('noise_scale') if ar.get('noise_scale') is not None
                    else NOISE_SCALE_MAP.get(int(lvl) if str(lvl).isdigit() else lvl),
                    'actual_leaker': tr['actual_leaked_client'],
                    'detected_leaker': ar.get('detected_leaker'),
                    'is_correct': ar.get('is_correct'),
                    'similarity_actual': sim_actual,
                    'similarity_top1': top1,
                    'similarity_second': second,
                    'similarity_margin': (None if sim_actual is None or second is None
                                          else round(sim_actual - second, 8)),
                })
    return rows


def build_detection_margin_rows(pairs):
    """
    逐轮的检测判别余量: 真实泄漏者相似度 - 其余客户端最高相似度。
    同时记录该轮 2sigma 阈值, 便于判断阈值是否真的起作用。
    """
    rows = []
    for p in pairs:
        d = p['detection']
        for r in d['rounds']:
            assessments = r.get('client_assessments') or {}
            actual = r.get('actual_leaked_client')
            leak_sim, other_max, n_susp, rank = None, None, 0, None
            for cid, v in assessments.items():
                sim = safe_float(v.get('similarity'))
                if v.get('is_suspicious'):
                    n_susp += 1
                if cid == str(actual) or (actual is not None and str(actual) == str(cid)):
                    leak_sim = sim
                    rank = v.get('similarity_rank')
                elif sim is not None:
                    other_max = sim if other_max is None else max(other_max, sim)
            rows.append({
                'model': d['model'],
                'dataset': d['dataset'],
                'client_num': d['client_num'],
                'round': r['round'],
                'actual_leaked_client': actual,
                'detected_malicious_client': r.get('detected_malicious_client'),
                'is_correct': r.get('is_correct'),
                'leaker_similarity': leak_sim,
                'max_other_similarity': other_max,
                'margin': (None if leak_sim is None or other_max is None
                           else round(leak_sim - other_max, 8)),
                'leaker_rank': rank,
                'n_suspicious': n_susp,
                'threshold_mean': r.get('threshold_mean'),
                'threshold_std': r.get('threshold_std'),
                'threshold_2sigma': r.get('threshold_2sigma'),
                'leaker_above_threshold': (
                    None if leak_sim is None or r.get('threshold_2sigma') is None
                    else bool(leak_sim > r.get('threshold_2sigma'))
                ),
            })
    return rows


def build_trace_similarity_rows(pairs):
    """
    溯源场景的长表: 逐 (轮次 x 噪声等级 x 客户端) 的相似度与标签。
    label=1 表示该客户端是真实泄漏者, 0 表示非泄漏者。
    用于构造 "K 类溯源" 的二分类等价 ROC / PR。
    """
    rows = []
    for p in pairs:
        t = p['trace']
        for tr in t['rounds']:
            actual = tr['actual_leaked_client']
            levels = tr['attack_results'] or {}
            if not levels:
                levels = {'2': {
                    'detected_leaker': tr['detected_leaked_client'],
                    'is_correct': tr['is_correct'],
                    'similarity_rankings': tr['similarity_rankings'],
                }}
            for lvl, ar in levels.items():
                rankings = ar.get('similarity_rankings') or []
                detected = ar.get('detected_leaker')
                for rank, (cid, sim) in enumerate(rankings, 1):
                    rows.append({
                        'model': t['model'],
                        'dataset': t['dataset'],
                        'client_num': t['client_num'],
                        'round': tr['round'],
                        'noise_level': int(lvl) if str(lvl).isdigit() else lvl,
                        'client_id': int(cid),
                        'similarity': safe_float(sim),
                        'rank': rank,
                        'is_leaker': bool(cid == actual),
                        'is_detected': bool(cid == detected),
                        'detected_leaker': detected,
                        'actual_leaker': actual,
                    })
    return rows


def build_client_assessment_rows(pairs):
    """逐 (轮次 x 客户端) 的检测评估"""
    rows = []
    for p in pairs:
        d = p['detection']
        for r in d['rounds']:
            for cid, v in (r.get('client_assessments') or {}).items():
                rows.append({
                    'model': d['model'],
                    'dataset': d['dataset'],
                    'client_num': d['client_num'],
                    'round': r['round'],
                    'client_id': int(cid) if str(cid).isdigit() else cid,
                    'similarity': safe_float(v.get('similarity')),
                    'watermark_mean': safe_float(v.get('watermark_mean')),
                    'watermark_std': safe_float(v.get('watermark_std')),
                    'similarity_rank': v.get('similarity_rank'),
                    'is_suspicious': v.get('is_suspicious'),
                    'actual_leaked_client': r.get('actual_leaked_client'),
                    'detected_malicious_client': r.get('detected_malicious_client'),
                    'is_correct': r.get('is_correct'),
                    'threshold_2sigma': r.get('threshold_2sigma'),
                })
    return rows


def build_pivot_tables(summary_df):
    """生成若干常用透视表, 便于直接看趋势"""
    tables = {}

    if summary_df.empty:
        return tables

    def _pivot(index, columns, values):
        try:
            return summary_df.pivot_table(
                index=index, columns=columns, values=values, aggfunc='mean'
            )
        except Exception:  # noqa: BLE001
            return None

    tables['trace_acc_by_model_dataset'] = _pivot(
        ['model', 'dataset'], 'client_num', 'trace_accuracy')
    tables['detect_acc_by_model_dataset'] = _pivot(
        ['model', 'dataset'], 'client_num', 'detection_accuracy')
    tables['trace_acc_by_clientnum'] = _pivot(
        ['client_num'], ['model', 'dataset'], 'trace_accuracy')
    tables['detect_acc_by_clientnum'] = _pivot(
        ['client_num'], ['model', 'dataset'], 'detection_accuracy')

    noise_cols = [f'noise{l}_accuracy' for l in (1, 2, 3)]
    noise = summary_df.groupby(['model', 'dataset', 'client_num'])[noise_cols].mean()
    tables['noise_accuracy'] = noise
    return tables


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def aggregate(trace_dir=DEFAULT_TRACE_DIR,
              detection_dir=DEFAULT_DETECTION_DIR,
              top_n=18,
              out_dir=DEFAULT_OUT_DIR,
              verbose=True):
    """执行整合, 返回结果字典"""
    trace_files = list_trace_files(trace_dir)
    det_files = list_detection_files(detection_dir)

    if top_n and top_n > 0:
        trace_files = trace_files[:top_n]
        det_files = det_files[:top_n]

    if verbose:
        print(f'[INFO] 选中 trace 文件 {len(trace_files)} 个, detection 文件 {len(det_files)} 个')

    trace_parsed, det_parsed = {}, {}
    skipped = []

    for meta in trace_files:
        data, err = load_json(meta['path'])
        if data is None:
            skipped.append({'file': meta['name'], 'error': err})
            continue
        if 'overall_accuracy' not in data or 'details' not in data:
            skipped.append({'file': meta['name'], 'error': '旧格式文件(缺少 overall_accuracy/details)'})
            continue
        trace_parsed[meta['path']] = parse_trace_file(meta, data)

    for meta in det_files:
        data, err = load_json(meta['path'])
        if data is None:
            skipped.append({'file': meta['name'], 'error': err})
            continue
        if 'rounds' not in data:
            skipped.append({'file': meta['name'], 'error': '缺少 rounds 字段'})
            continue
        det_parsed[meta['path']] = parse_detection_file(meta, data)

    pairs, unmatched_trace, unmatched_detection = pair_files(
        trace_files, det_files, trace_parsed, det_parsed)

    if verbose:
        print(f'[INFO] 成功配对 {len(pairs)} 组, '
              f'未配对 trace {len(unmatched_trace)} 个, '
              f'未配对 detection {len(unmatched_detection)} 个, '
              f'跳过 {len(skipped)} 个文件')
        for m, _ in unmatched_trace:
            print(f'  [WARN] trace 无对应 detection: {m["name"]}')
        for m, _ in unmatched_detection:
            print(f'  [WARN] detection 无对应 trace: {m["name"]}')
        for s in skipped:
            print(f'  [SKIP] {s["file"]}: {s["error"]}')

    # ---------- 表格 ----------
    summary_df = pd.DataFrame(build_summary_rows(pairs))
    rounds_df = pd.DataFrame(build_rounds_rows(pairs))
    events_df = pd.DataFrame(build_leakage_event_rows(pairs))
    assess_df = pd.DataFrame(build_client_assessment_rows(pairs))
    detmargin_df = pd.DataFrame(build_detection_margin_rows(pairs))
    tracesim_df = pd.DataFrame(build_trace_similarity_rows(pairs))
    pivots = build_pivot_tables(summary_df)

    # ---------- 保存 ----------
    os.makedirs(out_dir, exist_ok=True)

    # 1) 合并后的 JSON
    merged_json = os.path.join(out_dir, 'aggregate_summary.json')
    payload = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_dirs': {'trace': trace_dir, 'detection': detection_dir},
        'n_paired': len(pairs),
        'skipped_files': skipped,
        'unmatched_trace': [m['name'] for m, _ in unmatched_trace],
        'unmatched_detection': [m['name'] for m, _ in unmatched_detection],
        'experiments': [
            {
                'model': p['trace']['model'],
                'dataset': p['trace']['dataset'],
                'client_num': p['trace']['client_num'] or p['detection']['client_num'],
                'timestamp': p['trace']['timestamp'],
                'trace_file': p['trace']['trace_file'],
                'detection_file': p['detection']['detection_file'],
                'trace_accuracy': p['trace']['overall_accuracy'],
                'detection_accuracy': p['detection']['detection_accuracy'],
                'noise_level_stats': p['trace']['noise_level_stats'],
                'threshold_stats': {
                    'mean': p['detection']['threshold_mean'],
                    'std': p['detection']['threshold_std'],
                    'threshold_3sigma': p['detection']['threshold_3sigma'],
                },
            }
            for p in pairs
        ],
    }
    with open(merged_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 2) CSV
    csv_paths = {}
    for name, df in [('summary_by_experiment', summary_df),
                     ('rounds_merged', rounds_df),
                     ('leakage_events', events_df),
                     ('client_assessments', assess_df),
                     ('detection_margins', detmargin_df),
                     ('trace_similarities', tracesim_df)]:
        p = os.path.join(out_dir, f'{name}.csv')
        df.to_csv(p, index=False, encoding='utf-8-sig')
        csv_paths[name] = p

    # 3) Excel
    xlsx_path = os.path.join(out_dir, 'aggregate_results.xlsx')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            rounds_df.to_excel(writer, sheet_name='Rounds_Merged', index=False)
            events_df.to_excel(writer, sheet_name='Leakage_Events', index=False)
            assess_df.to_excel(writer, sheet_name='Client_Assessments', index=False)
            detmargin_df.to_excel(writer, sheet_name='Detection_Margins', index=False)
            tracesim_df.to_excel(writer, sheet_name='Trace_Similarities', index=False)
            for name, tbl in pivots.items():
                if tbl is None or tbl.empty:
                    continue
                t = tbl.reset_index()
                # 多级列名展平为单级, 便于 Excel 导出
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = [
                        '_'.join(str(c) for c in col if str(c) != '')
                        for col in t.columns.to_flat_index()
                    ]
                sheet = name[:31]
                t.to_excel(writer, sheet_name=sheet, index=False)
            if skipped:
                pd.DataFrame(skipped).to_excel(writer, sheet_name='Skipped_Files', index=False)
    except Exception as exc:  # noqa: BLE001
        print(f'[WARN] Excel 导出失败: {exc}')
        xlsx_path = None

    # ---------- 控制台摘要 ----------
    if verbose and not summary_df.empty:
        cols = ['model', 'dataset', 'client_num', 'trace_accuracy',
                'noise1_accuracy', 'noise2_accuracy', 'noise3_accuracy',
                'detection_accuracy']
        cols = [c for c in cols if c in summary_df.columns]
        print('\n=== 汇总 (每个实验一行) ===')
        print(summary_df[cols].to_string(index=False))

    result = {
        'out_dir': out_dir,
        'merged_json': merged_json,
        'csv_paths': csv_paths,
        'xlsx_path': xlsx_path,
        'n_pairs': len(pairs),
        'skipped': skipped,
        'summary_df': summary_df,
        'rounds_df': rounds_df,
        'events_df': events_df,
        'assess_df': assess_df,
        'detmargin_df': detmargin_df,
        'tracesim_df': tracesim_df,
        'pivots': pivots,
    }
    if verbose:
        print(f'\n[OK] 输出目录: {out_dir}')
        print(f'[OK] 合并 JSON: {merged_json}')
        for k, v in csv_paths.items():
            print(f'[OK] CSV  {k}: {v}')
        if xlsx_path:
            print(f'[OK] Excel: {xlsx_path}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='整合 save/trace 与 save/detection 下的水印实验 JSON 结果')
    parser.add_argument('--trace-dir', default=DEFAULT_TRACE_DIR,
                        help='trace 结果目录')
    parser.add_argument('--detection-dir', default=DEFAULT_DETECTION_DIR,
                        help='detection 结果目录')
    parser.add_argument('--top-n', type=int, default=18,
                        help='每个目录取最新的 N 个文件 (默认 18, 传 0 表示全部)')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                        help='输出目录')
    args = parser.parse_args(argv)

    aggregate(
        trace_dir=args.trace_dir,
        detection_dir=args.detection_dir,
        top_n=args.top_n,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
