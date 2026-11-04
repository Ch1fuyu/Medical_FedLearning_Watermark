"""
校验 pkl 模型中的「水印区域参数」是否与初始编码器参数一致

背景
----
联邦学习结束时保存到磁盘的 pkl 里，`net_info['best_model'][0]` 是
**FedAvg 聚合后的下发模型 w_t**（水印区与非水印区一视同仁做加权平均）。
而「权威水印值」来自自编码器 encoder 的扁平参数：
    客户端 k 的权威值 = encoder_flat[offset_k : offset_k + len(P_k)]
两者在理论上**并不相等**（除非水印区完全没有被平均）。

本脚本用于定量回答这个问题。

用法
----
    # 单个 pkl（模型/数据集/客户端数自动从 pkl 元数据 + 文件名推断）
    python utils/check_watermark_in_pkl.py --pkl save/resnet/chestmnist/202611031643_....pkl

    # 数据集目录（递归扫描其下所有 pkl）
    python utils/check_watermark_in_pkl.py --pkl save/resnet/chestmnist

    # 模型根目录（递归扫描全部数据集子目录）
    python utils/check_watermark_in_pkl.py --pkl save/alexnet

    # 多个路径混合
    python utils/check_watermark_in_pkl.py --pkl save/alexnet save/resnet/chestmnist/a.pkl

    # 放宽「一致」判定阈值 + 导出 JSON
    python utils/check_watermark_in_pkl.py --pkl save/alexnet --tol 1e-4 --json save/wm_check.json

注意
----
* 训练过程中编码器会被 `AutoencoderFinetuner` 在**内存中**微调
  （见 utils/trainer_private_enhanced.py::_fine_tune_autoencoder），
  因此训练时实际写入的权威值与 `save/autoencoder/encoder.pth`（初始值）
  存在差异。本脚本比较的对象是**初始编码器参数**，
  差异中既包含「联邦平均稀释」，也包含「编码器微调漂移」。
* pkl 若是以 `watermark_only` 格式保存的（只有水印区参数），
  脚本会自动改用 `client_watermark_values` 字段做比对。
* 模型/客户端数的推断优先级：
      路径 `save/<model>/<dataset>/xxx.pkl`  →  pkl 内 `arguments`  →  文件名 `_cn_<N>_`
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.light_autoencoder import LightAutoencoder  # noqa: E402


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def is_conv_weight(name, tensor):
    """与 utils/key_matrix_utils.py 保持一致的卷积层权重判定"""
    conv = ('conv' in name.lower() and 'weight' in name.lower()) or \
           ('downsample.0.weight' in name.lower())
    return conv and len(tensor.shape) == 4


def build_offset_map(state_dict):
    """
    构建卷积层权重的全局偏移映射。

    state_dict 的键顺序与 model.named_parameters() 的顺序一致
    （都按模块注册顺序递归），因此可以直接用 state_dict 构造。
    """
    offset_map = {}
    current = 0
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if is_conv_weight(name, tensor):
            offset_map[name] = current
            current += tensor.numel()
    return offset_map


def resolve_position(state_dict, offset_map, param_name, idx):
    """
    把位置文件中的 (param_name, idx) 解析成 (实际参数名, 局部索引)。

    兼容两种索引：
      1) 局部索引：idx < param_numel，直接使用
      2) 全局索引：idx >= param_numel，按 offset_map 换算
    """
    if param_name in state_dict:
        size = state_dict[param_name].numel()
        if idx < size:
            return param_name, idx
        if param_name in offset_map:
            off = offset_map[param_name]
            if off <= idx < off + size:
                return param_name, idx - off
        for name, off in offset_map.items():
            size = state_dict[name].numel()
            if off <= idx < off + size:
                return name, idx - off
        return None, None
    # 参数名不匹配：按全局偏移兜底查找
    for name, off in offset_map.items():
        size = state_dict[name].numel()
        if off <= idx < off + size:
            return name, idx - off
    return None, None


_ENCODER_CACHE = {}


def load_encoder_flat(encoder_path):
    """加载初始编码器权重并展平为 1D tensor（顺序同 encoder.parameters()）"""
    if encoder_path in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder_path]
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f'编码器权重不存在: {encoder_path}')
    ae = LightAutoencoder(input_channels=1)
    sd = torch.load(encoder_path, map_location='cpu', weights_only=False)
    ae.encoder.load_state_dict(sd)
    flat = torch.cat([p.detach().view(-1) for p in ae.encoder.parameters()]).float()
    _ENCODER_CACHE[encoder_path] = flat
    return flat


def load_positions(key_matrix_dir, client_id):
    path = os.path.join(key_matrix_dir, f'client_{client_id}', 'positions.json')
    with open(path, 'r') as f:
        positions = json.load(f)
    return [(p[0], int(p[1])) for p in positions]


def encoder_offsets(positions_per_client):
    """各客户端在 encoder_flat 中的起始偏移（按 client_id 顺序连续排布）"""
    offsets = {}
    cur = 0
    for cid in range(len(positions_per_client)):
        offsets[cid] = cur
        cur += len(positions_per_client[cid])
    return offsets


# ----------------------------------------------------------------------------
# pkl 读取
# ----------------------------------------------------------------------------
def read_pkl_model(pkl_path):
    """
    读取 pkl 中的模型权重。

    Returns:
        (state_dict, meta)
        state_dict: {param_name: tensor}，若为 watermark_only 格式则为 None
        meta: 附加信息字典
    """
    obj = torch.load(pkl_path, map_location='cpu', weights_only=False)
    meta = {}

    net_info = obj.get('net_info', {}) if isinstance(obj, dict) else {}
    best = net_info.get('best_model') if isinstance(net_info, dict) else None

    if best is None:
        meta['error'] = 'pkl 中没有 net_info["best_model"]'
        return None, meta

    if isinstance(best, list):
        best = best[0] if best else None
    if not isinstance(best, dict):
        meta['error'] = 'best_model 格式异常'
        return None, meta

    args = obj.get('arguments', {}) or {}
    meta['model_name'] = args.get('model_name')
    meta['client_num'] = args.get('client_num')
    meta['dataset'] = args.get('dataset')
    meta['epochs'] = args.get('epochs')
    meta['watermark_only'] = bool(best.get('_watermark_only', False))

    if meta['watermark_only']:
        # 只有水印区参数，直接返回 client_watermark_values
        meta['client_watermark_values'] = best.get('client_watermark_values', {})
        return None, meta

    state = {k: v.detach().float() for k, v in best.items()
             if torch.is_tensor(v) and not k.startswith('_')}
    return state, meta


def collect_wm_values_from_state(state, offset_map, positions_per_client):
    """
    从完整 state_dict 中抽取各客户端水印区的取值。

    Returns:
        {client_id: [values]}
    """
    out = {}
    for cid, positions in positions_per_client.items():
        vals = []
        for param_name, idx in positions:
            actual, local = resolve_position(state, offset_map, param_name, idx)
            if actual is None:
                continue
            flat = state[actual].view(-1)
            if local >= flat.numel():
                continue
            vals.append(flat[local].item())
        out[cid] = vals
    return out


# ----------------------------------------------------------------------------
# 比对
# ----------------------------------------------------------------------------
def compare(cur_vals, auth_vals, tol):
    """逐元素比对，返回指标字典"""
    if not cur_vals or not auth_vals:
        return {
            'count': len(cur_vals),
            'auth_count': len(auth_vals),
            'error': '数量不匹配，无法逐元素比对',
        }

    # 长度不一致时按较短的一侧截断（并记录差异）
    n = min(len(cur_vals), len(auth_vals))
    mismatch = len(cur_vals) != len(auth_vals)
    cur_vals = cur_vals[:n]
    auth_vals = auth_vals[:n]

    cur = torch.tensor(cur_vals, dtype=torch.float32)
    auth = torch.tensor(auth_vals, dtype=torch.float32)
    diff = cur - auth

    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.abs().mean().item()) if diff.numel() else 0.0
    exact_ratio = float((diff.abs() <= tol).float().mean().item()) if diff.numel() else 0.0
    cos = float(F.cosine_similarity(cur.unsqueeze(0), auth.unsqueeze(0)).item()) \
        if cur.numel() else float('nan')
    rel = float(diff.norm().item() / (auth.norm().item() + 1e-12)) if cur.numel() else float('nan')

    out = {
        'count': int(cur.numel()),
        'cos_sim': cos,
        'rel_dev': rel,
        'max_abs_diff': max_abs,
        'mean_abs_diff': mean_abs,
        'exact_match_ratio': exact_ratio,
        'cur_norm': float(cur.norm().item()),
        'auth_norm': float(auth.norm().item()),
    }
    if mismatch:
        out['count_mismatch'] = True
    return out


def infer_model_and_clients(pkl_path, meta):
    """
    推断模型名与客户端数。

    优先级：
      1) pkl 内 `arguments`（最可靠）
      2) 路径 `save/<model>/<dataset>/xxx.pkl` 中的 <model>
      3) 文件名中的 `_cn_<N>_`
    """
    model_name = meta.get('model_name')
    client_num = meta.get('client_num')

    if not model_name:
        # 从路径里找 save/<model>/<dataset>/ 结构
        parts = os.path.normpath(pkl_path).split(os.sep)
        for i, p in enumerate(parts):
            if p.lower() == 'save' and i + 1 < len(parts):
                cand = parts[i + 1]
                if cand in ('alexnet', 'resnet'):
                    model_name = cand
                    break

    if not client_num:
        import re
        m = re.search(r'_cn_(\d+)_', os.path.basename(pkl_path))
        if m:
            client_num = int(m.group(1))

    return model_name or 'alexnet', client_num


def analyze_pkl(pkl_path, key_matrix_root, encoder_path, tol, verbose=True):
    """分析单个 pkl 文件"""
    state, meta = read_pkl_model(pkl_path)
    result = {'pkl': pkl_path, 'meta': {k: v for k, v in meta.items()
                                        if k != 'client_watermark_values'}}

    if state is None and 'error' in meta:
        result['status'] = 'ERROR'
        result['error'] = meta['error']
        return result

    model_name, client_num = infer_model_and_clients(pkl_path, meta)
    result['meta']['model_name'] = model_name
    result['meta']['client_num'] = client_num

    km_dir = os.path.join(key_matrix_root, model_name, f'client{client_num}')
    if not os.path.isdir(km_dir):
        result['status'] = 'ERROR'
        result['error'] = f'密钥矩阵目录不存在: {km_dir}'
        return result

    try:
        positions_per_client = {cid: load_positions(km_dir, cid)
                                for cid in range(client_num)}
        enc_flat = load_encoder_flat(encoder_path)
    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)
        return result

    enc_offsets = encoder_offsets(positions_per_client)

    # 取模型中的水印区取值
    if meta['watermark_only']:
        wm_values = meta.get('client_watermark_values', {})
        result['note'] = 'pkl 为 watermark_only 格式，直接使用 client_watermark_values'
    else:
        offset_map = build_offset_map(state)
        wm_values = collect_wm_values_from_state(state, offset_map, positions_per_client)

    per_client = {}
    all_cur, all_auth = [], []
    for cid in range(client_num):
        cur_vals = wm_values.get(cid, [])
        start = enc_offsets[cid]
        n = len(positions_per_client[cid])
        auth_vals = enc_flat[start:start + n].tolist() if start + n <= enc_flat.numel() else []

        m = compare(cur_vals, auth_vals, tol)
        m['encoder_offset'] = start
        per_client[cid] = m

        all_cur.extend(cur_vals)
        all_auth.extend(auth_vals[:len(cur_vals)])

    result['per_client'] = per_client
    result['overall'] = compare(all_cur, all_auth, tol)
    result['status'] = 'OK'

    if verbose:
        print(f'\n=== {os.path.basename(pkl_path)} ===')
        print(f"  模型: {model_name} / {meta.get('dataset')} / K={client_num} "
              f"| epochs={meta.get('epochs')} | watermark_only={meta['watermark_only']}")
        print(f"  {'client':>7} {'count':>7} {'cos_sim':>10} {'rel_dev':>10} "
              f"{'max_abs':>12} {'exact_ratio':>12}")
        for cid in range(client_num):
            m = per_client[cid]
            if 'error' in m:
                print(f"  {cid:>7} {m['count']:>7}   {m['error']}")
                continue
            print(f"  {cid:>7} {m['count']:>7} {m['cos_sim']:>10.6f} {m['rel_dev']:>10.6f} "
                  f"{m['max_abs_diff']:>12.6e} {m['exact_match_ratio']:>12.4f}")
        o = result['overall']
        if 'error' not in o:
            print(f"  {'ALL':>7} {o['count']:>7} {o['cos_sim']:>10.6f} {o['rel_dev']:>10.6f} "
                  f"{o['max_abs_diff']:>12.6e} {o['exact_match_ratio']:>12.4f}")
            verdict = ('一致（与初始编码器参数逐元素相同）'
                       if o['exact_match_ratio'] >= 1.0 - 1e-9
                       else ('接近一致' if o['cos_sim'] > 0.999 else '不一致'))
            print(f"  结论: {verdict}")

    return result


def collect_pkl_files(inputs):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, '**', '*.pkl'),
                                          recursive=True)))
        elif os.path.isfile(item):
            files.append(item)
    # 去重并保持顺序
    seen, uniq = set(), []
    for f in files:
        rf = os.path.abspath(f)
        if rf not in seen:
            seen.add(rf)
            uniq.append(f)
    return uniq


def main():
    ap = argparse.ArgumentParser(description='校验 pkl 水印区参数是否与初始编码器参数一致')
    ap.add_argument('--pkl', nargs='+', required=True,
                    help='pkl 文件或包含 pkl 的目录（可多个）')
    ap.add_argument('--key_matrix_root', default='save/key_matrix',
                    help='密钥矩阵根目录（默认 save/key_matrix）')
    ap.add_argument('--encoder_path', default='save/autoencoder/encoder.pth',
                    help='初始编码器权重路径')
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='判定「逐元素相同」的绝对误差阈值')
    ap.add_argument('--json', default=None, help='把完整结果写入 JSON 文件')
    args = ap.parse_args()

    files = collect_pkl_files(args.pkl)
    if not files:
        print('没有找到任何 pkl 文件')
        return

    print(f'共发现 {len(files)} 个 pkl 文件；tol={args.tol}')

    results = []
    for f in files:
        try:
            results.append(analyze_pkl(f, args.key_matrix_root, args.encoder_path, args.tol))
        except Exception as e:
            print(f'\n=== {os.path.basename(f)} ===\n  异常: {e}')
            results.append({'pkl': f, 'status': 'ERROR', 'error': str(e)})

    ok = [r for r in results if r.get('status') == 'OK' and 'cos_sim' in r.get('overall', {})]
    if ok:
        print('\n===== 汇总 =====')
        print(f"{'pkl (相对 save/)':<58} {'cos_sim':>10} {'rel_dev':>10} "
              f"{'exact':>8}  结论")
        for r in ok:
            o = r['overall']
            name = os.path.relpath(r['pkl']).replace('\\', '/')
            if name.startswith('save/'):
                name = name[5:]
            verdict = ('一致' if o['exact_match_ratio'] >= 1.0 - 1e-9
                       else ('接近一致' if o['cos_sim'] > 0.999 else '不一致'))
            print(f"{name[-58:]:<58} {o['cos_sim']:>10.6f} {o['rel_dev']:>10.6f} "
                  f"{o['exact_match_ratio']:>8.4f}  {verdict}")

    skipped = [r for r in results
               if r.get('status') == 'OK' and 'cos_sim' not in r.get('overall', {})]
    for r in skipped:
        print(f"  [跳过] {os.path.basename(r['pkl'])}: "
              f"{r['overall'].get('error', '数量不匹配')}")

    errs = [r for r in results if r.get('status') == 'ERROR']
    for r in errs:
        print(f"  [错误] {os.path.basename(r['pkl'])}: {r.get('error')}")

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n完整结果已写入: {args.json}')


if __name__ == '__main__':
    main()
