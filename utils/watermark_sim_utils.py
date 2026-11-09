#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印区域相似度（sim）计算工具

用途
----
在微调攻击 / 剪枝攻击实验中，除了 ΔPCC（基于重建自编码器的 MSE 变化）之外，
补充一个更直观的指标：**水印区域相似度 sim**。

定义
----
sim = cos( 当前模型水印区域参数 , 参考模型水印区域参数 )

其中「水印区域参数」= 按 client_id 顺序拼接各客户端水印位置 P_k 上的参数值，
与 `utils/watermark_reconstruction.py` 中重建自编码器时的拼接顺序完全一致，
因此 sim 越接近 1 说明水印区域越接近原始（未受攻击）状态。

- 微调攻击：参考模型 = 微调前的原始水印模型
- 剪枝攻击：参考模型 = 剪枝前的原始水印模型

剪枝时水印位置被置 0，sim 会显著下降，与直觉一致。

同时提供：
- `wm_sim`        : 全部水印拼接后的整体余弦相似度（主指标）
- `wm_sim_mean`   : 各客户端相似度的平均值（避免长客户端主导）
- `wm_sim_per_client`: 每个客户端的相似度
- `wm_rel_dev`    : 相对偏差 ||cur - ref|| / ||ref||（辅助观察水印漂移幅度）
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _to_float_tensor(values) -> torch.Tensor:
    """把 tensor / list / ndarray 统一转成 1-D float32 tensor（始终拷贝，避免与原参数共享内存）"""
    if isinstance(values, torch.Tensor):
        return values.detach().view(-1).to(torch.float32).cpu().clone()
    return torch.as_tensor(values, dtype=torch.float32).view(-1).cpu().clone()


def cosine_similarity(vec_a, vec_b) -> float:
    """
    计算两个向量的余弦相似度（与 main.py 中的 _cosine_similarity 逻辑一致）

    任一向量范数为 0 时返回 0.0（避免除零）。
    """
    a = _to_float_tensor(vec_a)
    b = _to_float_tensor(vec_b)

    min_len = min(a.numel(), b.numel())
    if min_len == 0:
        return 0.0

    a = a[:min_len]
    b = b[:min_len]

    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)

    if norm_a.item() == 0.0 or norm_b.item() == 0.0:
        return 0.0

    return float(torch.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# 水印区域提取
# ----------------------------------------------------------------------
def extract_watermark_vector(model_state: Dict[str, torch.Tensor],
                             key_manager,
                             client_ids: Optional[Iterable[int]] = None,
                             model=None) -> torch.Tensor:
    """
    从模型状态字典中提取「水印区域参数向量」

    按 client_id 升序拼接各客户端水印位置上的参数值，
    与 reconstruct_autoencoder_from_all_clients 的拼接顺序保持一致。

    Args:
        model_state: 模型状态字典
        key_manager: KeyMatrixManager 实例
        client_ids: 需要提取的客户端列表，None 表示全部客户端
        model: 模型对象（可选，用于保证参数顺序与密钥矩阵生成时一致）

    Returns:
        1-D float32 tensor；无可用水印时返回空 tensor
    """
    if client_ids is None:
        client_ids = key_manager.list_clients()

    chunks: List[torch.Tensor] = []
    for cid in sorted(client_ids):
        try:
            values = key_manager.extract_watermark(
                model_state, cid, check_pruning=True, model=model
            )
        except Exception:
            continue
        if values is None or len(values) == 0:
            continue
        chunks.append(_to_float_tensor(values))

    if not chunks:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(chunks)


def extract_watermark_vectors_per_client(model_state: Dict[str, torch.Tensor],
                                         key_manager,
                                         client_ids: Optional[Iterable[int]] = None,
                                         model=None) -> Dict[int, torch.Tensor]:
    """
    分别提取每个客户端的水印区域参数向量

    Returns:
        {client_id: 1-D float32 tensor}
    """
    if client_ids is None:
        client_ids = key_manager.list_clients()

    result: Dict[int, torch.Tensor] = {}
    for cid in sorted(client_ids):
        try:
            values = key_manager.extract_watermark(
                model_state, cid, check_pruning=True, model=model
            )
        except Exception:
            continue
        if values is None or len(values) == 0:
            continue
        result[cid] = _to_float_tensor(values)

    return result


# ----------------------------------------------------------------------
# 相似度计算
# ----------------------------------------------------------------------
def compute_watermark_similarity(current_state: Dict[str, torch.Tensor],
                                 reference_state: Dict[str, torch.Tensor],
                                 key_manager,
                                 model=None) -> Dict:
    """
    计算当前模型水印区域相对于参考模型水印区域的相似度

    Args:
        current_state: 当前（受攻击）模型状态字典
        reference_state: 参考（原始水印）模型状态字典
        key_manager: KeyMatrixManager 实例
        model: 模型对象（可选，用于保证参数顺序一致）

    Returns:
        dict:
            wm_sim          : float，整体余弦相似度（主指标）
            wm_sim_mean     : float，各客户端相似度平均值
            wm_sim_min      : float，各客户端相似度最小值
            wm_sim_per_client: {client_id: float}
            wm_rel_dev      : float，相对偏差 ||cur-ref|| / ||ref||
            wm_nonzero_ratio: float，当前水印区域中非零元素占比（剪枝破坏的直接体现）
    """
    empty_result = {
        'wm_sim': None,
        'wm_sim_mean': None,
        'wm_sim_min': None,
        'wm_sim_per_client': {},
        'wm_sim_per_client_detail': {},
        'wm_rel_dev': None,
        'wm_nonzero_ratio': None,
    }

    try:
        cur_vec = extract_watermark_vector(current_state, key_manager, model=model)
        ref_vec = extract_watermark_vector(reference_state, key_manager, model=model)

        if cur_vec.numel() == 0 or ref_vec.numel() == 0:
            return empty_result

        overall_sim = cosine_similarity(cur_vec, ref_vec)

        # 相对偏差
        min_len = min(cur_vec.numel(), ref_vec.numel())
        cur_v = cur_vec[:min_len]
        ref_v = ref_vec[:min_len]
        ref_norm = torch.linalg.vector_norm(ref_v).item()
        if ref_norm > 0:
            rel_dev = float(torch.linalg.vector_norm(cur_v - ref_v).item() / ref_norm)
        else:
            rel_dev = None

        # 非零占比（剪枝会直接体现）
        nonzero_ratio = float((cur_v != 0).to(torch.float32).mean().item())

        # 每客户端相似度
        cur_per_client = extract_watermark_vectors_per_client(current_state, key_manager, model=model)
        ref_per_client = extract_watermark_vectors_per_client(reference_state, key_manager, model=model)

        per_client: Dict[int, float] = {}
        per_client_detail: Dict[int, Dict] = {}
        for cid, ref_c in sorted(ref_per_client.items()):
            cur_c = cur_per_client.get(cid)
            if cur_c is None:
                per_client[cid] = 0.0
                per_client_detail[cid] = {
                    'sim': 0.0,
                    'rel_dev': None,
                    'nonzero_ratio': 0.0,
                    'wm_size': int(ref_c.numel()),
                }
                continue

            c_sim = cosine_similarity(cur_c, ref_c)
            per_client[cid] = c_sim

            # 该客户端水印区域的相对偏差
            c_ref_norm = torch.linalg.vector_norm(ref_c).item()
            c_rel_dev = None
            if c_ref_norm > 0:
                c_rel_dev = float(
                    torch.linalg.vector_norm(cur_c - ref_c).item() / c_ref_norm
                )

            # 该客户端水印区域中非零元素占比
            c_nonzero = float((cur_c != 0).to(torch.float32).mean().item())

            per_client_detail[cid] = {
                'sim': c_sim,
                'rel_dev': c_rel_dev,
                'nonzero_ratio': c_nonzero,
                'wm_size': int(ref_c.numel()),
            }

        if per_client:
            sim_values = list(per_client.values())
            sim_mean = float(sum(sim_values) / len(sim_values))
            sim_min = float(min(sim_values))
        else:
            sim_mean = None
            sim_min = None

        return {
            'wm_sim': overall_sim,
            'wm_sim_mean': sim_mean,
            'wm_sim_min': sim_min,
            'wm_sim_per_client': per_client,
            'wm_sim_per_client_detail': per_client_detail,
            'wm_rel_dev': rel_dev,
            'wm_nonzero_ratio': nonzero_ratio,
        }
    except Exception as e:
        print(f"❌ 水印相似度计算失败: {e}")
        return empty_result


def format_watermark_similarity(sim_result: Optional[Dict]) -> Dict:
    """
    把相似度结果整理成可直接写入结果字典的标准格式

    每个客户端的相似度以 'wm_sim_c{cid}' 的形式平铺，
    便于直接写入 CSV 的独立列。
    """
    if sim_result is None:
        formatted = {
            'wm_sim': None,
            'wm_sim_mean': None,
            'wm_sim_min': None,
            'wm_rel_dev': None,
            'wm_nonzero_ratio': None,
        }
        return formatted

    formatted = {
        'wm_sim': sim_result.get('wm_sim'),
        'wm_sim_mean': sim_result.get('wm_sim_mean'),
        'wm_sim_min': sim_result.get('wm_sim_min'),
        'wm_rel_dev': sim_result.get('wm_rel_dev'),
        'wm_nonzero_ratio': sim_result.get('wm_nonzero_ratio'),
    }

    # 各客户端相似度平铺为独立列
    per_client = sim_result.get('wm_sim_per_client') or {}
    for cid, sim in sorted(per_client.items()):
        formatted[f'wm_sim_c{cid}'] = sim

    return formatted


def print_watermark_similarity(sim_result: Optional[Dict], prefix: str = "",
                               show_per_client: bool = True):
    """
    打印水印相似度摘要

    Args:
        sim_result: 相似度结果
        prefix: 输出前缀
        show_per_client: 是否打印各客户端明细
    """
    if sim_result is None or sim_result.get('wm_sim') is None:
        print(f"{prefix}水印相似度 sim: N/A")
        return

    print(f"{prefix}水印相似度 sim: {sim_result['wm_sim']:.6f} | "
          f"各客户端均值: {sim_result['wm_sim_mean']:.6f} | "
          f"最小值: {sim_result['wm_sim_min']:.6f} | "
          f"相对偏差: {sim_result['wm_rel_dev']:.6f} | "
          f"非零占比: {sim_result['wm_nonzero_ratio']:.2%}")

    if not show_per_client:
        return

    detail = sim_result.get('wm_sim_per_client_detail') or {}
    per_client = sim_result.get('wm_sim_per_client') or {}
    if not per_client:
        return

    print(f"{prefix}  各客户端水印区域相似度:")
    for cid in sorted(per_client):
        d = detail.get(cid, {})
        rel_dev = d.get('rel_dev')
        rel_dev_str = f"{rel_dev:.6f}" if rel_dev is not None else "N/A"
        nonzero = d.get('nonzero_ratio')
        nonzero_str = f"{nonzero:.2%}" if nonzero is not None else "N/A"
        wm_size = d.get('wm_size', '?')
        print(f"{prefix}    客户端 {cid} (水印长度 {wm_size}): "
              f"sim={per_client[cid]:.6f} | 相对偏差={rel_dev_str} | 非零占比={nonzero_str}")
