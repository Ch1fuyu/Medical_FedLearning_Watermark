"""
BIBD-ACC 码本生成脚本的回归测试。

覆盖：
  * 参数求解（精确解存在/不存在）
  * Singer 差集构造正确性
  * 关联矩阵三项验证（列权 = k、两两内积 <= 1、列数 >= n）
  * K-弹性 AND 唯一性（严格枚举）
  * 浮点映射正确性
  * 合谋归因端到端

运行： python tests/test_bibd_acc_codebook.py
"""

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bibd_acc_codebook import (  # noqa: E402
    build_cyclic_bibd_matrix,
    find_difference_set,
    generate_codebook,
    generate_exact_codebook,
    generate_hybrid_codebook,
    generate_signature_rows,
    is_difference_set,
    map_to_float,
    recommend_config,
    simulate_collusion,
    singer_difference_set,
    solve_bibd_params,
    validate_bibd_matrix,
    verify_and_uniqueness,
)


def test_solve_params():
    print("== test_solve_params ==")
    # 精确解存在当且仅当 n = K(K+1)+1
    for K in (1, 2, 3, 4, 5):
        n = K * (K + 1) + 1
        p = solve_bibd_params(n, K)
        assert p is not None, f"n={n} K={K} 应有精确解"
        v, k = p
        assert k == K + 1
        assert v == n, (v, n)
        assert v * v - v == n * (k * k - k)
        print(f"  n={n:3d} K={K} -> v={v} k={k}  OK")
    # 非精确解
    assert solve_bibd_params(20, 3) is None
    assert solve_bibd_params(10, 3) is None
    print("  非精确解情形 OK")


def test_singer():
    print("== test_singer ==")
    for q in (2, 3, 5, 7):
        res = singer_difference_set(q)
        assert res is not None, f"q={q} Singer 构造失败"
        v, k, D = res
        assert v == q * q + q + 1
        assert k == q + 1
        assert len(D) == k
        assert is_difference_set(v, k, D), f"q={q} D={D} 不是差集"
        # 有序差恰好覆盖 1..v-1
        diffs = set()
        for a, b in combinations(D, 2):
            diffs.add((a - b) % v)
            diffs.add((b - a) % v)
        assert diffs == set(range(1, v)), diffs
        print(f"  q={q}: v={v} k={k} D={D} 差集校验 OK")


def test_bibd_matrix():
    print("== test_bibd_matrix ==")
    for q in (2, 3, 5):
        v, k, D = singer_difference_set(q)
        M = build_cyclic_bibd_matrix(v, v, k, D)
        assert M.shape == (v, v)
        ok, rep = validate_bibd_matrix(M, k, v)
        assert ok, rep
        assert rep["max_pairwise_inner_product"] <= 1
        assert rep["col_weight_all_k"]
        print(f"  q={q}: 关联矩阵 {M.shape} 验证 OK "
              f"(最大内积={rep['max_pairwise_inner_product']})")


def test_exact_and_uniqueness():
    print("== test_exact_and_uniqueness ==")
    for n, K in [(5, 1), (6, 2), (7, 3), (8, 3), (9, 4)]:
        cb, info = generate_exact_codebook(n, K, verbose=False)
        ok, conflicts = verify_and_uniqueness(cb, K, verbose=False)
        assert ok, f"n={n} K={K} exact 未通过 AND 唯一性: {conflicts}"
        assert cb.shape[1] == n
        print(f"  n={n} K={K}: v={cb.shape[0]} AND 唯一性 OK")


def test_hybrid_and_uniqueness():
    print("== test_hybrid_and_uniqueness ==")
    for n, K in [(5, 1), (7, 2), (8, 3), (10, 3), (13, 3), (10, 4)]:
        cb, info = generate_hybrid_codebook(n, K, verbose=False)
        ok, _ = verify_and_uniqueness(cb, K, verbose=False)
        assert ok, f"n={n} K={K} hybrid 未通过 AND 唯一性"
        print(f"  n={n} K={K}: v={cb.shape[0]} (骨架 {info['v_bibd']} "
              f"+ 签名 {info['v_signature']}) AND 唯一性 OK")


def test_signature_rows_proof():
    print("== test_signature_rows_proof ==")
    from math import comb
    n, K = 7, 3
    sig, _ = generate_signature_rows(n, K)
    assert sig.shape == (comb(n, K), n), sig.shape
    assert sig.sum() == comb(n, K) * K
    print(f"  签名行 shape={sig.shape} OK")


def test_float_mapping():
    print("== test_float_mapping ==")
    cb, _ = generate_exact_codebook(6, 2, verbose=False)
    fc, mu, delta = map_to_float(cb, mu=0.5, delta=0.02)
    assert fc.shape == cb.shape
    assert np.allclose(fc[cb == 1], 0.52)
    assert np.allclose(fc[cb == 0], 0.48)
    # 数组形式
    mu_v = np.linspace(0.0, 1.0, cb.shape[0])
    d_v = np.full(cb.shape[0], 0.1)
    fc2, _, _ = map_to_float(cb, mu=mu_v, delta=d_v)
    assert fc2.shape == cb.shape
    print("  标量/数组 mu,delta 映射 OK")


def test_collusion_attribution():
    print("== test_collusion_attribution ==")
    n, K = 9, 3
    fc, info = generate_codebook(n, K, mode="hybrid", verbose=False)
    for colluders in [(0,), (0, 1), (0, 3, 5), (2, 7)]:
        success, cands = simulate_collusion(fc, info, colluders, verbose=False)
        assert success, f"合谋 {colluders} 归因失败, 候选={cands}"
    print("  各种合谋规模归因 OK")


def test_bibd_mode_raises_for_k_ge_2():
    print("== test_bibd_mode_raises_for_k_ge_2 ==")
    # BIBD 纯模式在 K>=2 时必须报错（其固有性质）
    raised = False
    try:
        generate_codebook(13, 3, mode="bibd", verbose=False)
    except RuntimeError:
        raised = True
    assert raised, "bibd 模式 K=3 应抛出 RuntimeError"
    print("  bibd 模式 K=3 如实报错 OK")


def test_random_mode():
    print("== test_random_mode ==")
    for n, K in [(5, 3), (10, 3), (20, 3)]:
        fc, info = generate_codebook(n, K, mode="random", verbose=False,
                                     max_rows=200, seed=7)
        assert info["construction_used"] == "random"
        assert fc.shape[0] <= 200, fc.shape
        assert info["and_unique"] is True
        print(f"  n={n} K={K}: v={fc.shape[0]} (<=200) AND 唯一性 OK")


def test_budget_aware_auto():
    print("== test_budget_aware_auto ==")
    # 预算充足 -> hybrid；预算不足 -> random
    fc, info = generate_codebook(5, 3, mode="auto", verbose=False, max_rows=1894)
    assert info["construction_used"] == "hybrid", info["construction_used"]
    print(f"  n=5 budget=1894 -> {info['construction_used']} (v={fc.shape[0]})")

    fc, info = generate_codebook(20, 3, mode="auto", verbose=False, max_rows=473)
    assert info["construction_used"] == "random", info["construction_used"]
    print(f"  n=20 budget=473 -> {info['construction_used']} (v={fc.shape[0]})")


def test_recommended_configs():
    print("== test_recommended_configs ==")
    for n in (5, 10, 20):
        rec = recommend_config(n, K=3, encoder_params=9472)
        print(f"  n={n}: {rec}")
        assert rec["max_rows"] == 9472 // n
        fc, info = generate_codebook(n, rec["K"], mode="auto", verbose=False,
                                     max_rows=rec["max_rows"])
        assert fc.shape[0] <= rec["max_rows"]


def test_save_load():
    print("== test_save_load ==")
    import json
    import tempfile
    from utils.bibd_acc_codebook import save_codebook
    fc, info = generate_codebook(8, 2, mode="hybrid", verbose=False)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cb.npz")
        save_codebook(p, fc, info)
        with np.load(p, allow_pickle=False) as d:
            assert d["float_codebook"].shape == fc.shape
            meta = json.loads(str(d["meta"]))
            assert meta["K"] == 2
            assert meta["n_clients"] == 8
    print("  保存/加载 OK")


if __name__ == "__main__":
    test_solve_params()
    test_singer()
    test_bibd_matrix()
    test_exact_and_uniqueness()
    test_hybrid_and_uniqueness()
    test_float_mapping()
    test_collusion_attribution()
    test_bibd_mode_raises_for_k_ge_2()
    test_random_mode()
    test_budget_aware_auto()
    test_recommended_configs()
    test_save_load()
    print("\n全部测试通过 ✅")
