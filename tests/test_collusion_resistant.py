"""抗合谋水印模块（collusion_resistant.py）的回归测试。

覆盖：
  - Singer 差集构造（q=2/3/4）与差集校验
  - 关联矩阵性质（列权、行权、满秩、列内积 <= 1）
  - 见证点设计选型（v >= n，行权 >= 2 的点数）
  - 嵌入 -> 合谋 -> 解码的端到端闭环
  - 抗噪/抗漂移鲁棒性
  - 误报：随机模型不应被误判为合谋
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.collusion_resistant import (  # noqa: E402
    DEFAULT_ENCODER_PARAMS,
    build_incidence_matrix,
    build_witness_design,
    decode_colluders,
    embed_witness_watermark,
    estimate_collusion_size,
    is_difference_set,
    simulate_collusion,
    singer_difference_set_generic,
)


# ====================================================================
def test_singer_difference_set():
    print("== test_singer_difference_set ==")
    for q, expect_v in [(2, 7), (3, 13), (4, 21)]:
        sp = singer_difference_set_generic(q)
        assert sp.v == expect_v, f"q={q}: v={sp.v}, 期望 {expect_v}"
        assert sp.k == q + 1
        assert is_difference_set(sp.v, sp.k, sp.D), f"q={q} 差集校验失败"
        print(f"  q={q}: v={sp.v} k={sp.k} D={list(sp.D)} 差集校验 OK")


# ====================================================================
def test_incidence_matrix():
    print("== test_incidence_matrix ==")
    for q in (2, 3, 4):
        sp = singer_difference_set_generic(q)
        for n in (5, 10, 20):
            if n > sp.v:
                continue
            A = build_incidence_matrix(sp.v, n, sp.k, sp.D)
            col_w = A.sum(axis=0)
            assert np.all(col_w == sp.k), f"列权应恒为 k={sp.k}，实际 {set(col_w)}"
            # 任意两列内积 <= 1（BIBD 性质）
            G = A.T @ A
            off = G - np.diag(np.diag(G))
            assert off.max() <= 1, f"列内积应 <=1，实际 {off.max()}"
            assert np.linalg.matrix_rank(A) == n, "A 应满秩"
            print(f"  q={q} n={n}: 列权={sp.k} 最大列内积={int(off.max())} "
                  f"rank={np.linalg.matrix_rank(A)}/{n}  OK")


# ====================================================================
def test_witness_design():
    print("== test_witness_design ==")
    for n in (5, 10, 20):
        d = build_witness_design(n, K=3, verbose=False)
        assert d.v >= n, f"v={d.v} 应 >= n={n}"
        assert d.A.shape == (d.v, n)
        rw = d.row_weights
        n_ge2 = int((rw >= 2).sum())
        assert n_ge2 > 0, "应存在行权 >= 2 的点"
        total = d.v * d.G
        assert total <= DEFAULT_ENCODER_PARAMS, \
            f"位置池 {total} 超过水印总量 {DEFAULT_ENCODER_PARAMS}"
        print(f"  n={n}: q={d.singer.q} v={d.v} k={d.k} G={d.G} "
              f"位置池={total} 行权>=2的点={n_ge2}  OK")


# ====================================================================
def _make_scene(n, K, v_pool=400, seed=0, drift=0.3, noise=0.2):
    """构造一个可控的仿真场景。

    返回 (design, client_models, encoder_values, rng)
    """
    rng = np.random.default_rng(seed)
    d = build_witness_design(n, K=K, encoder_params=v_pool, verbose=False)
    # 位置池：单个假参数名 + 局部下标
    d.positions = [("conv.weight", i) for i in range(d.v * d.G)]

    # 权威值 e_p
    E = rng.normal(0, 1, size=(d.v, d.G))
    # 全局模型 w_t
    WT = rng.normal(0, 1, size=(d.v, d.G))

    client_models = {}
    for c in range(n):
        marked = d.client_points(c)
        m = WT.copy()
        m[marked] = E[marked]
        # 本地训练漂移
        m = m + drift * rng.normal(size=m.shape)
        client_models[c] = {"conv.weight": m}
    return d, client_models, E, rng


def test_end_to_end_attribution():
    print("== test_end_to_end_attribution ==")
    for n in (5, 10, 20):
        for K in (1, 2, 3):
            if K > n:
                continue
            d, cmodels, E, rng = _make_scene(n, K, seed=n * 10 + K)
            for trial in range(5):
                S = sorted(rng.choice(n, K, replace=False).tolist())
                leaked = simulate_collusion(cmodels, S, d, noise_scale=0.05, rng=rng)
                found, diag = decode_colluders(leaked, cmodels, d,
                                              max_colluders=K, verbose=False)
                assert set(found) == set(S), \
                    f"n={n} K={K} 真值={S} 解出={found}"
            print(f"  n={n} K={K}: 5/5 次精确归因 OK (v={d.v}, G={d.G})")


# ====================================================================
def test_robustness_noise_drift():
    print("== test_robustness_noise_drift ==")
    d, cmodels, E, rng = _make_scene(10, 2, seed=99)
    for drift, noise in [(0.3, 0.1), (0.5, 0.3), (1.0, 0.5)]:
        # 重新生成带指定漂移的场景
        d2, cm2, _, rng2 = _make_scene(10, 2, seed=99, drift=drift)
        ok = 0
        T = 10
        for _ in range(T):
            S = sorted(rng2.choice(10, 2, replace=False).tolist())
            leaked = simulate_collusion(cm2, S, d2, noise_scale=noise, rng=rng2)
            found, _ = decode_colluders(leaked, cm2, d2, max_colluders=2)
            if set(found) == set(S):
                ok += 1
        assert ok == T, f"drift={drift} noise={noise}: 仅 {ok}/{T}"
        print(f"  drift={drift} noise={noise}: {ok}/{T} 精确归因 OK")


# ====================================================================
def test_no_false_positive():
    print("== test_no_false_positive ==")
    # 泄漏模型就是某个客户端自己的模型 -> 应识别出该客户端
    d, cmodels, E, rng = _make_scene(10, 3, seed=7)
    for c in range(10):
        leaked = {"conv.weight": np.asarray(cmodels[c]["conv.weight"]).copy()}
        found, _ = decode_colluders(leaked, cmodels, d, max_colluders=3)
        assert set(found) == {c}, f"应识别出 {c}，实际 {found}"
    print("  单客户端泄漏（K=1 情形）10/10 正确 OK")

    # 随机模型 -> 解码应给出某个结果但残差很大（不是真的合谋）
    rand_model = {"conv.weight": rng.normal(size=(d.v, d.G))}
    found, diag = decode_colluders(rand_model, cmodels, d, max_colluders=3)
    assert diag["best_error"] > 1.0, \
        f"随机模型残差应很大，实际 {diag['best_error']}"
    print(f"  随机模型残差 = {diag['best_error']:.3f}（远大于真实合谋） OK")


# ====================================================================
def test_decode_without_positions():
    print("== test_decode_without_positions ==")
    # design.positions 为空时，应回退到"按参数名拼全部数值"
    d, cmodels, E, rng = _make_scene(8, 2, seed=11)
    d.positions = []
    S = sorted(rng.choice(8, 2, replace=False).tolist())
    leaked = simulate_collusion(cmodels, S, d, noise_scale=0.05, rng=rng)
    found, _ = decode_colluders(leaked, cmodels, d, max_colluders=2)
    assert set(found) == set(S), f"真值={S} 解出={found}"
    print(f"  无位置掩码回退路径 OK (n=8 K=2 真值={S})")


# ====================================================================
def test_embed_function():
    print("== test_embed_function ==")
    d = build_witness_design(6, K=2, encoder_params=120, verbose=False)
    d.positions = [("conv.weight", i) for i in range(d.v * d.G)]
    rng = np.random.default_rng(3)
    E = rng.normal(0, 1, size=(d.v, d.G))
    WT = rng.normal(0, 1, size=(d.v, d.G))
    base = {"conv.weight": WT.copy()}

    out = embed_witness_watermark(base, d, client_id=2, encoder_values=E)
    got = out["conv.weight"]
    marked = d.client_points(2)
    unchanged = set(range(d.v)) - set(marked.tolist())
    # 标记点应写入 e_p
    assert np.allclose(got[marked], E[marked]), "标记点应写入权威值"
    # 未标记点应保持原值
    assert np.allclose(got[list(unchanged)], WT[list(unchanged)]), \
        "未标记点应保持原值"
    # 原字典不应被修改
    assert np.allclose(base["conv.weight"], WT), "入参不应被修改"
    print(f"  n=6 v={d.v} G={d.G}: 标记点={len(marked)} 写入正确、未修改入参 OK")


# ====================================================================
def test_estimate_collusion_size():
    print("== test_estimate_collusion_size ==")
    for n in (5, 10, 20):
        for true_K in (1, 2, 3):
            if true_K > n:
                continue
            d, cmodels, E, rng = _make_scene(n, true_K, seed=n * 31 + true_K)
            est_ok = attr_ok = 0
            T = 8
            for _ in range(T):
                S = sorted(rng.choice(n, true_K, replace=False).tolist())
                leaked = simulate_collusion(cmodels, S, d, noise_scale=0.1, rng=rng)
                est_k, _ = estimate_collusion_size(leaked, cmodels, d,
                                                   max_colluders=5)
                if est_k == true_K:
                    est_ok += 1
                found, _ = decode_colluders(leaked, cmodels, d,
                                           max_colluders=est_k)
                if set(found) == set(S):
                    attr_ok += 1
            assert est_ok == T, f"n={n} trueK={true_K} 规模推断仅 {est_ok}/{T}"
            assert attr_ok == T, f"n={n} trueK={true_K} 归因仅 {attr_ok}/{T}"
            print(f"  n={n:2d} trueK={true_K}: 规模推断 {est_ok}/{T}, "
                  f"归因 {attr_ok}/{T}  OK")


# ====================================================================
def test_realistic_budget():
    print("== test_realistic_budget ==")
    # 使用真实水印预算（9472），G 值较大
    for n in (5, 10, 20):
        d = build_witness_design(n, K=3, encoder_params=DEFAULT_ENCODER_PARAMS,
                                 verbose=False)
        assert d.v * d.G <= DEFAULT_ENCODER_PARAMS
        d.positions = [("conv.weight", i) for i in range(d.v * d.G)]
        rng = np.random.default_rng(n)
        E = rng.normal(0, 1, size=(d.v, d.G))
        WT = rng.normal(0, 1, size=(d.v, d.G))
        cm = {}
        for c in range(n):
            m = WT.copy()
            m[d.client_points(c)] = E[d.client_points(c)]
            cm[c] = {"conv.weight": m + 0.3 * rng.normal(size=m.shape)}
        for K in (1, 2, 3):
            if K > n:
                continue
            ok = 0
            T = 5
            for _ in range(T):
                S = sorted(rng.choice(n, K, replace=False).tolist())
                leaked = simulate_collusion(cm, S, d, noise_scale=0.2, rng=rng)
                found, _ = decode_colluders(leaked, cm, d, max_colluders=K)
                if set(found) == set(S):
                    ok += 1
            assert ok == T, f"n={n} K={K}: {ok}/{T}"
            print(f"  n={n:2d} K={K} (v={d.v}, G={d.G}): {ok}/{T} 精确归因 OK")


# ====================================================================
def main():
    test_singer_difference_set()
    test_incidence_matrix()
    test_witness_design()
    test_end_to_end_attribution()
    test_estimate_collusion_size()
    test_robustness_noise_drift()
    test_no_false_positive()
    test_decode_without_positions()
    test_embed_function()
    test_realistic_budget()
    print("\n全部测试通过 ✅")


if __name__ == "__main__":
    main()
