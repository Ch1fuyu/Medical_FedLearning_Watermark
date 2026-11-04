"""
抗合谋归因演示
==============

两种模式：

    python demo_collusion_attribution.py            # toy 模式（默认，手算版，看原理）
    python demo_collusion_attribution.py --mode real   # real 模式（真实 Singer 差集，5 客户端）

复现 utils/collusion_resistant.py 的完整链路：

    Step 0  构造见证点关联矩阵 A（BIBD / Singer 差集）
    Step 1  服务器为每个客户端生成定制模型（共享点写同一权威值）
    Step 2  K 个坏客户端把模型参数平均 -> 泄漏模型 L
    Step 3  服务器穷举所有候选子集，找残差最小的组合 = 合谋者集合
    Step 4  残差随规模呈 V 形 -> 自动推断合谋规模 K

归因核心调用的是项目真实代码：
    utils.collusion_resistant.decode_colluders
    utils.collusion_resistant.estimate_collusion_size

环境注意：
    本机默认 `python`（anaconda base, numpy 1.21.5）缺少 mkl_rt.dll，
    任何 `@` 矩阵乘法都会直接退出进程。请用：
        & "D:\\Environment\\anaconda\\envs\\Med_FL_WM\\python.exe" demo_collusion_attribution.py
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.collusion_resistant import (
    SingerParams,
    WitnessDesign,
    build_witness_design,
    decode_colluders,
    estimate_collusion_size,
    is_difference_set,
    singer_difference_set_generic,
)


# ====================================================================
# 通用工具
# ====================================================================
def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_matrix(title, rows, names, row_names=None):
    """打印一个数值矩阵。rows = [(label, array), ...]"""
    print(f"\n{title}")
    print("        " + "".join(f"{n:>9}" for n in names))
    print("        " + "-" * (9 * len(names)))
    for i, (label, vec) in enumerate(rows):
        cells = "".join(f"{v:>9.4f}" for v in vec)
        print(f"  {label:<6}{cells}")


# ====================================================================
# toy 模式：4 客户端 / 3 合谋者，手工设计（每点给"某一对客户端"共享）
# ====================================================================
TOY_CLIENTS = 4
TOY_POINTS = 6
TOY_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TOY_AUTHORITY = 0.9
TOY_ORIGINAL = 0.0


def build_toy_design():
    """构造玩具关联矩阵：A[p, c] = 1 表示客户端 c 在点 p 上写入权威值。

    规则：每个见证点分给"某一对客户端"，保证任意两客户端交集恰好 = 1
    （这正是 (v, k, 1) BIBD 的性质，只是规模小到能手画）。
    """
    n = TOY_CLIENTS
    A = np.zeros((TOY_POINTS, n), dtype=np.float64)
    for p, (c1, c2) in enumerate(TOY_PAIRS):
        A[p, c1] = 1.0
        A[p, c2] = 1.0

    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            inter = int(np.sum(A[:, c1] * A[:, c2]))
            assert inter <= 1, f"c{c1} 与 c{c2} 交集={inter} > 1"

    singer = SingerParams(q=2, p=2, a=1, v=TOY_POINTS, k=2,
                          D=tuple(d for d, _ in TOY_PAIRS),
                          poly=(1, 1, 0, 1))
    return WitnessDesign(n_clients=n, K=3, v=TOY_POINTS, k=2, G=1,
                          A=A, singer=singer)


def build_toy_models(design):
    """每个客户端一个标量参数 'witness'，值即各见证点上的取值。"""
    models = {}
    for c in range(design.n_clients):
        vals = np.array([TOY_AUTHORITY if design.A[p, c] > 0 else TOY_ORIGINAL
                         for p in range(design.v)], dtype=np.float64)
        models[c] = {"witness": vals}
    return models


# ====================================================================
# real 模式：真实 Singer 差集设计（5 客户端 / 最多 3 合谋）
# ====================================================================
REAL_CLIENTS = 5
REAL_K = 3
ENCODER_PARAMS = 9472
# 见证点权威值：随点变化，模拟真实 encoder 参数向量的不同片段
REAL_AUTHORITY_BASE = 0.50
REAL_AUTHORITY_STEP = 0.05


def build_real_design():
    """走项目真实路径：build_witness_design -> Singer 差集 -> 关联矩阵。"""
    return build_witness_design(
        n_clients=REAL_CLIENTS,
        K=REAL_K,
        encoder_params=ENCODER_PARAMS,
        positions=None,
        verbose=False,
    )


def build_real_models(design):
    """真实布局里每个见证点占 G 个模型位置。

    这里统一用一个扁平参数 'wm' 承载 v*G 个位置（真实代码里分布在多个 conv 层）。
    客户端 c 的定制模型：
        - 见证点 p 被 c 标记 -> p 的 G 个位置全写 e_p
        - 未被标记            -> 保持全局模型原值 w_t
    """
    v, G = design.v, design.G
    n_pos = v * G

    # 每个见证点的权威值 e_p（取自 encoder 参数向量的不同片段）
    point_auth = np.array(
        [REAL_AUTHORITY_BASE + REAL_AUTHORITY_STEP * p for p in range(v)],
        dtype=np.float64)

    models = {}
    for c in range(design.n_clients):
        vals = np.zeros(n_pos, dtype=np.float64)
        for p in range(v):
            if design.A[p, c] > 0:
                vals[p * G:(p + 1) * G] = point_auth[p]
            else:
                vals[p * G:(p + 1) * G] = 0.0     # w_t 原值
        models[c] = {"wm": vals}
    return models, point_auth


def point_values_of(vec, design):
    """从扁平向量还原每个见证点的取值（取该点组的第一个位置）。"""
    G = design.G
    return np.array([vec[p * G] for p in range(design.v)], dtype=np.float64)


# ====================================================================
# 主流程
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["toy", "real"], default="toy")
    args = ap.parse_args()

    if args.mode == "toy":
        design = build_toy_design()
        models = build_toy_models(design)
        colluders = [0, 1, 2]
        point_auth = np.array([TOY_AUTHORITY] * design.v)
        feature_key = "witness"
        point_names = [f"p{i}" for i in range(design.v)]
        position_mask = [(feature_key, p) for p in range(design.v)]
    else:
        design = build_real_design()
        models, point_auth = build_real_models(design)
        colluders = [0, 1, 3]
        feature_key = "wm"
        point_names = [f"p{i}" for i in range(design.v)]
        position_mask = [(feature_key, p * design.G) for p in range(design.v)]

    n_clients = design.n_clients
    A = design.A

    # ----------------------------------------------------------------
    hr(f"Step 0  见证点关联矩阵 A  (v={design.v} x n={n_clients}, k={design.k})")

    if args.mode == "real":
        print_singer_derivation(design)

    print("\n  关联矩阵 A  (行=见证点, 列=客户端, 1=该客户端在此点写权威值):")
    print("        " + "".join(f"{f'c{c}':>6}" for c in range(n_clients)))
    for p in range(design.v):
        cells = "".join(f"{int(A[p, c]):>6}" for c in range(n_clients))
        print(f"    {point_names[p]:<5}{cells}")

    rw = A.sum(axis=1).astype(int)
    cw = A.sum(axis=0).astype(int)
    print(f"\n  行权(每点被几个客户端标记) = {rw.tolist()}")
    print(f"  列权(每客户端标记几个点) = {cw.tolist()}")
    print(f"  全 0 行(无人使用的点)     = {int((rw == 0).sum())} 个")
    print(f"  关联矩阵 rank             = {np.linalg.matrix_rank(A)}/{n_clients}")

    print("\n  两两交集校验 (BIBD 性质, 必须 <= 1):")
    ok = True
    for c1 in range(n_clients):
        for c2 in range(c1 + 1, n_clients):
            inter = int(np.sum(A[:, c1] * A[:, c2]))
            ok &= inter <= 1
            print(f"    c{c1} ∩ c{c2} = {inter}", end="")
        print()
    print(f"  -> {'全部满足' if ok else '存在违规!'}")

    # ----------------------------------------------------------------
    hr("Step 1  服务器为每个客户端生成定制模型")

    if args.mode == "toy":
        print_matrix("  各客户端收到的模型 M_c：",
                     [(f"M{c}", models[c][feature_key]) for c in range(n_clients)],
                     point_names)
    else:
        # 真实模式下每个点占 G 个位置，打印时按点折叠显示
        rows = [(f"M{c}", point_values_of(models[c][feature_key], design))
                for c in range(n_clients)]
        print_matrix(f"  各客户端收到的模型 M_c（每个见证点取其 {design.G} "
                     f"个位置中的第一个）:", rows, point_names)

    print("\n  每个见证点的权威值 e_p:")
    print("    " + "  ".join(f"{point_names[p]}={point_auth[p]:.3f}"
                             for p in range(design.v)))

    # ----------------------------------------------------------------
    hr(f"Step 2  K={len(colluders)} 个坏客户端合谋，发布参数平均值")

    leaked_vec = models[colluders[0]][feature_key].copy()
    for c in colluders[1:]:
        leaked_vec = leaked_vec + models[c][feature_key]
    leaked_vec = leaked_vec / len(colluders)

    if args.mode == "toy":
        print_matrix("  泄漏模型 L = mean(" +
                     ", ".join(f"M{c}" for c in colluders) + ")：",
                     [("L", leaked_vec)], point_names)
    else:
        print_matrix("  泄漏模型 L = mean(" +
                     ", ".join(f"M{c}" for c in colluders) + ")：",
                     [("L", point_values_of(leaked_vec, design))], point_names)

    print("\n  解读（每个点上被稀释的程度暴露了谁参与了）：")
    for p in range(design.v):
        marked = [c for c in colluders if A[p, c] > 0]
        val = point_values_of(leaked_vec, design)[p]
        if marked:
            note = (f"c{marked} 参与 -> {len(marked)}/{len(colluders)} "
                    f"写了权威值，被稀释")
        else:
            note = "无合谋者参与 -> 完全稀释成原值"
        print(f"    {point_names[p]}: L={val:.4f}   ({note})")

    # ----------------------------------------------------------------
    hr("Step 3  服务器穷举所有候选子集，找残差最小的组合")

    print("  残差 = |L - mean(S)|^2（只统计见证点位置）:")
    rows = []
    for r in range(1, n_clients + 1):
        for combo in combinations(range(n_clients), r):
            avg = np.mean([models[c][feature_key] for c in combo], axis=0)
            if args.mode == "real":
                # real 模式下按点折叠后再算残差（与解码特征维度一致）
                avg_pts = point_values_of(avg, design)
                leak_pts = point_values_of(leaked_vec, design)
                err = float(np.sum((leak_pts - avg_pts) ** 2))
            else:
                err = float(np.sum((leaked_vec - avg) ** 2))
            rows.append((combo, err))
    rows.sort(key=lambda x: x[1])
    for combo, err in rows[:10]:
        mark = "  <== 命中" if err < 1e-12 else ""
        print(f"    {str(list(combo)):<14} 残差 = {err:.6f}{mark}")
    if len(rows) > 10:
        print(f"    ...（共 {len(rows)} 种组合）")

    # ----------------------------------------------------------------
    hr("Step 4  调用项目真实归因代码")

    client_models = {c: {feature_key: models[c][feature_key].copy()}
                     for c in range(n_clients)}
    leaked_model = {feature_key: leaked_vec}

    found, diag = decode_colluders(
        leaked_model, client_models, design,
        max_colluders=n_clients,
        position_mask=position_mask,
    )
    est_k, est_err = estimate_collusion_size(
        leaked_model, client_models, design,
        max_colluders=n_clients,
        position_mask=position_mask,
    )

    print(f"\n  decode_colluders        -> 合谋者 = {found}")
    print(f"  best_error              = {diag['best_error']:.6e}")
    print(f"  margin (领先第二名)     = {diag['margin']:.6f}")
    print(f"  候选组合数              = {diag['n_candidates']}")
    print(f"  estimate_collusion_size -> K = {est_k} (残差 {est_err:.6e})")

    hr("结果")
    ok_set = set(found) == set(colluders)
    ok_size = est_k == len(colluders)
    print(f"  实际合谋者 : {colluders}")
    print(f"  检出合谋者 : {found}       -> {'✓ 正确' if ok_set else '✗ 错误'}")
    print(f"  估计规模   : {est_k} / 实际 {len(colluders)}"
          f"   -> {'✓ 正确' if ok_size else '✗ 错误'}")

    # ----------------------------------------------------------------
    hr("附加  噪声攻击下的表现")
    rng = np.random.default_rng(0)
    for noise in (0.01, 0.05, 0.1):
        noisy = leaked_vec + noise * rng.normal(size=leaked_vec.shape)
        f2, d2 = decode_colluders({feature_key: noisy}, client_models, design,
                                  max_colluders=n_clients,
                                  position_mask=position_mask)
        k2, _ = estimate_collusion_size({feature_key: noisy}, client_models,
                                        design, max_colluders=n_clients,
                                        position_mask=position_mask)
        print(f"  噪声 std={noise:<6} 检出={str(f2):<14} K={k2}  "
              f"{'✓' if set(f2) == set(colluders) else '✗'}")


def print_singer_derivation(design):
    """打印 Singer 差集的完整推导（real 模式）。"""
    sp = design.singer

    print("  [候选 q 对比]")
    print(f"    {'q':>3} {'v_raw':>6} {'v_used':>7} {'k':>3} {'rank':>7} "
          f"{'>=2标记的点':>12} {'行权方差':>10}")
    from utils.collusion_resistant import build_incidence_matrix
    for q in (2, 3, 4):
        try:
            s2 = singer_difference_set_generic(q)
        except ValueError:
            continue
        A2 = build_incidence_matrix(s2.v, design.n_clients, s2.k, s2.D)
        if np.linalg.matrix_rank(A2) < design.n_clients:
            continue
        cov2 = np.where(A2.sum(axis=1) > 0)[0]
        A2c = A2[cov2, :]
        rw2 = A2c.sum(axis=1)
        print(f"    {q:>3} {s2.v:>6} {len(cov2):>7} {s2.k:>3} "
              f"{np.linalg.matrix_rank(A2c):>4}/{design.n_clients} "
              f"{int((rw2 >= 2).sum()):>12} {rw2.var():>10.4f}")
    print(f"    -> 选中 q={sp.q}  (打分: >=2标记的点最多, 其次行权方差最小)")
    print(f"    v_raw = 差集原始地面集 q^2+q+1; v_used = 剔除无人标记的点后实际使用数")

    print("\n  [Singer 差集构造]")
    print(f"    GF({sp.p}^{3 * sp.a}), 本原多项式(低次在前) = {list(sp.poly)}")
    print(f"    v_raw = q^2+q+1 = {sp.q}^2+{sp.q}+1 = {sp.v},  k = q+1 = {sp.k}")
    print(f"    D = {{ i : Tr(alpha^i) = 0 }} = {list(sp.D)}")
    print(f"    差集校验 is_difference_set = {is_difference_set(sp.v, sp.k, sp.D)}")

    # 手工统计有序非零差
    diffs = {}
    for a in sp.D:
        for b in sp.D:
            if a != b:
                diffs[(a - b) % sp.v] = diffs.get((a - b) % sp.v, 0) + 1
    print(f"    非零有序差种类 = {len(diffs)} (应 = v-1 = {sp.v - 1}), "
          f"每种出现次数 = {sorted(set(diffs.values()))}")

    print("\n  [列 = 差集的平移 D+c mod v_raw，再压缩掉无人标记的点]")
    used = sorted({(d + c) % sp.v for c in range(design.n_clients) for d in sp.D})
    remap = {raw: new for new, raw in enumerate(used)}
    for c in range(design.n_clients):
        raw_marks = sorted((d + c) % sp.v for d in sp.D)
        new_marks = [remap[m] for m in raw_marks]
        print(f"    c{c}: 原始 {raw_marks}  ->  压缩后 {new_marks}")

    print(f"\n  [地面集压缩]")
    print(f"    v_raw = {sp.v} -> v_used = {design.v} "
          f"(剔除 {sp.v - design.v} 个无人标记的点)")
    print(f"    BIBD 性质保持: 任意两列交集 <= 1, rank = "
          f"{np.linalg.matrix_rank(design.A)}/{design.n_clients}")

    print(f"\n  [见证点位置预算]")
    G = design.G
    print(f"    G = encoder_params // (n_clients * v_used) = "
          f"{ENCODER_PARAMS} // ({design.n_clients} * {design.v}) = {G}")
    print(f"    见证点总占用 = v_used * G = {design.v} * {G} = "
          f"{design.v * G} 个模型位置")


if __name__ == "__main__":
    main()
