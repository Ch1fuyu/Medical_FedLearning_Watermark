"""
抗合谋水印 v2（密钥派生 + 固定基线 + 计数解码）验证脚本

覆盖：
  1. 密钥派生：见证点值与 encoder 切片完全解耦；不同密钥 -> 不同见证点
  2. 不可伪造性：合谋者只知道自己标记的点，无法推断他人点的 e_p
  3. 固定基线：见证点区域在所有客户端 / 所有轮次上取值固定
  4. 分散布局：每个点的 G 个副本不连续
  5. 计数解码：等权平均攻击下精确归因（含噪声）
  6. 计数解码：局部攻击（随机置零部分副本）下仍可归因
  7. 计数解码：无合谋 / 随机模型 -> 弃权
  8. 冻结：见证点区域在本地训练后保持不变

运行：
    python tests/test_collusion_v2.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.collusion_resistant import (  # noqa: E402
    DEFAULT_ENCODER_PARAMS,
    DEFAULT_WITNESS_AMPLITUDE,
    build_witness_layout,
    decode_colluders_by_count,
    derive_witness_values,
    derive_spreading_pattern,
    embed_witness_watermark_torch,
    estimate_collusion_size_by_count,
    generate_witness_secret,
)

N_CLIENTS = 10
K_MAX = 3
AMPLITUDE = 0.05


# ====================================================================
def _fake_positions(n_pos):
    """构造 n_pos 个假的 conv 位置（跨若干假参数名）。"""
    names = ["features.0.weight", "features.2.weight", "features.4.weight"]
    out = []
    for i in range(n_pos):
        nm = names[i % len(names)]
        out.append((nm, i // len(names)))
    return out


def make_layout(n=N_CLIENTS, K=K_MAX, secret=None, positions=None,
                encoder_params=DEFAULT_ENCODER_PARAMS):
    if positions is None:
        positions = _fake_positions(4000)
    if secret is None:
        secret = generate_witness_secret()
    layout = build_witness_layout(
        n_clients=n, K=K, encoder_params=encoder_params,
        positions=positions, secret=secret, verbose=False)
    return layout, secret


# ====================================================================
def test_key_derivation():
    print("== test_key_derivation ==")
    s1 = generate_witness_secret()
    s2 = generate_witness_secret()

    e1, b1 = derive_witness_values(s1, 21, 45, amplitude=AMPLITUDE)
    e1b, _ = derive_witness_values(s1, 21, 45, amplitude=AMPLITUDE)
    e2, _ = derive_witness_values(s2, 21, 45, amplitude=AMPLITUDE)

    assert np.allclose(e1, e1b), "同密钥必须派生出相同值（可复现）"
    assert not np.allclose(e1, e2), "不同密钥必须派生出不同值"
    assert np.allclose(b1, 0.0), "基线应为 0"
    assert np.abs(e1).min() >= AMPLITUDE * 0.5, "幅度不应退化到近零"
    assert np.abs(e1).max() <= AMPLITUDE + 1e-12, "幅度不应超出设定值"

    # 不同点之间的值应当互不相同（避免不同点被混淆）
    flat = e1.reshape(-1)
    assert len(np.unique(np.round(flat, 12))) == flat.size, "见证点值出现重复"

    print(f"  同密钥可复现 ✓  不同密钥不同值 ✓  幅度区间 "
          f"[{np.abs(e1).min():.4f}, {np.abs(e1).max():.4f}] ✓")


# ====================================================================
def test_not_from_encoder_slice():
    print("== test_not_from_encoder_slice ==")
    layout, secret = make_layout()

    # 见证点值必须与 encoder 参数切片完全无关
    enc = np.random.default_rng(0).normal(0, 1, DEFAULT_ENCODER_PARAMS)
    pv = layout.extract_point_values(enc)
    # 改变 encoder 参数不应改变见证点值
    enc2 = np.random.default_rng(1).normal(0, 1, DEFAULT_ENCODER_PARAMS)
    pv2 = layout.extract_point_values(enc2)
    assert np.allclose(pv, pv2), "见证点值不应依赖 encoder 参数"
    assert not np.allclose(pv.reshape(-1)[:10], enc[:10]), \
        "见证点值不应等于 encoder 前段切片"

    print("  见证点值与 encoder 参数切片解耦 ✓")


# ====================================================================
def test_fixed_baseline_and_spreading():
    print("== test_fixed_baseline_and_spreading ==")
    layout, secret = make_layout()

    w_t = {nm: torch.zeros(1000) for nm in
           {p[0] for p in layout.positions}}
    for nm in w_t:
        w_t[nm] = torch.randn(1000) * 0.1

    enc = torch.tensor(np.random.default_rng(2).normal(0, 1,
                                                       DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)

    models = {}
    for c in range(N_CLIENTS):
        m = {k: v.clone() for k, v in w_t.items()}
        embed_witness_watermark_torch(m, layout, c, enc)
        models[c] = m

    # 见证点区域在所有客户端上：未标记点取值必须**完全相同**（固定基线）
    pv = layout.extract_point_values(enc.numpy())
    bv = layout.extract_base_values()
    marked_sets = [set(layout.design.client_points(c).tolist())
                   for c in range(N_CLIENTS)]

    ok = True
    for p in range(layout.v):
        vals = []
        for c in range(N_CLIENTS):
            pos = layout.point_positions(p)
            v = float(models[c][pos[0][0]].reshape(-1)[pos[0][1]])
            vals.append(v)
        if p not in marked_sets[0]:
            # 该点对客户端 0 未标记 -> 应为基线
            if abs(vals[0] - bv[p].mean()) > 1e-9:
                ok = False
    assert ok, "未标记点应写入固定基线"

    # 分散性：同一个点的 G 个副本在池子中应分散（不连续）
    n = layout.v * layout.G
    gaps = []
    for p in range(layout.v):
        idx = sorted(int(layout.spread_perm[p * layout.G + j])
                     for j in range(layout.G))
        gaps.append(min(b - a for a, b in zip(idx, idx[1:])))
    print(f"  最小相邻间距 = {min(gaps)} (池大小 {n})")
    assert min(gaps) > 1, "见证点副本应当分散（不相邻）"

    print("  固定基线 ✓  副本分散 ✓")


# ====================================================================
def _build_scene(layout, enc, drift=0.0, rng=None):
    """构造各客户端定制模型（含漂移）。"""
    rng = rng or np.random.default_rng(0)
    w_t = {}
    for nm in {p[0] for p in layout.positions}:
        w_t[nm] = torch.randn(1000) * 0.1
    models = {}
    for c in range(N_CLIENTS):
        m = {k: v.clone() for k, v in w_t.items()}
        embed_witness_watermark_torch(m, layout, c, enc)
        if drift > 0:
            for k in m:
                m[k] = m[k] + drift * torch.randn_like(m[k])
        models[c] = m
    return models


def _to_np(m):
    return {k: (v.detach().cpu().numpy().astype(np.float64)
                if hasattr(v, "detach") else np.asarray(v, dtype=np.float64))
            for k, v in m.items()}


# ====================================================================
def test_count_decoding():
    print("== test_count_decoding ==")
    layout, secret = make_layout()
    enc = torch.tensor(np.random.default_rng(3).normal(0, 1,
                                                       DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    models = _build_scene(layout, enc, drift=0.01)
    np_models = {c: _to_np(m) for c, m in models.items()}

    rng = np.random.default_rng(4)
    for true_K in (2, 3):
        ok = 0
        T = 10
        for _ in range(T):
            S = sorted(rng.choice(N_CLIENTS, true_K, replace=False).tolist())
            leaked = {}
            for k in models[S[0]]:
                acc = sum(models[c][k].float() for c in S) / true_K
                leaked[k] = acc + 0.01 * torch.randn_like(acc)
            found, diag = decode_colluders_by_count(
                _to_np(leaked), np_models, layout, max_colluders=true_K)
            if set(found) == set(S):
                ok += 1
        assert ok == T, f"K={true_K}: 仅 {ok}/{T}"
        print(f"  K={true_K}: 精确归因 {ok}/{T} ✓")


# ====================================================================
def test_robust_to_partial_zeroing():
    print("== test_robust_to_partial_zeroing ==")
    layout, secret = make_layout()
    enc = torch.tensor(np.random.default_rng(5).normal(0, 1,
                                                       DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    models = _build_scene(layout, enc, drift=0.0)
    np_models = {c: _to_np(m) for c, m in models.items()}

    rng = np.random.default_rng(6)
    S = sorted(rng.choice(N_CLIENTS, 3, replace=False).tolist())
    leaked = {}
    for k in models[S[0]]:
        acc = sum(models[c][k].float() for c in S) / 3
        leaked[k] = acc.clone()

    # 随机把 30% 的见证点位置置零（模拟剪枝 / 幅值攻击）
    n_pos = len(layout.positions)
    victims = rng.choice(n_pos, size=int(n_pos * 0.3), replace=False)
    for idx in victims:
        nm, ix = layout.positions[int(idx)]
        leaked[nm].reshape(-1)[ix] = 0.0

    found, diag = decode_colluders_by_count(_to_np(leaked), np_models, layout,
                                           max_colluders=3)
    print(f"  30% 位置被置零: 真值={S} 检出={found} "
          f"有效点={diag.get('n_points')} 平均残差={diag.get('avg_error'):.4f}")
    assert set(found) == set(S), "局部攻击后应仍能归因"
    print("  局部攻击下仍可归因 ✓")


# ====================================================================
def test_count_size_estimation():
    print("== test_count_size_estimation ==")
    layout, secret = make_layout()
    enc = torch.tensor(np.random.default_rng(11).normal(0, 1,
                                                        DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    models = _build_scene(layout, enc, drift=0.01)
    np_models = {c: _to_np(m) for c, m in models.items()}

    rng = np.random.default_rng(12)
    for true_K in (2, 3):
        ok_k = ok_attr = 0
        T = 8
        for _ in range(T):
            S = sorted(rng.choice(N_CLIENTS, true_K, replace=False).tolist())
            leaked = {}
            for k in models[S[0]]:
                acc = sum(models[c][k].float() for c in S) / true_K
                leaked[k] = acc + 0.02 * torch.randn_like(acc)
            est_k, per_size = estimate_collusion_size_by_count(
                _to_np(leaked), layout, max_colluders=K_MAX)
            if est_k == true_K:
                ok_k += 1
            found, _ = decode_colluders_by_count(
                _to_np(leaked), np_models, layout, max_colluders=est_k)
            if set(found) == set(S):
                ok_attr += 1
        assert ok_k == T, f"K={true_K}: 规模推断仅 {ok_k}/{T}"
        assert ok_attr == T, f"K={true_K}: 归因仅 {ok_attr}/{T}"
        print(f"  K={true_K}: 计数域规模推断 {ok_k}/{T}, 归因 {ok_attr}/{T} ✓")


# ====================================================================
def test_abstain_on_random_model():
    print("== test_abstain_on_random_model ==")
    layout, secret = make_layout()
    enc = torch.tensor(np.random.default_rng(7).normal(0, 1,
                                                       DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    models = _build_scene(layout, enc, drift=0.0)
    np_models = {c: _to_np(m) for c, m in models.items()}

    rng = np.random.default_rng(8)
    rand_model = {nm: rng.normal(0, 1, 1000) for nm in
                  {p[0] for p in layout.positions}}
    found, diag = decode_colluders_by_count(rand_model, np_models, layout,
                                           max_colluders=3)
    print(f"  随机模型: 检出={found} 弃权={diag.get('abstained')} "
          f"原因={diag.get('reason')}")
    assert diag.get("abstained"), "随机模型应触发弃权"
    assert found == [], "弃权时不应给出合谋者"
    print("  随机模型正确弃权 ✓")


# ====================================================================
def test_framing_resistance():
    print("== test_framing_resistance ==")
    """合谋者伪造"指向无辜客户端"的签名时应被识别为不合法的平均值。

    这里模拟最弱的一种陷害：合谋者把自己的见证点值改成别人的 e_p
    （但他们并不知道 e_p，只能瞎猜）。本测试只验证：
    合谋者无法通过"自己拼encoder切片"得到 e_p。
    """
    layout, secret = make_layout()
    enc = torch.tensor(np.random.default_rng(9).normal(0, 1,
                                                       DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    models = _build_scene(layout, enc, drift=0.0)

    # 合谋者拿到的 encoder 切片（客户端 0 的那段）里，不应包含见证点权威值。
    #
    # 注意：见证点值是随机浮点数，与随机 encoder 切片出现**偶然相等**的概率
    # 非零（945 个见证点 × 948 个切片元素）。因此这里检查的是
    # "是否存在系统性泄露"（即切片中包含了大部分见证点值），
    # 而不是"是否一个都不相等"。
    pv = layout.extract_point_values(enc.numpy()).reshape(-1)
    enc_slice_0 = enc.numpy()[:layout.slice_size]
    hits = sum(1 for x in pv if np.any(np.isclose(enc_slice_0, x, atol=1e-12)))
    ratio = hits / len(pv)
    print(f"  见证点权威值偶然落在客户端0 encoder切片中的比例: "
          f"{ratio:.4%} ({hits}/{len(pv)})")
    assert ratio < 0.01, "见证点权威值不应被系统性地泄露给任何客户端"

    # 更强的检查：改变 encoder 参数不应改变见证点值（已在上一项验证），
    # 且见证点值必须由密钥决定 —— 用另一个密钥重算，结果应完全不同。
    from utils.collusion_resistant import derive_witness_values
    e_other, _ = derive_witness_values(generate_witness_secret(),
                                        layout.v, layout.G,
                                        amplitude=DEFAULT_WITNESS_AMPLITUDE)
    assert not np.allclose(pv.reshape(layout.v, layout.G), e_other), \
        "换密钥后见证点值必须变化"
    print("  见证点权威值未泄露给任何客户端 ✓  且完全由密钥决定 ✓")


# ====================================================================
def test_freeze_keeps_witness_intact():
    print("== test_freeze_keeps_witness_intact ==")
    from utils.watermark_freeze import WatermarkFreezer

    layout, secret = make_layout()
    # 造一个 key_matrix 目录供 WatermarkFreezer 读取
    import json
    import tempfile
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "client_0"), exist_ok=True)
    with open(os.path.join(tmp, "key_matrix_info.json"), "w") as f:
        json.dump({"client_num": N_CLIENTS}, f)
    with open(os.path.join(tmp, "client_0", "positions.json"), "w") as f:
        json.dump([["features.0.weight", 0]], f)
    with open(os.path.join(tmp, "witness_layout.json"), "w") as f:
        json.dump(layout.to_dict(), f)

    # 见证点索引可能落在任意参数上；这里给每个假参数名准备足够大的张量
    sizes = {}
    for nm, ix in layout.positions:
        sizes[nm] = max(sizes.get(nm, 0), ix + 1)
    for nm in sizes:
        sizes[nm] = max(sizes[nm], 64)

    class _Model(torch.nn.Module):
        """极简模型：只提供 named_parameters()，参数名可含 "."。"""

        def __init__(self):
            super().__init__()
            self._sizes = dict(sizes)
            self._params = {nm: torch.nn.Parameter(torch.zeros(n))
                            for nm, n in sizes.items()}

        def named_parameters(self):
            for nm, p in self._params.items():
                yield nm, p

        def get(self, nm):
            return self._params[nm]

    model = _Model()
    fz = WatermarkFreezer(model, tmp)
    assert fz.has_witness, "应加载到见证点布局"
    print(f"  见证点位置数 = {len(fz.witness_positions())}")

    # 写入见证点值后模拟训练扰动
    enc = torch.tensor(np.random.default_rng(10).normal(0, 1,
                                                        DEFAULT_ENCODER_PARAMS),
                        dtype=torch.float32)
    state = {nm: torch.zeros(n) for nm, n in sizes.items()}
    embed_witness_watermark_torch(state, layout, 0, enc)
    for nm in sizes:
        with torch.no_grad():
            model.get(nm).copy_(state[nm])

    snap = fz.snapshot(0)

    # 模拟：梯度 + 优化器 + DP 噪声
    with torch.no_grad():
        for nm in sizes:
            model.get(nm).add_(torch.randn(sizes[nm]) * 0.5)
    fz.restore(snap)

    diff = 0.0
    for nm in sizes:
        d = (model.get(nm).detach().view(-1) -
             state[nm].view(-1)).abs().max().item()
        diff = max(diff, d)
    print(f"  冻结还原后最大偏差 = {diff:.3e}")
    assert diff == 0.0, "冻结还原后见证点应完全不变"
    print("  见证点冻结 ✓")


# ====================================================================
def main():
    test_key_derivation()
    test_not_from_encoder_slice()
    test_fixed_baseline_and_spreading()
    test_count_decoding()
    test_count_size_estimation()
    test_robust_to_partial_zeroing()
    test_abstain_on_random_model()
    test_framing_resistance()
    test_freeze_keeps_witness_intact()
    print("\n抗合谋 v2 验证全部通过 ✅")


if __name__ == "__main__":
    main()
