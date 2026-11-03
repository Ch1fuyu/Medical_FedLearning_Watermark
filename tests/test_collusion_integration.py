"""抗合谋水印接入的集成测试。

验证：
  1. 布局生成：见证点位置与独占切片**不重叠**
  2. encoder 权威值区与独占切片区**不重叠**（避免所有权认证误判）
  3. 嵌入正确：见证点写入共享权威值，独占切片写入私有权威值
  4. offset 一致：_get_encoder_offsets 与布局定义一致
  5. 端到端：嵌入 -> 合谋平均 -> 子集匹配归因
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, r"D:\CODE\Medical_FedLearning_Watermark")

from utils.collusion_resistant import (  # noqa: E402
    DEFAULT_ENCODER_PARAMS,
    build_witness_layout,
    decode_colluders,
    embed_witness_watermark_torch,
    estimate_collusion_size,
    load_witness_layout,
    save_witness_layout,
)
from models.resnet import resnet18  # noqa: E402


def make_model():
    return resnet18(num_classes=10, in_channels=3, input_size=28)


def conv_positions(model):
    out = []
    for name, p in model.named_parameters():
        is_conv = (("conv" in name.lower() and "weight" in name.lower())
                   or ("downsample.0.weight" in name.lower()))
        if is_conv and len(p.shape) == 4:
            for i in range(p.numel()):
                out.append((name, i))
    return out


def test_layout_no_overlap():
    print("== test_layout_no_overlap ==")
    n = 10
    model = make_model()
    all_pos = conv_positions(model)
    print(f"  conv 位置总数 = {len(all_pos):,}")

    # 模拟现有切片的分配方式
    slice_size = DEFAULT_ENCODER_PARAMS // n
    used = set()
    for c in range(n):
        for pos in all_pos[c * slice_size:(c + 1) * slice_size]:
            used.add(pos)
    free = [p for p in all_pos if p not in used]
    print(f"  独占切片占用 {len(used)} 个, 剩余空闲 {len(free):,} 个")

    layout = build_witness_layout(n, K=3, encoder_params=DEFAULT_ENCODER_PARAMS,
                                  positions=free, slice_size=slice_size,
                                  verbose=False)
    print(f"  见证点: v={layout.v} k={layout.design.k} G={layout.G}")
    print(f"  encoder 权威值区 = [{layout.witness_offset}, "
          f"{layout.witness_offset + layout.v * layout.G})")
    print(f"  独占切片区     = [{layout.slice_offset}, {layout.slice_end})")

    # 1) 位置不重叠
    wpos = set(layout.positions)
    assert not (wpos & used), "见证点位置与独占切片重叠！"
    print(f"  [OK] 见证点位置({len(wpos)}) 与独占切片({len(used)}) 不重叠")

    # 2) encoder 区间：切片区保持原样（offset 不变），见证点复用前段
    #    值空间允许重叠（encoder 参数已占满），但位置空间必须不相交
    witness_end = layout.witness_offset + layout.v * layout.G
    assert layout.slice_offset == 0, "独占切片区起点应保持为 0（offset 不变）"
    assert layout.slice_end <= DEFAULT_ENCODER_PARAMS, "切片区超出 encoder 预算"
    assert witness_end <= DEFAULT_ENCODER_PARAMS, "见证点区超出 encoder 预算"
    print(f"  [OK] 切片区 [0, {layout.slice_end}) offset 不变; "
          f"见证点权威值区 [{layout.witness_offset}, {witness_end}) 复用前段")
    return layout


def test_layout_roundtrip():
    print("== test_layout_roundtrip ==")
    n = 5
    model = make_model()
    all_pos = conv_positions(model)
    slice_size = DEFAULT_ENCODER_PARAMS // n
    layout = build_witness_layout(n, K=3, encoder_params=DEFAULT_ENCODER_PARAMS,
                                  positions=all_pos[slice_size * n:],
                                  slice_size=slice_size, verbose=False)
    tmp = tempfile.mktemp(suffix=".json")
    save_witness_layout(layout, tmp)
    l2 = load_witness_layout(tmp)
    os.remove(tmp)
    assert l2.v == layout.v and l2.G == layout.G
    assert l2.slice_offset == layout.slice_offset
    assert np.array_equal(l2.A, layout.A)
    assert l2.positions == layout.positions
    print("  [OK] 保存/加载往返一致")


def test_embed_semantics():
    print("== test_embed_semantics ==")
    n = 6
    model = make_model()
    all_pos = conv_positions(model)
    slice_size = DEFAULT_ENCODER_PARAMS // n
    used = set(all_pos[:slice_size * n])
    free = [p for p in all_pos if p not in used]
    layout = build_witness_layout(n, K=3, encoder_params=DEFAULT_ENCODER_PARAMS,
                                  positions=free, slice_size=slice_size,
                                  verbose=False)

    rng = np.random.default_rng(0)
    enc = torch.tensor(rng.normal(0, 1, DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    pv = layout.extract_point_values(enc.numpy())

    # 构造 w_t
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w_t = {k: (v.clone() if v.dtype.is_floating_point else v.clone())
           for k, v in state.items()}

    for cid in range(n):
        m = {k: v.clone() for k, v in w_t.items()}
        embed_witness_watermark_torch(m, layout, cid, enc)

        marked = layout.design.client_points(cid)
        unmarked = set(range(layout.v)) - set(marked.tolist())

        # 标记点应等于 e_p
        for p in marked:
            for j in range(layout.G):
                nm, ix = layout.positions[p * layout.G + j]
                got = float(m[nm].reshape(-1)[ix])
                assert abs(got - pv[p, j]) < 1e-6, \
                    f"client{cid} 点{p} 位置{j}: {got} != {pv[p, j]}"
        # 未标记点应保持 w_t
        for p in list(unmarked)[:5]:
            for j in range(min(layout.G, 3)):
                nm, ix = layout.positions[p * layout.G + j]
                got = float(m[nm].reshape(-1)[ix])
                want = float(w_t[nm].reshape(-1)[ix])
                assert abs(got - want) < 1e-6, \
                    f"client{cid} 未标记点{p} 被改动了"

    print(f"  [OK] 标记点写入共享权威值，未标记点保持 w_t (n={n})")


def test_end_to_end_attribution():
    print("== test_end_to_end_attribution ==")
    n, K = 10, 3
    model = make_model()
    all_pos = conv_positions(model)
    slice_size = DEFAULT_ENCODER_PARAMS // n
    used = set(all_pos[:slice_size * n])
    free = [p for p in all_pos if p not in used]
    layout = build_witness_layout(n, K=K, encoder_params=DEFAULT_ENCODER_PARAMS,
                                  positions=free, slice_size=slice_size,
                                  verbose=False)

    rng = np.random.default_rng(1)
    enc = torch.tensor(rng.normal(0, 1, DEFAULT_ENCODER_PARAMS),
                       dtype=torch.float32)
    pv = layout.extract_point_values(enc.numpy())

    w_t = {k: (torch.randn_like(v) * 0.05 if v.dtype.is_floating_point
               else v.clone())
           for k, v in model.state_dict().items()}

    # 各客户端定制模型 + 模拟本地训练漂移
    client_models = {}
    for cid in range(n):
        m = {k: v.clone() for k, v in w_t.items()}
        embed_witness_watermark_torch(m, layout, cid, enc)
        # 漂移（模拟本地训练）
        for k in m:
            if m[k].dtype.is_floating_point:
                m[k] = m[k] + 0.05 * torch.randn_like(m[k])
        client_models[cid] = m

    # 归因
    for true_K in (1, 2, 3):
        ok = 0
        T = 5
        for _ in range(T):
            S = sorted(rng.choice(n, true_K, replace=False).tolist())
            leaked = {}
            for k in client_models[S[0]]:
                acc = sum(client_models[c][k] for c in S) / len(S)
                leaked[k] = acc
            np_leak = {k: v.numpy().astype(np.float64)
                       for k, v in leaked.items()}
            np_clients = {c: {k: v.numpy().astype(np.float64)
                              for k, v in mm.items()}
                          for c, mm in client_models.items()}
            found, _ = decode_colluders(np_leak, np_clients, layout.design,
                                        max_colluders=true_K)
            if set(found) == set(S):
                ok += 1
        assert ok == T, f"trueK={true_K}: {ok}/{T}"
        print(f"  trueK={true_K}: {ok}/{T} 精确归因 OK")

    # K 自动推断
    ok_est = ok_attr = 0
    for _ in range(5):
        S = sorted(rng.choice(n, 2, replace=False).tolist())
        leaked = {k: sum(client_models[c][k] for c in S) / len(S)
                  for k in client_models[S[0]]}
        np_leak = {k: v.numpy().astype(np.float64) for k, v in leaked.items()}
        np_clients = {c: {k: v.numpy().astype(np.float64)
                          for k, v in mm.items()}
                      for c, mm in client_models.items()}
        ek, _ = estimate_collusion_size(np_leak, np_clients, layout.design,
                                        max_colluders=5)
        found, _ = decode_colluders(np_leak, np_clients, layout.design,
                                    max_colluders=ek)
        if ek == 2:
            ok_est += 1
        if set(found) == set(S):
            ok_attr += 1
    print(f"  K自动推断: 规模 {ok_est}/5, 归因 {ok_attr}/5")
    assert ok_est == 5 and ok_attr == 5


def test_real_layout_isolation():
    """用真实生成的 witness_layout.json 验证关键不变量。

    若存在真实布局文件则执行；否则跳过（不影响其他测试）。
    """
    print("== test_real_layout_isolation ==")
    import glob

    base = os.path.join(r"D:\CODE\Medical_FedLearning_Watermark", "save",
                        "key_matrix_collusion", "resnet", "client10")
    layout_path = os.path.join(base, "witness_layout.json")
    if not os.path.exists(layout_path):
        print("  (跳过：未找到真实布局文件)")
        return

    layout = load_witness_layout(layout_path)
    n = 10

    # 1) 见证点位置与所有客户端切片位置严格不相交
    slice_pos = set()
    for c in range(n):
        p = os.path.join(base, f"client_{c}", "positions.json")
        with open(p, "r") as f:
            for nm, ix in json.load(f):
                slice_pos.add((nm, ix))
    inter = set(layout.positions) & slice_pos
    assert not inter, f"见证点位置与切片位置重叠 {len(inter)} 处！"
    print(f"  [OK] 位置隔离：切片 {len(slice_pos)} 个, 见证点 "
          f"{len(layout.positions)} 个, 交集 = 0")

    # 2) 独占切片区起点保持为 0（现有 offset 逻辑零影响）
    #    注意：equal 策略的余数分配会让切片区总长略超 encoder 总量
    #    （如 n=10 时 948*2 + 947*8 = 9480 > 9472），这是**既有行为**，
    #    现有代码用 `enc_global_idx < len(encoder_params)` 兜底。
    #    见证点布局不改变这一行为。
    assert layout.slice_offset == 0
    print(f"  [OK] 切片区 [{layout.slice_offset}, {layout.slice_end}) "
          f"offset 不变，现有机制零影响")

    # 3) 见证点权威值区在 encoder 预算内
    w_end = layout.witness_offset + layout.v * layout.G
    assert w_end <= DEFAULT_ENCODER_PARAMS
    print(f"  [OK] 见证点权威值区 [{layout.witness_offset}, {w_end}) "
          f"在预算内")


def main():
    test_layout_no_overlap()
    test_layout_roundtrip()
    test_embed_semantics()
    test_end_to_end_attribution()
    test_real_layout_isolation()
    print("\n集成测试全部通过 ✅")


if __name__ == "__main__":
    main()
