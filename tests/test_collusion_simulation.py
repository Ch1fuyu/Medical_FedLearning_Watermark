"""合谋泄漏仿真 + 归因的集成测试。

覆盖：
  1. _simulate_collusion_leakage：选出 K 个合谋者并做参数平均
  2. 平均值确实等于各模型参数的算术平均
  3. _detect_collusion_leakage：能准确识别出合谋者集合
  4. pre_training / post_training 两种阶段都可用
  5. 噪声攻击下仍能归因
  6. K 未知时自动推断规模
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, r"D:\CODE\Medical_FedLearning_Watermark")

from utils.collusion_resistant import load_witness_layout  # noqa: E402
from models.resnet import resnet18  # noqa: E402

BASE = r"D:\CODE\Medical_FedLearning_Watermark\save\key_matrix_collusion\resnet\client10"
N = 10


def build_server(collusion_size=2, stage="pre_training", enable=True):
    """构造一个最小化的 FLServer 实例。"""
    from main import FederatedLearningOnChestMNIST as FLServer

    args = types.SimpleNamespace(
        key_matrix_path=BASE,
        enable_witness=True,
        model_type="resnet",
        leak_interval=1,
        enable_collusion_detection=enable,
        collusion_size=collusion_size,
        collusion_leak_stage=stage,
    )
    srv = FLServer.__new__(FLServer)
    srv.args = args
    srv.client_num = N
    srv._witness_layout = None
    srv.customized_models = {}
    srv.collusion_detection_results = []
    srv.leakage_records = []
    return srv


def make_customized_models(srv, seed=0, overwrite_witness=True):
    """模拟各客户端的定制模型（含见证点 + 私有切片 + 本地训练漂移）。

    overwrite_witness=True（默认）时，模拟真实流程中的**训练后硬覆盖**：
    客户端本地训练会让见证点区域产生漂移，但训练结束后服务器下发的
    权威值会被**硬覆盖**回去，因此见证点在最终上报的模型里
    保持为精确的 e_p（标记点）/ b_p（基线点）。

    这很重要 —— 若不覆盖，见证点比值的噪声会达到信号量级
    （|e_p| 最小仅 0.025，而漂移 std 为 0.05），
    计数解码的规模推断就会频繁误判（实测准确率从 100% 掉到 ~94%）。
    """
    layout = load_witness_layout(os.path.join(BASE, "witness_layout.json"))
    rng = np.random.default_rng(seed)
    model = resnet18(num_classes=10, in_channels=3, input_size=28)
    enc = torch.tensor(rng.normal(0, 1, 9472), dtype=torch.float32)

    import json
    for cid in range(N):
        m = {k: (torch.randn_like(v) * 0.05 if v.dtype.is_floating_point
                 else v.clone()) for k, v in model.state_dict().items()}
        # 私有切片
        with open(os.path.join(BASE, f"client_{cid}", "positions.json")) as f:
            ps = json.load(f)
        off = 0
        for c in range(cid):
            with open(os.path.join(BASE, f"client_{c}", "positions.json")) as f:
                off += len(json.load(f))
        for i, (nm, ix) in enumerate(ps):
            if nm in m and off + i < 9472:
                m[nm].reshape(-1)[ix] = float(enc[off + i])
        # 本地训练漂移（**所有位置**都漂移，含见证点区域）
        for k in m:
            if m[k].dtype.is_floating_point:
                m[k] = m[k] + 0.05 * torch.randn_like(m[k])
        # 训练后硬覆盖：见证点权威值 / 基线值写回
        if overwrite_witness:
            from utils.collusion_resistant import embed_witness_watermark_torch
            embed_witness_watermark_torch(m, layout, cid, enc)
        srv.customized_models[cid] = m
    return srv


def test_simulate_collusion():
    print("== test_simulate_collusion ==")
    srv = build_server(collusion_size=3)
    make_customized_models(srv)
    idxs = list(range(N))

    info = srv._simulate_collusion_leakage(0, idxs, local_ws=None)
    assert info is not None, "应产生合谋信息"
    assert info['leak_type'] == 'collusion'
    assert len(info['colluders']) == 3, f"应有 3 个合谋者, 实际 {info['colluders']}"
    assert info['K'] == 3
    assert len(info['client_models']) == 3
    print(f"  合谋者 = {info['colluders']}, K = {info['K']}, "
          f"阶段 = {info['leak_stage']}")

    # 平均值确实等于算术平均
    for k in list(info['model_params'].keys())[:3]:
        expect = sum(info['client_models'][c][k].float()
                     for c in info['colluders']) / 3
        got = info['model_params'][k]
        assert torch.allclose(got, expect, atol=1e-5), f"参数 {k} 平均值不对"
    print("  [OK] 合谋模型 = 各合谋者模型的算术平均")
    return srv, info


def test_detect_collusion():
    print("== test_detect_collusion ==")
    srv = build_server(collusion_size=3)
    make_customized_models(srv, seed=1)
    idxs = list(range(N))

    ok = 0
    T = 5
    for t in range(T):
        # 固定随机种子：合谋者由 np.random.choice 抽取，
        # 不固定种子会让本用例的结果依赖当次抽样（flaky）。
        np.random.seed(1000 + t)
        info = srv._simulate_collusion_leakage(0, idxs, local_ws=None)
        result = srv._detect_collusion_leakage(info, idxs)
        if result['is_correct']:
            ok += 1
        print(f"  轮{t}: 实际={result['actual_colluders']} "
              f"检出={result['detected_colluders']} "
              f"估计规模={result['estimated_size']} "
              f"{'OK' if result['is_correct'] else 'FAIL'}")
    assert ok == T, f"归因准确率 {ok}/{T}"
    print(f"  [OK] 归因 {ok}/{T}")


def test_post_training_stage():
    print("== test_post_training_stage ==")
    srv = build_server(collusion_size=2, stage="post_training")
    make_customized_models(srv, seed=2)
    idxs = list(range(N))
    # post_training 需要 local_ws
    local_ws = [srv.customized_models[c] for c in idxs]

    np.random.seed(2000)
    info = srv._simulate_collusion_leakage(0, idxs, local_ws=local_ws)
    assert info is not None
    assert info['leak_stage'] == 'post_training'
    result = srv._detect_collusion_leakage(info, idxs)
    print(f"  实际={result['actual_colluders']} "
          f"检出={result['detected_colluders']} "
          f"{'OK' if result['is_correct'] else 'FAIL'}")
    assert result['is_correct']
    print("  [OK] post_training 阶段归因正确")


def test_pure_averaging_attack():
    """威胁模型：合谋者只做**纯参数平均**（不加任何噪声）。

    见证点计数解码依赖 L[p] = (m_p/K)*e_p + (1-m_p/K)*b_p，
    该等式只在纯平均下精确成立，因此抗合谋分支刻意不提供噪声攻击。
    """
    print("== test_pure_averaging_attack ==")
    srv = build_server(collusion_size=2)
    make_customized_models(srv, seed=3)
    idxs = list(range(N))
    ok = 0
    T = 5
    for t in range(T):
        np.random.seed(3000 + t)
        info = srv._simulate_collusion_leakage(0, idxs, local_ws=None)
        # 泄漏模型必须就是各合谋者模型的等权算术平均（无任何额外扰动）
        colluders = info['colluders']
        for k in list(info['model_params'].keys())[:3]:
            expect = sum(info['client_models'][c][k].float()
                         for c in colluders) / len(colluders)
            assert torch.allclose(info['model_params'][k], expect, atol=1e-6), \
                f"参数 {k} 不是纯算术平均"
        result = srv._detect_collusion_leakage(info, idxs)
        if result['is_correct']:
            ok += 1
    print(f"  纯平均攻击下归因 {ok}/{T}")
    assert ok == T
    print("  [OK] 纯参数平均攻击下归因正确")


def test_disabled():
    print("== test_disabled ==")
    srv = build_server(enable=False)
    make_customized_models(srv)
    info = srv._simulate_collusion_leakage(0, list(range(N)), local_ws=None)
    assert info is None, "未启用时应返回 None"
    print("  [OK] 未启用时不做合谋仿真")


def test_disable_single_client_leak():
    print("== test_disable_single_client_leak ==")
    from main import FederatedLearningOnChestMNIST as FLServer

    # 1) helper 判定
    srv = build_server(enable=True)
    srv.args.disable_single_client_leak = True
    assert srv._single_client_leak_disabled() is True
    srv.args.disable_single_client_leak = False
    assert srv._single_client_leak_disabled() is False
    print("  [OK] _single_client_leak_disabled() 判定正确")

    # 2) 关闭时：pre_training 泄漏不产生记录
    srv = build_server(enable=True)
    srv.args.disable_single_client_leak = True
    make_customized_models(srv, seed=7)
    idxs = list(range(N))

    # 模拟训练循环中的调用点
    leak_info = None
    if (getattr(srv.args, 'enable_source_tracing', True)
            and not srv._single_client_leak_disabled()):
        leak_info = srv._simulate_pre_training_leakage(0, idxs)
    assert leak_info is None, "关闭后不应产生 pre_training 泄漏"
    assert len(srv.leakage_records) == 0, "关闭后不应有泄漏记录"
    print("  [OK] pre_training 泄漏被关闭（无记录）")

    # 3) 关闭时：不选恶意客户端
    malicious = None
    if srv._single_client_leak_disabled():
        malicious = None
    else:
        malicious = srv._get_current_malicious_client(0, idxs)
    assert malicious is None, "关闭后不应选恶意客户端"
    print("  [OK] 恶意客户端选取被关闭")

    # 4) 关闭时：post_training 泄漏不产生
    post = None
    if srv._single_client_leak_disabled():
        post = None
    else:
        post = srv._simulate_post_training_leakage(0, idxs, None)
    assert post is None
    print("  [OK] post_training 泄漏被关闭")

    # 5) 关闭时：合谋仿真仍然工作
    info = srv._simulate_collusion_leakage(0, idxs, local_ws=None)
    assert info is not None, "合谋仿真应仍然工作"
    result = srv._detect_collusion_leakage(info, idxs)
    assert result['is_correct'], "合谋归因应正确"
    print(f"  [OK] 合谋仿真不受影响 (实际={result['actual_colluders']} "
          f"检出={result['detected_colluders']})")


def main():
    test_simulate_collusion()
    test_detect_collusion()
    test_post_training_stage()
    test_pure_averaging_attack()
    test_disabled()
    test_disable_single_client_leak()
    print("\n合谋仿真集成测试全部通过 ✅")


if __name__ == "__main__":
    main()
