"""
训练前泄漏追踪回归 —— 确定性验证脚本（不依赖 pytest）

背景
----
提交 4877a2c 把 `_watermark_aggregation` 从 no-op 改成"服务器端把所有客户端的
水印区域 P_k 统一覆盖为 encoder_params 权威值"。这导致：

    M_k 的所有水印区域 P_j 全部 = 权威值（对所有客户端完全相同）
      → 泄漏模型的每个 P_j 分片都等于对应权威水印
      → 所有客户端的追踪相似度全部并列在 1.0 → 无法区分 → 退化为随机

本脚本用**真实密钥矩阵 + 真实编码器参数**构造定制模型，
直接验证追踪能否正确定位泄漏者，并对比"覆盖"与"不覆盖"两种聚合行为。

运行：
    python tests/test_pre_training_tracing.py

退出码 0 表示全部通过。
"""

import os
import sys
import json
import shutil
import tempfile

import numpy as np
import torch

# Windows 控制台默认 GBK，输出中文/emoji 会抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _build_args(model_name='alexnet', client_num=5, key_matrix_dir=None):
    """构造最小可用的 args 对象"""
    from utils.key_matrix_utils import get_key_matrix_path

    class _Args:
        pass

    a = _Args()
    a.model_name = model_name
    a.client_num = client_num
    a.key_matrix_dir = key_matrix_dir or os.path.join(PROJECT_ROOT, 'save', 'key_matrix')
    a.use_key_matrix = True
    a.watermark_mode = 'enhanced'
    a.key_matrix_path = get_key_matrix_path(a.key_matrix_dir, a.model_name, a.client_num)
    return a


def _load_encoder_params():
    """加载编码器权威水印（与 main.py 中 trainer._extract_encoder_parameters 一致）"""
    from models.light_autoencoder import LightAutoencoder
    from utils.autoencoder_finetuner import extract_encoder_parameters

    ae = LightAutoencoder(input_channels=1)
    enc_path = os.path.join(PROJECT_ROOT, 'save', 'autoencoder', 'encoder.pth')
    if os.path.exists(enc_path):
        ae.encoder.load_state_dict(
            torch.load(enc_path, map_location='cpu', weights_only=False)
        )
    return extract_encoder_parameters(ae)


def _build_customized_models(model_state, encoder_params, key_manager, client_num, client_ids):
    """
    复现 main.py 的 `_generate_customized_model`：
      客户端 k 的 P_k = 权威值；P_j (j≠k) 保持 w_t 原值。
    """
    from main import FederatedLearningOnChestMNIST

    # 计算各客户端在 encoder_params 中的偏移（按 client_id 顺序）
    lengths = {cid: len(key_manager.load_positions(cid)) for cid in range(client_num)}
    offsets, cum = {}, 0
    for cid in range(client_num):
        offsets[cid] = cum
        cum += lengths.get(cid, 0)

    customized = {}
    for cid in client_ids:
        positions = key_manager.load_positions(cid)
        model = FederatedLearningOnChestMNIST.__new__(FederatedLearningOnChestMNIST)
        out = model._generate_customized_model(
            cid, model_state, positions, encoder_params, offsets[cid],
            model._build_param_offset_map(model_state)
        )
        customized[cid] = out
    return customized


def _simulate_aggregate(model_state, customized, client_ids, overwrite_authority):
    """
    模拟一轮服务器端聚合，返回下一轮的 w_t。

    overwrite_authority=True  → 复现 4877a2c：把所有 P_k 覆盖为权威值
    overwrite_authority=False → 保持纯 FedAvg（修复后的行为）
    """
    from main import FederatedLearningOnChestMNIST
    from utils.key_matrix_utils import KeyMatrixManager, get_key_matrix_path

    args = _build_args()
    key_manager = KeyMatrixManager(args.key_matrix_path, args=args)

    # 简易 FedAvg：所有客户端定制模型的均值
    w_avg = {}
    for k in customized[client_ids[0]].keys():
        w_avg[k] = sum(customized[c][k] for c in client_ids) / len(client_ids)

    if overwrite_authority:
        encoder_params = _load_encoder_params()
        lengths = {cid: len(key_manager.load_positions(cid)) for cid in range(len(client_ids))}
        offsets, cum = {}, 0
        for cid in range(len(client_ids)):
            offsets[cid] = cum
            cum += lengths.get(cid, 0)

        model = FederatedLearningOnChestMNIST.__new__(FederatedLearningOnChestMNIST)
        offset_map, _ = model._build_param_offset_map(w_avg)
        for cid in range(len(client_ids)):
            positions = key_manager.load_positions(cid)
            for wm_local_idx, (pname, gidx) in enumerate(positions):
                local_idx, actual = model._convert_global_to_local_idx(gidx, pname, w_avg, offset_map)
                if local_idx is None or actual is None:
                    continue
                enc_idx = offsets[cid] + wm_local_idx
                if enc_idx < len(encoder_params):
                    w_avg[actual].view(-1)[local_idx] = encoder_params[enc_idx].item()

    return w_avg


def _track(exp, leaked_model, client_ids, encoder_params):
    """复现 main.py 的 `_track_model_leakage`"""
    from utils.key_matrix_utils import KeyMatrixManager
    key_manager = KeyMatrixManager(exp.args.key_matrix_path, args=exp.args)

    encoder_np = encoder_params.cpu().numpy()
    cum = 0
    authority = {}
    for cid in range(exp.client_num):
        wm_len = len(key_manager.load_positions(cid))
        authority[cid] = encoder_np[cum:cum + wm_len]
        cum += wm_len

    rankings = []
    for cid in client_ids:
        leaked_wm = exp._extract_single_client_watermark_params(leaked_model, cid)
        auth = authority[cid]
        if len(leaked_wm) > 0 and len(auth) > 0:
            n = min(len(leaked_wm), len(auth))
            sim = exp._cosine_similarity(leaked_wm[:n], auth[:n])
        else:
            sim = 0.0
        rankings.append((cid, sim))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings, (rankings[0][0] if rankings else None)


def _make_experiment(args):
    from main import FederatedLearningOnChestMNIST
    exp = FederatedLearningOnChestMNIST.__new__(FederatedLearningOnChestMNIST)
    exp.args = args
    exp.client_num = args.client_num
    return exp


def main():
    from models.alexnet import AlexNet
    from utils.key_matrix_utils import KeyMatrixManager, get_key_matrix_path

    args = _build_args()

    if not os.path.isdir(args.key_matrix_path):
        print(f'⚠️  密钥矩阵目录不存在: {args.key_matrix_path}')
        print('   请先运行: python train_key_matrix.py --model_type alexnet --client_num 5')
        return 1

    key_manager = KeyMatrixManager(args.key_matrix_path, args=args)
    encoder_params = _load_encoder_params()
    print(f'编码器权威水印维度: {len(encoder_params)}')

    # 构造一个初始全局模型（随机初始化 + 模拟一次 FedAvg 后状态）
    torch.manual_seed(0)
    model = AlexNet(num_classes=14, in_channels=1, input_size=28)
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    client_ids = list(range(args.client_num))
    exp = _make_experiment(args)

    results = {}

    for overwrite in (True, False):
        label = '4877a2c(覆盖)' if overwrite else '修复后(纯FedAvg)'
        print()
        print('=' * 74)
        print(f'场景: {label}')
        print('=' * 74)

        w_t = {k: v.clone() for k, v in model_state.items()}
        correct = 0
        total = 0
        sim_gaps = []

        for round_idx in range(3):
            customized = _build_customized_models(
                w_t, encoder_params, key_manager, args.client_num, client_ids
            )

            # 检查"同一区域 P_j 在不同客户端模型中的内容是否相同"（4877a2c 的核心症状）
            # 4877a2c：M_k 的 P_j 全部 = 权威值 → 不同客户端的同一个区域内容完全相同
            # 修复后：M_k 的 P_j (j≠k) = w_t 的 FedAvg 值 → 不同客户端之间不同
            if round_idx == 1:
                probe_client = client_ids[0]
                slices = {}
                for cid in client_ids:
                    # 取 customized[cid] 中 probe_client 负责的区域 P_probe
                    slices[cid] = exp._extract_single_client_watermark_params(
                        customized[cid], probe_client
                    )
                all_same = True
                for cid in client_ids[1:]:
                    a, b = slices[client_ids[0]], slices[cid]
                    n = min(len(a), len(b))
                    if n == 0 or not np.allclose(a[:n], b[:n], atol=1e-6):
                        all_same = False
                        break
                print(f'  [症状检查] 区域 P{probe_client} 在各客户端定制模型中是否完全相同: {all_same}'
                      f'  (相同=无法区分，追踪必然退化)')

            # 泄漏客户端（选取一个有区分度的客户端）
            leaker = client_ids[round_idx % len(client_ids)]
            leaked_model = customized[leaker]

            rankings, detected = _track(exp, leaked_model, client_ids, encoder_params)
            total += 1
            if detected == leaker:
                correct += 1

            top = rankings[0]
            second = rankings[1] if len(rankings) > 1 else (None, 0.0)
            sim_gaps.append(top[1] - second[1])
            print(f'  第{round_idx + 1}轮: 实际泄漏者=C{leaker} -> 检测=C{detected} '
                  f'| top sim={top[1]:.6f}, 次高=C{second[0]} sim={second[1]:.6f} '
                  f'| 区分度={top[1] - second[1]:.6f}')

            # 下一轮的 w_t
            w_t = _simulate_aggregate(w_t, customized, client_ids, overwrite)

        acc = correct / total if total else 0.0
        results[label] = {'accuracy': acc, 'correct': correct, 'total': total}
        print(f'  => 追踪准确率: {correct}/{total} = {acc * 100:.2f}%')

    print()
    print('=' * 74)
    print('汇总')
    print('=' * 74)
    for label, r in results.items():
        print(f'  {label:<22} 追踪准确率 = {r["accuracy"] * 100:6.2f}%  ({r["correct"]}/{r["total"]})')

    ok = True
    fixed = results['修复后(纯FedAvg)']
    broken = results['4877a2c(覆盖)']

    print()
    if fixed['accuracy'] < 0.99:
        print('❌ 修复后追踪准确率未恢复到 100%')
        ok = False
    else:
        print('✅ 修复后追踪准确率恢复到 100%')

    if broken['accuracy'] > fixed['accuracy']:
        print('❌ 覆盖场景表现不劣于修复场景（回归未复现，测试可能失效）')
        ok = False
    else:
        print('✅ 成功复现 4877a2c 引入的追踪退化（覆盖场景显著更差）')

    print()
    print('=' * 74)
    print('结果: ' + ('全部通过 ✅' if ok else '存在失败 ❌'))
    print('=' * 74)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
