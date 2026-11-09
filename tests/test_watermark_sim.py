"""验证水印区域相似度 sim 的计算逻辑。

覆盖：
1. 余弦相似度基本性质（自身=1、正交=0、零向量=0）
2. 与原始模型比较时 sim=1（微调前基线）
3. 水印区域被扰动后 sim 下降
4. 剪枝（水印位置置 0）后 sim 显著下降
5. 格式化和打印函数不报错
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import AlexNet
from utils.key_matrix_utils import KeyMatrixManager
from utils.watermark_sim_utils import (
    cosine_similarity,
    compute_watermark_similarity,
    extract_watermark_vector,
    format_watermark_similarity,
    print_watermark_similarity,
)


def build_model():
    model = AlexNet(3, 10, input_size=32, dropout_rate=0.5)
    # 模拟一次训练，让参数不是初始值
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model


def test_cosine_similarity():
    print('=== 测试余弦相似度基本性质 ===')
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6, '自身相似度应为1'
    b = torch.tensor([0.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0, '零向量相似度应为0'
    c = torch.tensor([-1.0, -2.0, -3.0])
    assert abs(cosine_similarity(a, c) + 1.0) < 1e-6, '反向向量相似度应为-1'
    d = torch.tensor([-2.0, 1.0, 0.0])  # 与 a=[1,2,3] 正交
    assert abs(cosine_similarity(a, d)) < 1e-6, '正交向量相似度应为0'
    print('  ✓ 全部通过')


def test_sim_pipeline():
    key_matrix_path = 'save/key_matrix/alexnet/client5'
    if not os.path.exists(key_matrix_path):
        print(f'⚠️  密钥矩阵不存在: {key_matrix_path}，跳过集成测试')
        return

    print('\n=== 测试水印相似度计算流程 ===')
    model = build_model()
    km = KeyMatrixManager(key_matrix_path)

    state = model.state_dict()

    # 1. 自身对比 -> sim = 1
    res = compute_watermark_similarity(state, state, km, model=model)
    print(f'  自身对比 sim: {res["wm_sim"]:.6f} (期望 1.0)')
    assert abs(res['wm_sim'] - 1.0) < 1e-5, f'自身对比 sim 应为1，实际 {res["wm_sim"]}'
    assert res['wm_sim_mean'] is not None

    # 2. 微扰动 -> sim 略降但仍很高
    perturbed = {k: v.clone() for k, v in state.items()}
    vec = extract_watermark_vector(state, km, model=model)
    print(f'  水印向量长度: {vec.numel()}')
    assert vec.numel() > 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in perturbed:
                perturbed[name] += torch.randn_like(param) * 0.001
    res2 = compute_watermark_similarity(perturbed, state, km, model=model)
    print(f'  微扰动后 sim: {res2["wm_sim"]:.6f} (期望 ≈1)')
    assert res2['wm_sim'] > 0.99, f'微扰动后 sim 应接近1，实际 {res2["wm_sim"]}'

    # 3. 大扰动 -> sim 明显下降
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in perturbed:
                perturbed[name] += torch.randn_like(param) * 1.0
    res3 = compute_watermark_similarity(perturbed, state, km, model=model)
    print(f'  大扰动后 sim: {res3["wm_sim"]:.6f} (期望明显 < 1)')
    assert res3['wm_sim'] < res2['wm_sim'], '大扰动后 sim 应下降'

    # 4. 剪枝（水印位置置0）-> sim 大幅下降，非零占比下降
    pruned = {k: v.clone() for k, v in state.items()}
    pruned_count = 0
    for cid in km.list_clients():
        for param_name, local_idx in km.load_positions(cid):
            if param_name in pruned:
                flat = pruned[param_name].view(-1)
                if local_idx < flat.numel():
                    flat[local_idx] = 0.0
                    pruned_count += 1
    print(f'  已置零水印位置数: {pruned_count}')
    res4 = compute_watermark_similarity(pruned, state, km, model=model)
    print(f'  剪枝后 sim: {res4["wm_sim"]:.6f}, 非零占比: {res4["wm_nonzero_ratio"]:.2%}')
    assert res4['wm_sim'] < 0.5, f'剪枝后 sim 应大幅下降，实际 {res4["wm_sim"]}'
    assert res4['wm_nonzero_ratio'] == 0.0

    # 5. 格式化与打印
    fmt = format_watermark_similarity(res4)
    assert 'wm_sim' in fmt and 'wm_rel_dev' in fmt
    print_watermark_similarity(res4, prefix='  [剪枝] ')
    print_watermark_similarity(None)
    print_watermark_similarity(format_watermark_similarity(None))
    print('  ✓ 全部通过')


def test_per_client_sim():
    """验证每个客户端的相似度被单独计算并平铺输出"""
    key_matrix_path = 'save/key_matrix/alexnet/client5'
    if not os.path.exists(key_matrix_path):
        print(f'⚠️  密钥矩阵不存在: {key_matrix_path}，跳过集成测试')
        return

    print('\n=== 测试各客户端相似度 ===')
    model = build_model()
    km = KeyMatrixManager(key_matrix_path)
    state = model.state_dict()

    # 自身对比 -> 每个客户端都应为 1
    res = compute_watermark_similarity(state, state, km, model=model)
    per_client = res['wm_sim_per_client']
    detail = res['wm_sim_per_client_detail']

    print(f'  客户端数量: {len(per_client)}')
    assert len(per_client) == km.client_num, '每个客户端都应有相似度'
    for cid in sorted(per_client):
        print(f'    客户端 {cid}: sim={per_client[cid]:.6f} '
              f'| 水印长度={detail[cid]["wm_size"]} '
              f'| 非零占比={detail[cid]["nonzero_ratio"]:.2%}')
        assert abs(per_client[cid] - 1.0) < 1e-5, f'客户端 {cid} 自身对比 sim 应为1'
        assert detail[cid]['wm_size'] > 0, f'客户端 {cid} 水印长度应大于0'
        assert detail[cid]['rel_dev'] is not None

    # format 后应平铺为 wm_sim_c{cid} 独立列
    fmt = format_watermark_similarity(res)
    for cid in sorted(per_client):
        key = f'wm_sim_c{cid}'
        assert key in fmt, f'格式化结果应包含 {key}'
        assert fmt[key] is not None

    # 破坏单个客户端的水印 -> 只有该客户端 sim 下降
    broken = {k: v.clone() for k, v in state.items()}
    target_cid = 0
    for param_name, local_idx in km.load_positions(target_cid):
        if param_name in broken:
            flat = broken[param_name].view(-1)
            if local_idx < flat.numel():
                flat[local_idx] = 0.0

    res_broken = compute_watermark_similarity(broken, state, km, model=model)
    pc_broken = res_broken['wm_sim_per_client']
    print(f'  破坏客户端 {target_cid} 后:')
    for cid in sorted(pc_broken):
        print(f'    客户端 {cid}: sim={pc_broken[cid]:.6f}')
    assert pc_broken[target_cid] < 0.99, '被破坏客户端的 sim 应显著下降'
    for cid in sorted(pc_broken):
        if cid != target_cid:
            assert pc_broken[cid] > 0.999, f'未破坏客户端 {cid} 的 sim 应保持接近1'

    print('  ✓ 全部通过')


def main():
    test_cosine_similarity()
    test_sim_pipeline()
    test_per_client_sim()
    print('\n✅ 水印相似度测试全部通过')


if __name__ == '__main__':
    main()
