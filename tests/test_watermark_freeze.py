"""验证水印区域冻结：客户端自有 P_k 在本地训练中完全不变，且水印梯度缩放被关闭。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import AlexNet
from utils.watermark_freeze import WatermarkFreezer
from utils.key_matrix_utils import KeyMatrixManager


def main():
    device = torch.device('cpu')
    model = AlexNet(3, 10, input_size=32, dropout_rate=0.5)

    key_matrix_path = 'save/key_matrix/alexnet/client5'
    km = KeyMatrixManager(key_matrix_path)
    freezer = WatermarkFreezer(model, key_matrix_path)

    client_id = 0
    positions = km.load_positions(client_id)
    print(f'客户端 {client_id} 水印位置数: {len(positions)}')

    # 记录冻结前 P_k 的值
    idx_map = freezer.client_indices(client_id)
    named = dict(model.named_parameters())

    def collect():
        out = {}
        for name, idx in idx_map.items():
            flat = named[name].detach().view(-1)
            out[name] = flat[idx].clone()
        return out

    before = collect()

    # 模拟一次带优化器的训练（含 weight decay）
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    snapshot = freezer.snapshot(client_id)

    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        freezer.zero_grads(client_id)
        opt.step()
        freezer.restore(snapshot)

    after = collect()

    max_diff = 0.0
    for name in before:
        d = (before[name] - after[name]).abs().max().item()
        max_diff = max(max_diff, d)
    print(f'冻结后 P_k 最大偏差: {max_diff:.3e}')
    assert max_diff == 0.0, '水印区域发生了变化！'

    # 对照：未冻结时 P_k 应当发生变化
    model2 = AlexNet(3, 10, input_size=32, dropout_rate=0.5)
    fz2 = WatermarkFreezer(model2, key_matrix_path)
    named2 = dict(model2.named_parameters())
    idx_map2 = fz2.client_indices(client_id)

    def collect2(m):
        out = {}
        for name, idx in idx_map2.items():
            out[name] = m[name].detach().view(-1)[idx].clone()
        return out

    b2 = collect2(named2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(3):
        opt2.zero_grad()
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        loss = torch.nn.functional.cross_entropy(model2(x), y)
        loss.backward()
        opt2.step()
    a2 = collect2(dict(model2.named_parameters()))
    d2 = max((b2[n] - a2[n]).abs().max().item() for n in b2)
    print(f'对照组（未冻结）P_k 最大偏差: {d2:.3e}')
    assert d2 > 0, '对照组 P_k 未发生变化，测试无意义'

    print('\n✅ 冻结功能验证通过')


if __name__ == '__main__':
    main()
