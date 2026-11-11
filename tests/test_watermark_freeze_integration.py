"""
水印区域冻结 —— 集成测试（增强模式 TrainerPrivateEnhanced）

验证：
1. 开启冻结后，客户端自有水印区域 P_k 在本地训练全程（含优化器动量、
   weight decay、DP 噪声）保持不变；
2. 冻结模式下水印梯度缩放（capping）不再生效
   （MultiLoss.enable_capping=False，cap_applied 恒为 False）；
3. 非水印区域仍然正常训练（参数确实发生变化）。

运行：
    python tests/test_watermark_freeze_integration.py
"""

import os
import sys
import types

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.alexnet import AlexNet  # noqa: E402
from utils.key_matrix_utils import KeyMatrixManager  # noqa: E402
from utils.trainer_private_enhanced import TrainerPrivateEnhanced  # noqa: E402
from utils.watermark_freeze import WatermarkFreezer  # noqa: E402


def _make_args(key_matrix_path, freeze=True, dp=False, sigma=0.1):
    args = types.SimpleNamespace()
    args.use_key_matrix = True
    args.key_matrix_path = key_matrix_path
    args.watermark_mode = 'enhanced'
    args.watermark_grad_target_ratio = 0.3
    args.freeze_watermark_region = freeze
    args.dp = dp
    args.sigma = sigma
    args.optim = 'adam'
    args.wd = 1e-5
    args.lr = 0.001
    args.use_lr_scheduler = False
    args.task_type = 'multiclass'
    args.class_weights = False
    args.pos_weight_factor = 2.0
    args.focal_gamma = 1.0
    args.focal_reduction = 'mean'
    return args


def _wm_values(model, index_map):
    named = dict(model.named_parameters())
    out = {}
    for name, idx in index_map.items():
        out[name] = named[name].detach().cpu().view(-1)[idx].clone()
    return out


def _max_diff(a, b):
    return max((a[n] - b[n]).abs().max().item() for n in a)


def main():
    key_matrix_path = 'save/key_matrix/alexnet/client5'
    km = KeyMatrixManager(key_matrix_path)
    client_id = 0

    model = AlexNet(3, 10, input_size=32, dropout_rate=0.5)
    args = _make_args(key_matrix_path, freeze=True, dp=True, sigma=0.1)
    trainer = TrainerPrivateEnhanced(model, torch.device('cpu'), args.dp, args.sigma, args)

    assert trainer.freeze_watermark_region, '冻结模式未启用'
    assert trainer.multi_loss.enable_capping is False, 'capping 未被关闭'

    index_map = trainer.watermark_freezer.client_indices(client_id)

    # 记录训练前的 P_k 与非水印区参数
    before_wm = _wm_values(model, index_map)
    named = dict(model.named_parameters())
    before_nonwm = {n: p.detach().cpu().clone() for n, p in named.items()}

    # 构造合成数据
    data = torch.utils.data.TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
    loader = torch.utils.data.DataLoader(data, batch_size=16)

    state, loss, acc = trainer.local_update(
        dataloader=loader, local_ep=2, lr=0.01, client_id=client_id,
        current_epoch=1, total_epochs=10,
    )

    after_wm = _wm_values(model, index_map)
    diff = _max_diff(before_wm, after_wm)
    print(f'[冻结模式] P_k 最大偏差: {diff:.3e}')
    assert diff == 0.0, f'冻结模式下 P_k 发生了变化（偏差 {diff:.3e}）'

    # 非水印区域应当确实训练了
    after_nonwm = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    nonwm_changed = any(
        not torch.equal(before_nonwm[n], after_nonwm[n])
        for n in before_nonwm if n not in index_map
    )
    print(f'[冻结模式] 非水印区参数发生变化: {nonwm_changed}')
    assert nonwm_changed, '非水印区域没有变化，测试无意义'

    # 对照组：关闭冻结时 P_k 应当发生变化
    model2 = AlexNet(3, 10, input_size=32, dropout_rate=0.5)
    args2 = _make_args(key_matrix_path, freeze=False, dp=True, sigma=0.1)
    trainer2 = TrainerPrivateEnhanced(model2, torch.device('cpu'), args2.dp, args2.sigma, args2)
    assert trainer2.multi_loss.enable_capping is True, '对照组的 capping 应保持开启'

    index_map2 = WatermarkFreezer(model2, key_matrix_path).client_indices(client_id)
    before2 = _wm_values(model2, index_map2)
    trainer2.local_update(
        dataloader=loader, local_ep=2, lr=0.01, client_id=client_id,
        current_epoch=1, total_epochs=10,
    )
    after2 = _wm_values(model2, index_map2)
    diff2 = _max_diff(before2, after2)
    print(f'[对照组]  P_k 最大偏差: {diff2:.3e}')
    assert diff2 > 0, '对照组 P_k 未发生变化，测试无意义'

    print('\n✅ 集成测试通过：冻结模式生效，capping 已关闭，非水印区正常训练')


if __name__ == '__main__':
    main()
