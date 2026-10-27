# -*- coding: utf-8 -*-
"""
水印梯度 capping 单元测试
=========================

覆盖 ``models/losses/multi_loss.py`` 中
``compute_loss_and_backward`` 的水印梯度 capping 逻辑，验证：

1. **符号保持**：缩放后水印区梯度符号不得被翻转
    （历史 bug：用 sign() 拆分正负再重组，导致负梯度变正）
2. **目标达成**：当缩放系数落在 [0.1, 1.0] 区间内时，
   实际达成的 wm/non_wm 比例应精确等于 target_ratio
3. **缩放下限**：scale_factor 下限为 0.1（有意设计），
   当目标比例远低于原比例时，实际达成比例 = 0.1 × 原比例
4. **非水印区不受影响**：非水印区梯度必须原样保留
5. **未超阈值时不缩放**：wm_ratio <= target_ratio 时梯度不变
6. **target_ratio=0 关闭 capping**：梯度完全不变
7. **恶意客户端跳过缩放**：skip_gradient_scaling=True 时梯度不变
8. **缩放等比**：水印区所有元素乘以同一系数

运行（无需安装 pytest，兼容 pytest）:
    python tests/test_watermark_grad_capping.py
或:
    python -m pytest tests/test_watermark_grad_capping.py -v
"""

import os
import sys
import traceback

import torch
import torch.nn as nn

# 保证可以从项目根目录导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.losses.multi_loss import MultiLoss  # noqa: E402


class _TinyConvNet(nn.Module):
    """
    参数名中同时包含 'conv' 与 'weight' 的最小模型，
    以便触发 MultiLoss 中的卷积层梯度收集逻辑。

    conv1: 3->4, 3x3 => 108 个参数（用作水印区）
    conv2: 4->4, 1x1 =>  16 个参数（用作非水印区）
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, bias=False)
        self.conv2 = nn.Conv2d(4, 4, 1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


N_WM = 108
N_NON = 16


def _run_capping(wm_values, non_wm_values, target_ratio, skip_gradient_scaling=False):
    """
    驱动一次 MultiLoss capping，返回 (原始梯度, capping 后梯度, MultiLoss 实例)
    """
    model = _TinyConvNet()
    multi_loss = MultiLoss(model=model, target_ratio=target_ratio)
    multi_loss.set_device(torch.device('cpu'))

    n_wm = model.conv1.weight.numel()
    n_non = model.conv2.weight.numel()
    assert len(wm_values) == n_wm
    assert len(non_wm_values) == n_non

    # 有效掩码：conv1 全部为水印区，conv2 全部为非水印区
    effective_mask = torch.zeros(n_wm + n_non)
    effective_mask[:n_wm] = 1.0
    target_mask = torch.ones(n_wm + n_non)
    encoder_mask = effective_mask.clone()

    grads = torch.tensor(wm_values + non_wm_values, dtype=torch.float32)

    with torch.no_grad():
        model.conv1.weight.grad = grads[:n_wm].view_as(model.conv1.weight).clone()
        model.conv2.weight.grad = grads[n_wm:].view_as(model.conv2.weight).clone()

    loss = torch.tensor(1.0, requires_grad=True)
    multi_loss.compute_loss_and_backward(
        loss,
        target_mask,
        encoder_mask,
        effective_mask,
        current_epoch=1,        # 必须 > 0，才会走 capping 分支
        total_epochs=10,
        skip_gradient_scaling=skip_gradient_scaling,
    )

    out = torch.cat([
        model.conv1.weight.grad.view(-1),
        model.conv2.weight.grad.view(-1),
    ])
    return grads, out, multi_loss


# --------------------------------------------------------------------------- #
# 1. 符号保持
# --------------------------------------------------------------------------- #

def test_watermark_gradient_sign_is_preserved():
    """缩放后水印区梯度的符号必须与原始梯度一致（负梯度不得翻正）"""
    wm = [2.0, -2.0] + [2.0] * (N_WM - 2)
    nonwm = [1.0] * N_NON

    original, corrected, ml = _run_capping(wm, nonwm, target_ratio=0.3)

    # 确认 capping 确实被触发
    assert ml.current_wm_ratio > 0.3, "capping 未触发，测试前提不成立"

    wm_orig = original[:N_WM]
    wm_new = corrected[:N_WM]

    sign_orig = torch.sign(wm_orig)
    sign_new = torch.sign(wm_new)
    assert torch.equal(sign_orig, sign_new), (
        "水印区梯度符号被翻转："
        f"原前 4 个 {[f'{v:+.2f}' for v in wm_orig[:4].tolist()]} "
        f"-> 新 {[f'{v:+.2f}' for v in wm_new[:4].tolist()]}"
    )


def test_negative_gradient_stays_negative():
    """显式检查一个负梯度条目缩放后仍为负"""
    wm = [2.0, -2.0] + [2.0] * (N_WM - 2)
    nonwm = [1.0] * N_NON

    original, corrected, _ = _run_capping(wm, nonwm, target_ratio=0.3)

    assert original[1].item() < 0
    assert corrected[1].item() < 0, "负梯度被翻转为正梯度"
    assert abs(corrected[1].item()) < abs(original[1].item()), "负梯度幅度未被缩小"


# --------------------------------------------------------------------------- #
# 2. 目标达成
# --------------------------------------------------------------------------- #

def test_target_ratio_is_actually_achieved():
    """
    当所需缩放系数落在 [0.1, 1.0] 内时，实际达成的比例应精确等于 target_ratio。

    scale_factor = (target_ratio * non_wm_mean / wm_mean).clamp(0.1, 1.0)
    即：原比例 R = wm_mean / non_wm_mean，所需 scale = target_ratio / R。
    当 target_ratio / R >= 0.1 时 clamp 不生效，目标被精确达成。
    """
    wm = [2.0] * N_WM          # 原比例 R = 2.0
    nonwm = [1.0] * N_NON

    # target_ratio / 2.0 >= 0.1  =>  target_ratio >= 0.2
    for target_ratio in [0.5, 0.4, 0.3, 0.2]:
        original, corrected, ml = _run_capping(wm, nonwm, target_ratio=target_ratio)

        new_wm_mean = corrected[:N_WM].abs().mean().item()
        new_nonwm_mean = corrected[N_WM:].abs().mean().item()
        achieved = new_wm_mean / new_nonwm_mean

        assert abs(ml.cap_scale_factor - target_ratio / 2.0) < 1e-6, (
            f"target_ratio={target_ratio} 时 scale 应为 {target_ratio / 2.0:.4f}，"
            f"实际={ml.cap_scale_factor:.6f}"
        )
        assert abs(achieved - target_ratio) < 0.01, (
            f"target_ratio={target_ratio} 期望达成该比例，实际={achieved:.4f}"
        )


def test_achieved_ratio_is_floor_times_original_when_floor_active():
    """
    当 clamp 下限 0.1 生效时，实际达成比例 = 0.1 × 原比例。

    手工推算：
      - R = 10.0，target = 0.5  => 所需 scale = 0.05 < 0.1 => 截断为 0.1 => 达成 1.0
      - R = 10.0，target = 0.1  => 所需 scale = 0.01 < 0.1 => 截断为 0.1 => 达成 1.0
      - R =  2.0，target = 0.1  => 所需 scale = 0.05 < 0.1 => 截断为 0.1 => 达成 0.2
    """
    cases = [
        ([10.0] * N_WM, [1.0] * N_NON, 0.5, 1.0),
        ([10.0] * N_WM, [1.0] * N_NON, 0.1, 1.0),
        ([2.0] * N_WM, [1.0] * N_NON, 0.1, 0.2),
    ]

    for wm, nonwm, target_ratio, expected in cases:
        _, corrected, ml = _run_capping(wm, nonwm, target_ratio=target_ratio)

        new_wm_mean = corrected[:N_WM].abs().mean().item()
        new_nonwm_mean = corrected[N_WM:].abs().mean().item()
        achieved = new_wm_mean / new_nonwm_mean

        assert ml.cap_applied is True, "capping 应已应用"
        assert abs(ml.cap_scale_factor - 0.1) < 1e-6, (
            f"scale_factor 应被 clamp 到 0.1，实际={ml.cap_scale_factor:.6f}"
        )
        assert abs(achieved - expected) < 0.01, (
            f"target_ratio={target_ratio} 期望达成 {expected}，实际={achieved:.4f}"
        )


def test_scale_factor_lower_bound_is_0p1():
    """
    scale_factor 下限 0.1 是有意设计：最多把水印区梯度压到原来的 10%。

    当目标远低于原比例时，clamp 生效，实际达成比例 = 0.1 × 原比例。
    """
    wm = [10.0] * N_WM
    nonwm = [1.0] * N_NON

    _, _, ml = _run_capping(wm, nonwm, target_ratio=0.05)

    assert ml.cap_applied is True
    assert abs(ml.cap_scale_factor - 0.1) < 1e-6, (
        f"scale_factor 应被 clamp 到下限 0.1，实际={ml.cap_scale_factor:.6f}"
    )
    # 原比例 10.0，被压到 10% => 1.0
    assert abs(ml.current_wm_ratio_after_cap - 1.0) < 0.01, (
        f"实际达成比例应为 1.0，实际={ml.current_wm_ratio_after_cap:.4f}"
    )


def test_achieved_ratio_tracks_target_across_range():
    """
    在 clamp 不生效的区间内，达成比例随目标线性变化且精确。

    原比例固定为 2.0，所需 scale = target_ratio / 2；
    target_ratio >= 0.2 时 scale >= 0.1，clamp 不生效。
    """
    wm = [2.0] * N_WM
    nonwm = [1.0] * N_NON

    for target_ratio in [0.6, 0.4, 0.3, 0.2]:
        _, corrected, ml = _run_capping(wm, nonwm, target_ratio=target_ratio)
        achieved = corrected[:N_WM].abs().mean().item() / corrected[N_WM:].abs().mean().item()
        assert ml.cap_scale_factor >= 0.1 - 1e-9, "前提不成立：clamp 下限应未生效"
        assert abs(achieved - target_ratio) < 0.01, (
            f"target_ratio={target_ratio} 未达成，实际={achieved:.4f}"
        )


# --------------------------------------------------------------------------- #
# 3. 非水印区不受影响
# --------------------------------------------------------------------------- #

def test_non_watermark_region_is_untouched():
    """非水印区梯度必须逐元素原样保留"""
    wm = [2.0, -2.0] + [2.0] * (N_WM - 2)
    nonwm = [1.0, -1.0, 0.5, -0.5] + [1.0] * (N_NON - 4)

    original, corrected, _ = _run_capping(wm, nonwm, target_ratio=0.3)

    assert torch.equal(original[N_WM:], corrected[N_WM:]), "非水印区梯度被改动"


# --------------------------------------------------------------------------- #
# 4. 未超阈值时不缩放
# --------------------------------------------------------------------------- #

def test_no_scaling_when_below_threshold():
    """wm_ratio <= target_ratio 时不应发生任何缩放"""
    wm = [0.1] * N_WM
    nonwm = [1.0] * N_NON

    original, corrected, ml = _run_capping(wm, nonwm, target_ratio=0.3)

    assert ml.current_wm_ratio < 0.3, "测试前提不成立：wm_ratio 应低于阈值"
    assert torch.allclose(original, corrected), "低于阈值时梯度不应被修改"


# --------------------------------------------------------------------------- #
# 5. target_ratio = 0 关闭 capping
# --------------------------------------------------------------------------- #

def test_target_ratio_zero_disables_capping():
    """target_ratio=0 时完全关闭 capping，梯度原样保留"""
    wm = [2.0, -2.0] + [2.0] * (N_WM - 2)
    nonwm = [1.0] * N_NON

    original, corrected, _ = _run_capping(wm, nonwm, target_ratio=0.0)

    assert torch.allclose(original, corrected), "target_ratio=0 时梯度不应被修改"


# --------------------------------------------------------------------------- #
# 6. 恶意客户端跳过缩放
# --------------------------------------------------------------------------- #

def test_malicious_client_skips_scaling():
    """skip_gradient_scaling=True 时梯度原样保留"""
    wm = [2.0, -2.0] + [2.0] * (N_WM - 2)
    nonwm = [1.0] * N_NON

    original, corrected, _ = _run_capping(
        wm, nonwm, target_ratio=0.3, skip_gradient_scaling=True
    )

    assert torch.allclose(original, corrected), "恶意客户端不应被缩放梯度"


# --------------------------------------------------------------------------- #
# 7. 缩放是等比的（不改变水印区梯度分布形状）
# --------------------------------------------------------------------------- #

def test_scaling_is_proportional():
    """水印区内所有梯度应乘以同一个系数（保持相对关系）"""
    wm = [2.0, -2.0, 4.0, -4.0] + [2.0] * (N_WM - 4)
    nonwm = [1.0] * N_NON

    original, corrected, _ = _run_capping(wm, nonwm, target_ratio=0.3)

    nonzero = original[:N_WM].abs() > 1e-12
    ratios = corrected[:N_WM][nonzero] / original[:N_WM][nonzero]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-5), (
        "水印区各元素缩放系数不一致，缩放不是等比的"
    )


# --------------------------------------------------------------------------- #
# 8. capping 后比例的统计字段
# --------------------------------------------------------------------------- #

def test_after_cap_ratio_field_is_tracked():
    """MultiLoss 应记录 capping 之后的实际 wm/non_wm 比例"""
    wm = [2.0] * N_WM
    nonwm = [1.0] * N_NON

    _, _, ml = _run_capping(wm, nonwm, target_ratio=0.3)

    assert ml.cap_applied is True, "应记录 capping 已被应用"
    assert abs(ml.current_wm_ratio_after_cap - 0.3) < 0.01, (
        f"capping 后比例应为 0.3，实际={ml.current_wm_ratio_after_cap:.4f}"
    )
    stats = ml.get_stats()
    assert 'current_wm_ratio_after_cap' in stats
    assert 'cap_applied' in stats


def test_after_cap_ratio_when_cap_disabled():
    """target_ratio=0 时不应应用 capping，记录的比例应等于原始比例"""
    wm = [10.0] * N_WM
    nonwm = [1.0] * N_NON

    _, _, ml = _run_capping(wm, nonwm, target_ratio=0.0)

    assert ml.cap_applied is False, "target_ratio=0 不应应用 capping"
    # 未 capping 时记录值应与原始比例一致
    assert abs(ml.current_wm_ratio_after_cap - ml.current_wm_ratio) < 1e-6


def test_after_cap_ratio_not_exceed_target():
    """
    当 clamp 下限不生效时，capping 之后的水印比例不应超过目标值。

    原比例 2.0，所需 scale = target_ratio / 2 >= 0.1（即 target_ratio >= 0.2），
    此时 clamp 不生效，达成比例精确等于 target_ratio。
    """
    wm = [2.0] * N_WM
    nonwm = [1.0] * N_NON

    for target_ratio in [0.2, 0.3, 0.5]:
        _, _, ml = _run_capping(wm, nonwm, target_ratio=target_ratio)
        assert ml.cap_scale_factor >= 0.1 - 1e-9, "前提不成立：clamp 下限应未生效"
        assert ml.current_wm_ratio_after_cap <= target_ratio * 1.02, (
            f"capping 后比例 {ml.current_wm_ratio_after_cap:.4f} 超过目标 {target_ratio}"
        )


def test_after_cap_ratio_may_exceed_target_when_floor_active():
    """
    当 clamp 下限生效时，实际达成比例可能高于 target_ratio（这是下限 0.1 的已知后果）。

    原比例 10.0，目标 0.1 所需 scale = 0.01 < 0.1，被截断到 0.1，
    实际达成比例 = 0.1 × 10.0 = 1.0 > 0.1。
    """
    wm = [10.0] * N_WM
    nonwm = [1.0] * N_NON

    _, _, ml = _run_capping(wm, nonwm, target_ratio=0.1)

    assert ml.cap_applied is True
    assert abs(ml.cap_scale_factor - 0.1) < 1e-6, (
        f"scale_factor 应被 clamp 到 0.1，实际={ml.cap_scale_factor:.6f}"
    )
    assert ml.current_wm_ratio_after_cap > 0.1, (
        "下限生效时达成比例应高于目标（本例为 1.0）"
    )
    assert abs(ml.current_wm_ratio_after_cap - 1.0) < 0.01, (
        f"实际达成比例应为 1.0，实际={ml.current_wm_ratio_after_cap:.4f}"
    )


# --------------------------------------------------------------------------- #
# 独立运行器（无需 pytest）
# --------------------------------------------------------------------------- #

def _main():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]

    print('=' * 72)
    print('水印梯度 capping 单元测试')
    print('=' * 72)

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f'[FAIL] {name}')
            print(f'       {exc}')
            if os.environ.get('WM_TEST_TRACE'):
                traceback.print_exc()
        else:
            passed += 1
            print(f'[PASS] {name}')

    print('-' * 72)
    print(f'总计: {len(tests)} | 通过: {passed} | 失败: {failed}')
    print('=' * 72)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
