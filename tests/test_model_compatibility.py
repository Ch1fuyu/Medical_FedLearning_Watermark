"""密钥矩阵与模型结构兼容性校验的测试。

背景
----
卷积层形状随 ``in_channels`` 变化（chestmnist=1, cifar10/cifar100=3），
而密钥矩阵里的位置索引是按具体形状生成的。用 chestmnist 生成的密钥
矩阵跑 CIFAR10 会导致水印嵌入到错误位置。

本测试验证：
  1. 匹配的组合（模型 + in_channels）通过校验；
  2. 不匹配的组合被正确拒绝，并给出可操作的错误信息；
  3. 校验不依赖 ``num_classes``（FC 层形状与水印位置无关）；
  4. 配置表声明的数据集与实验数据集不一致时被拒绝。
"""
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.args import ArgConfigError, _check_model_compatibility  # noqa: E402
from utils.collusion_presets import (  # noqa: E402
    CollusionPresetError,
    load_presets,
    resolve_spec,
    validate_dataset,
)

#: chestmnist 版本的密钥矩阵目录（in_channels=1）
KM_CHEST = 'save/key_matrix_collusion/alexnet/client10'
#: cifar10 版本的密钥矩阵目录（in_channels=3）
KM_CIFAR = 'save/key_matrix_collusion_cifar10/alexnet/client10'


def _make_args(model_name, in_channels, num_classes, key_matrix_path,
               dataset='chestmnist'):
    return types.SimpleNamespace(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=32 if in_channels == 3 else 28,
        dropout_rate=0.5,
        dataset=dataset,
        key_matrix_path=key_matrix_path,
    )


def _spec(client_num=10, design_K=3, dataset=None):
    return types.SimpleNamespace(client_num=client_num, design_K=design_K,
                                  dataset=dataset)


# ====================================================================
def test_matching_passes():
    print("== test_matching_passes ==")
    args = _make_args('alexnet', 1, 10, KM_CHEST)
    _check_model_compatibility(args, _spec())
    print("  [OK] alexnet + chestmnist (in_ch=1) 通过校验")


# ====================================================================
def test_num_classes_does_not_matter():
    print("== test_num_classes_does_not_matter ==")
    # cifar100 是 100 类，但卷积层形状与 chestmnist 相同 (in_ch=1)
    args = _make_args('alexnet', 1, 100, KM_CHEST)
    _check_model_compatibility(args, _spec())
    print("  [OK] num_classes 变化不影响校验（FC 层与水印位置无关）")


# ====================================================================
def test_mismatch_rejected():
    print("== test_mismatch_rejected ==")
    # chestmnist 生成的目录拿去跑 cifar10
    args = _make_args('alexnet', 3, 10, KM_CHEST, dataset='cifar10')
    try:
        _check_model_compatibility(args, _spec())
        raise AssertionError("in_channels 不匹配本应报错")
    except ArgConfigError as e:
        msg = str(e)
        assert '模型结构不匹配' in msg
        assert 'train_key_matrix.py' in msg, "错误信息应给出修复命令"
        print(f"  [OK] in_channels 不匹配被拒绝: {msg.splitlines()[0][:50]}...")


# ====================================================================
def test_missing_dir_is_skipped():
    print("== test_missing_dir_is_skipped ==")
    args = _make_args('alexnet', 3, 10, 'save/does_not_exist/client10')
    _check_model_compatibility(args, _spec())
    print("  [OK] 目录不存在时跳过校验（由调用方报目录缺失）")


# ====================================================================
def test_validate_dataset_mismatch():
    print("== test_validate_dataset_mismatch ==")
    presets = load_presets()
    spec = resolve_spec(presets, 3, 'alexnet')
    assert spec.dataset == 'chestmnist', \
        f"默认配置表应声明 chestmnist, 实际 {spec.dataset}"

    # 磁盘目录是 chestmnist -> 一致
    validate_dataset(spec, KM_CHEST)
    print("  [OK] 数据集一致时通过")

    # 声明 cifar10 但磁盘是 chestmnist -> 应报错
    bad = resolve_spec(presets, 3, 'alexnet')
    bad.dataset = 'cifar10'
    try:
        validate_dataset(bad, KM_CHEST)
        raise AssertionError("数据集不一致本应报错")
    except CollusionPresetError as e:
        msg = str(e)
        assert '数据集' in msg
        assert 'train_key_matrix.py' in msg
        print("  [OK] 数据集不一致被拒绝")


# ====================================================================
def test_spec_carries_dataset():
    print("== test_spec_carries_dataset ==")
    presets = load_presets()
    for (size, model), spec in presets.items():
        assert spec.dataset, f"(K={size}, {model}) 缺少 dataset 声明"
    print(f"  [OK] 所有条目都声明了 dataset: "
          f"{sorted({s.dataset for s in presets.values()})}")


# ====================================================================
def main():
    test_matching_passes()
    test_num_classes_does_not_matter()
    test_mismatch_rejected()
    test_missing_dir_is_skipped()
    test_validate_dataset_mismatch()
    test_spec_carries_dataset()
    print("\n模型兼容性校验测试全部通过 ✅")


if __name__ == "__main__":
    main()
