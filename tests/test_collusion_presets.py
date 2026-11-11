"""抗合谋配置表（(K, 模型) -> 密钥矩阵规格）的解析与校验测试。

覆盖：
  1. 默认配置表可加载, 各条目字段合法
  2. 按 (K, 模型) 取规格, 缺失时给出可操作的错误信息
  3. 配置表内容非法时明确报错（而不是静默通过）
  4. 目录定位正确
  5. 布局与配置表一致时通过; design_K 冲突时报错
  6. 模型类型不会被配置表覆盖（模型是显式选择）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.collusion_presets import (  # noqa: E402
    DEFAULT_PRESET_PATH,
    SEARCH_MARGIN,
    CollusionPresetError,
    load_presets,
    resolve_key_matrix_path,
    resolve_spec,
    validate_layout,
)


# ====================================================================
def test_load_default_presets():
    print("== test_load_default_presets ==")
    presets = load_presets()
    assert presets, "默认配置表为空"
    for (size, model), spec in presets.items():
        assert spec.size == size
        assert spec.model_name == model
        assert spec.client_num >= 2, f"K={size}/{model} 的 client_num 非法"
        assert spec.size <= spec.client_num, f"K={size}/{model} 超过客户端数"
        assert model in ('resnet', 'alexnet'), \
            f"K={size} 的 model_name 非法: {model}"
    sizes = sorted({s for (s, _) in presets})
    models = sorted({m for (_, m) in presets})
    print(f"  已配置 K = {sizes}")
    print(f"  已配置模型 = {models}")
    print(f"  默认配置表路径 = {DEFAULT_PRESET_PATH}")
    print(f"  搜索余量 SEARCH_MARGIN = {SEARCH_MARGIN}")
    print("  [OK] 默认配置表合法")


# ====================================================================
def test_resolve_spec():
    print("== test_resolve_spec ==")
    presets = load_presets()
    size = sorted({s for (s, _) in presets})[0]
    spec = resolve_spec(presets, size, 'resnet')
    assert spec.size == size
    assert spec.model_name == 'resnet'
    print(f"  ({size}, resnet) -> {spec}")

    # 同一 K 下不同模型应解析到不同规格
    spec_a = resolve_spec(presets, size, 'alexnet')
    assert spec_a.model_name == 'alexnet'
    print(f"  ({size}, alexnet) -> {spec_a}")

    # 缺失的 K
    try:
        resolve_spec(presets, 9999, 'resnet')
        raise AssertionError("缺失的 K 本应报错")
    except CollusionPresetError as e:
        msg = str(e)
        assert "没有合谋规模" in msg or "没有 (合谋规模" in msg
        assert "train_key_matrix.py" in msg, "错误信息应给出修复命令"
        print("  [OK] 缺失 K 时报错信息可操作")

    # 缺失的模型（该 K 存在但模型不存在）
    presets2 = load_presets('{"3": {"resnet": {"client_num": 10}}}')
    try:
        resolve_spec(presets2, 3, 'alexnet')
        raise AssertionError("缺失的模型本应报错")
    except CollusionPresetError as e:
        msg = str(e)
        assert "alexnet" in msg
        assert "已配置的模型" in msg, "错误信息应列出可用模型"
        print("  [OK] 缺失模型时错误信息可操作")


# ====================================================================
def test_invalid_presets():
    print("== test_invalid_presets ==")
    cases = [
        ("键非整数", '{"abc": {"resnet": {"client_num": 10}}}'),
        ("模型条目缺字段", '{"3": {"resnet": {}}}'),
        ("client_num 过小",
         '{"3": {"resnet": {"client_num": 1}}}'),
        ("合谋规模超客户端数",
         '{"3": {"resnet": {"client_num": 2}}}'),
        ("design_K 非法",
         '{"3": {"resnet": {"client_num": 10, "design_K": 0}}}'),
        ("模型条目非对象", '{"3": {"resnet": 5}}'),
        ("K 下不是模型分组", '{"3": 5}'),
        ("未知模型类型", '{"3": {"vgg": {"client_num": 10}}}'),
        ("不是 JSON 对象", '[1, 2, 3]'),
    ]
    for name, raw in cases:
        try:
            load_presets(raw)
            raise AssertionError(f"{name} 本应报错")
        except CollusionPresetError:
            print(f"  [OK] {name} 被正确拒绝")

    # 空表
    try:
        load_presets('{}')
        raise AssertionError("空表本应报错")
    except CollusionPresetError:
        print("  [OK] 空配置表被正确拒绝")

    # JSON 语法错误
    try:
        load_presets('{bad')
        raise AssertionError("非法 JSON 本应报错")
    except CollusionPresetError:
        print("  [OK] 非法 JSON 被正确拒绝")

    # 文件不存在
    try:
        load_presets('config/definitely_not_exist.json')
        raise AssertionError("不存在的文件本应报错")
    except CollusionPresetError:
        print("  [OK] 不存在的配置文件被正确拒绝")


# ====================================================================
def test_key_matrix_path():
    print("== test_key_matrix_path ==")
    presets = load_presets()
    size = sorted({s for (s, _) in presets})[0]
    for model in ('resnet', 'alexnet'):
        spec = resolve_spec(presets, size, model)
        path = resolve_key_matrix_path('save/key_matrix_collusion', spec)
        expected = (f"save/key_matrix_collusion/{model}/"
                    f"client{spec.client_num}")
        assert path.replace('\\', '/') == expected, f"{path} != {expected}"
        print(f"  [OK] {path}")


# ====================================================================
class _FakeDesign:
    def __init__(self, k):
        self.K = k


class _FakeLayout:
    def __init__(self, k):
        self.design = _FakeDesign(k)


def test_validate_layout():
    print("== test_validate_layout ==")
    presets = load_presets()
    size = sorted({s for (s, _) in presets})[0]
    spec = resolve_spec(presets, size, 'resnet')

    # 1) design.K 一致 -> 通过
    warning = validate_layout(_FakeLayout(spec.design_K), spec)
    assert warning is None, f"不应有告警: {warning}"
    print("  [OK] design.K 一致时通过")

    # 2) design.K 冲突 -> 报错
    bad = resolve_spec(presets, size, 'resnet')
    bad.design_K = (spec.design_K or 3) + 4
    try:
        validate_layout(_FakeLayout(spec.design_K), bad)
        raise AssertionError("design_K 冲突本应报错")
    except CollusionPresetError as e:
        assert "不一致" in str(e)
        print("  [OK] design.K 与布局冲突时明确报错")

    # 3) 布局缺失 -> 报错
    try:
        validate_layout(None, spec)
        raise AssertionError("布局缺失本应报错")
    except CollusionPresetError as e:
        assert "缺少 witness_layout.json" in str(e)
        print("  [OK] 布局缺失时明确报错")

    # 4) design.K 偏小导致上界覆盖不了 K -> 返回告警文案
    small = resolve_spec(presets, size, 'resnet')
    small.size = 5          # K=5
    small.design_K = None
    layout = _FakeLayout(1)  # design.K=1 -> 上界 = 3 < 5
    warning = validate_layout(layout, small)
    assert warning is not None and "放宽" in warning, \
        f"应给出放宽告警, 实际 {warning}"
    assert small.design_K == 1, "校验后应把磁盘 design.K 写回规格"
    print("  [OK] design.K 偏小时给出放宽告警")


# ====================================================================
def test_spec_copy_isolation():
    print("== test_spec_copy_isolation ==")
    presets = load_presets()
    size = sorted({s for (s, _) in presets})[0]
    a = resolve_spec(presets, size, 'resnet')
    a.design_K = 999
    b = resolve_spec(presets, size, 'resnet')
    assert b.design_K != 999, "规格拷贝应相互隔离, 不能污染配置表"
    print("  [OK] 规格拷贝相互隔离")


# ====================================================================
def test_search_cap_matches_main():
    print("== test_search_cap_matches_main ==")
    from utils.collusion_presets import SEARCH_MARGIN as m
    assert m == 2, f"搜索余量应为 2, 实际 {m}"
    print(f"  [OK] SEARCH_MARGIN = {m}, 与 main.py 一致")


# ====================================================================
def main():
    test_load_default_presets()
    test_resolve_spec()
    test_invalid_presets()
    test_key_matrix_path()
    test_validate_layout()
    test_spec_copy_isolation()
    test_search_cap_matches_main()
    print("\n抗合谋配置表测试全部通过 ✅")


if __name__ == "__main__":
    main()
