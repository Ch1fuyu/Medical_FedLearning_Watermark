"""--collusion_only 模式的联动逻辑测试。

验证一次开启以下全部开关：
  1. 关闭单客户端泄漏仿真
  2. 关闭泄漏源追踪
  3. 开启合谋检测
  4. 密钥矩阵切到 collusion 目录
以及不带该参数时的行为不受影响。
"""
import os
import subprocess
import sys

ROOT = r"D:\CODE\Medical_FedLearning_Watermark"
PY = r"D:\Environment\anaconda\envs\Med_FL_WM\python.exe"

PROBE = """
import sys
sys.path.insert(0, r'{root}')
sys.argv = ['x'] + {extra!r} + ['--client_num', '5', '--model_name', 'alexnet',
                              '--dataset', {dataset!r}]
from utils.args import parser_args
a = parser_args()
print('R|' + repr({{
    'disable_single_client_leak': a.disable_single_client_leak,
    'enable_source_tracing': a.enable_source_tracing,
    'enable_collusion_detection': a.enable_collusion_detection,
    'key_matrix_profile': a.key_matrix_profile,
    'key_matrix_path': a.key_matrix_path,
}}))
"""


def probe(extra, dataset='chestmnist'):
    code = PROBE.format(root=ROOT, extra=extra, dataset=dataset)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       cwd=ROOT, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        if line.startswith("R|"):
            return eval(line[2:])
    raise RuntimeError("probe failed:\n" + r.stdout + "\n" + r.stderr)


def test_collusion_only():
    print("== test_collusion_only ==")
    d = probe(["--collusion_only"])
    print("  %s" % d)
    assert d["disable_single_client_leak"] is True, "未关闭单客户端泄漏"
    assert d["enable_source_tracing"] is False, "未关闭泄漏源追踪"
    assert d["enable_collusion_detection"] is True, "未开启合谋检测"
    assert d["key_matrix_profile"] == "collusion", "未切换 profile"
    assert "key_matrix_collusion" in d["key_matrix_path"], "路径未切到 collusion 目录"
    print("  [OK] 四个开关全部生效")


def test_default_unchanged():
    print("== test_default_unchanged ==")
    d = probe([])
    print("  %s" % d)
    assert d["disable_single_client_leak"] is False
    assert d["enable_collusion_detection"] is False
    assert d["key_matrix_profile"] == "default"
    assert d["key_matrix_path"] == "save/key_matrix/alexnet/client5"
    print("  [OK] 不带参数时保持原行为（传统水印基线）")


def test_explicit_override():
    print("== test_explicit_override ==")
    # 显式指定 key_matrix_dir + 配置表时，collusion_only 不应覆盖它们。
    # 注意：显式目录必须与配置表声明的数据集匹配，否则会被数据集校验拒绝。
    d = probe(["--collusion_only",
               "--key_matrix_dir", "save/key_matrix_collusion_cifar10",
               "--collusion_size_configs",
               "config/collusion_presets_cifar10.json"], dataset='cifar10')
    print("  %s" % d)
    assert d["key_matrix_path"].startswith("save/key_matrix_collusion_cifar10"), \
        "显式 key_matrix_dir 被覆盖了"
    assert d["disable_single_client_leak"] is True
    assert d["enable_collusion_detection"] is True
    print("  [OK] 显式 key_matrix_dir 优先级更高，未被覆盖")


def test_explicit_missing_dir_exits():
    print("== test_explicit_missing_dir_exits ==")
    # 显式指定了不存在的目录 -> 应该退出并说明，而不是带着坏配置继续跑
    r = subprocess.run(
        [PY, "-c", PROBE.format(root=ROOT, dataset='chestmnist',
                                 extra=["--collusion_only",
                                        "--key_matrix_dir", "save/mykeymatrix"])],
        capture_output=True, text=True, cwd=ROOT,
        encoding="utf-8", errors="replace")
    assert r.returncode != 0, "目录不存在时应以非零码退出"
    assert "配置错误" in r.stderr, "应给出明确的配置错误提示"
    assert "train_key_matrix.py" in r.stderr, "应给出修复命令"
    print("  [OK] 目录不存在时退出并说明原因")


def test_dataset_mismatch_exits():
    print("== test_dataset_mismatch_exits ==")
    # 目录是 cifar10 生成的，但用了 chestmnist 配置表 -> 应退出并说明
    r = subprocess.run(
        [PY, "-c", PROBE.format(root=ROOT, dataset='chestmnist',
                                 extra=["--collusion_only",
                                        "--key_matrix_dir",
                                        "save/key_matrix_collusion_cifar10"])],
        capture_output=True, text=True, cwd=ROOT,
        encoding="utf-8", errors="replace")
    assert r.returncode != 0, "数据集不一致时应以非零码退出"
    assert "数据集" in r.stderr, "应说明是数据集不匹配"
    print("  [OK] 数据集不匹配时退出并说明原因")


def main():
    test_collusion_only()
    test_default_unchanged()
    test_explicit_override()
    test_explicit_missing_dir_exits()
    test_dataset_mismatch_exits()
    print("\ncollusion_only 模式测试全部通过 ✅")


if __name__ == "__main__":
    main()
