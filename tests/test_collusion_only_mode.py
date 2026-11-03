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
sys.argv = ['x'] + {extra!r} + ['--client_num', '5', '--model_name', 'alexnet']
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


def probe(extra):
    code = PROBE.format(root=ROOT, extra=extra)
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
    # 显式指定 key_matrix_dir 时，collusion_only 不应覆盖它
    d = probe(["--collusion_only", "--key_matrix_dir", "save/mykeymatrix"])
    print("  %s" % d)
    assert d["key_matrix_path"].startswith("save/mykeymatrix"), \
        "显式 key_matrix_dir 被覆盖了"
    assert d["disable_single_client_leak"] is True
    print("  [OK] 显式 key_matrix_dir 优先级更高，未被覆盖")


def main():
    test_collusion_only()
    test_default_unchanged()
    test_explicit_override()
    print("\ncollusion_only 模式测试全部通过 ✅")


if __name__ == "__main__":
    main()
