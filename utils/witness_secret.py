"""见证点密钥的持久化管理。

背景
----
抗合谋水印的见证点权威值 e_p 与基线值 b_p 由**服务器密钥单向派生**：

    e_p = PRF(secret, "witness/e" || p)
    b_p = PRF(secret, "witness/b" || p)

密钥一旦确定就**长期不变**：同一密钥生成的所有 ``witness_layout.json``
互相兼容；换密钥则新旧布局的水印互不兼容（解码必然失败）。
因此密钥必须落盘保存，生成布局时默认从固定路径读取，
而不是每次临时随机生成（那会让每次生成的布局彼此对不上）。

默认路径
--------
    save/keys/witness_secret_collusion.key

``save/`` 已在 .gitignore 中，密钥不会进入版本库；同时密钥与它派生的
布局文件放在一起，删除 ``save/`` 时两者一起消失，不会出现
"布局还在、密钥已丢、无法复现"的半残状态。

安全提示
--------
密钥是服务器侧权威值的唯一来源。泄露它意味着攻击者可以伪造任意
客户端的见证点签名（陷害），因此：

- 不要把密钥提交到 git；
- 不要通过命令行明文传递（用 ``--witness_secret_file``）；
- 定期用 ``secret_fingerprint`` 核对布局与密钥是否匹配。
"""

import hashlib
import os
import secrets as _secrets
from typing import Optional, Tuple

#: 抗合谋分支的默认密钥文件路径（相对项目根目录）。
DEFAULT_SECRET_PATH = os.path.join('save', 'keys',
                                   'witness_secret_collusion.key')

#: 密钥的最小字节长度（与 build_witness_layout 的校验保持一致）。
MIN_SECRET_BYTES = 16


def secret_fingerprint(secret: bytes) -> str:
    """返回密钥的指纹（sha256 前 16 位十六进制）。

    与 ``WitnessLayout.secret_id`` 使用同一算法，因此可以直接用来
    核对"某个布局是不是由当前密钥生成的"。

    Args:
        secret: 密钥字节。

    Returns:
        16 位十六进制字符串。
    """
    return hashlib.sha256(secret).hexdigest()[:16]


def parse_secret(raw: str) -> bytes:
    """把命令行/文件中的密钥文本解析为字节。

    优先按十六进制解析（``bytes.fromhex``），失败则按 UTF-8 文本处理。
    长度不足 16 字节时补零到 16 字节（与 train_key_matrix.py 的
    历史行为保持一致）。

    Args:
        raw: 密钥文本（十六进制或任意文本）。

    Returns:
        密钥字节（长度 >= 16）。
    """
    raw = raw.strip()
    try:
        out = bytes.fromhex(raw)
    except ValueError:
        out = raw.encode("utf-8")
    if len(out) < MIN_SECRET_BYTES:
        out = out.ljust(MIN_SECRET_BYTES, b"\x00")
    return out


def load_secret(path: str) -> bytes:
    """从文件读取密钥。

    Args:
        path: 密钥文件路径（内容为十六进制或任意文本）。

    Returns:
        密钥字节。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    with open(path, 'r', encoding='utf-8') as f:
        return parse_secret(f.read())


def save_secret(secret: bytes, path: str) -> str:
    """把密钥写入文件（十六进制文本）。

    Args:
        secret: 密钥字节。
        path: 目标路径。

    Returns:
        写入的路径。
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(secret.hex())
    # 尽力收紧权限（Windows 上仅能到只读级别，POSIX 上是 0600）
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_or_create_secret(path: str, create: bool = True) -> Tuple[bytes, bool]:
    """读取密钥；不存在时按需创建。

    Args:
        path: 密钥文件路径。
        create: 文件不存在时是否自动生成并保存。

    Returns:
        (密钥字节, 是否新建)。

    Raises:
        FileNotFoundError: 文件不存在且 ``create=False``。
    """
    if os.path.exists(path):
        return load_secret(path), False
    if not create:
        raise FileNotFoundError(f"见证点密钥文件不存在: {path}")
    secret = _secrets.token_bytes(32)
    save_secret(secret, path)
    return secret, True


def resolve_secret(explicit: Optional[str] = None,
                   secret_file: Optional[str] = None,
                   default_path: Optional[str] = None,
                   create: bool = True) -> Tuple[Optional[bytes], str]:
    """按优先级解析见证点密钥。

    优先级：显式文本 > 指定密钥文件 > 默认密钥文件。

    Args:
        explicit: 命令行直接给出的密钥文本。
        secret_file: 密钥文件路径。
        default_path: 默认密钥文件路径（可为 None）。
        create: 默认路径下文件不存在时是否自动创建。

    Returns:
        (密钥字节 or None, 来源描述)。
    """
    if explicit:
        return parse_secret(explicit), "命令行"
    if secret_file:
        if not os.path.exists(secret_file):
            raise FileNotFoundError(f"见证点密钥文件不存在: {secret_file}")
        return load_secret(secret_file), f"文件 {secret_file}"
    if default_path:
        secret, created = load_or_create_secret(default_path, create=create)
        return secret, f"{default_path}（{'新建' if created else '已存在'}）"
    return None, "未指定"
