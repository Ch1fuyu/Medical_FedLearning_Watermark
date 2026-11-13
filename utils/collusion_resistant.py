"""
抗合谋水印（Collusion-Resistant Watermark, CRW）
=================================================

针对"多个恶意客户端将各自收到的定制模型做参数平均，以削弱水印并阻止溯源"
这一威胁模型，提供**精确合谋归因**能力：识别出到底是哪 K 个客户端合谋。

核心思想
--------
1. **共享见证点池（Shared Witness Pool）**
   所有客户端共享同一批"见证点"位置池。点 p 的权威值 e_p 与基线值 b_p
   **不再取自 encoder 参数切片，而是由服务器密钥单向派生**
   （见 `derive_witness_values`）：

       e_p = PRF(secret, "witness/e" || p)   （权威值）
       b_p = PRF(secret, "witness/b" || p)   （基线值，保证 e_p != b_p）

   BIBD 关联矩阵 A ∈ {0,1}^{v×n} 决定谁在哪个点写入权威值：

       A[p, c] = 1  ->  客户端 c 在点 p 写入 e_p
       A[p, c] = 0  ->  客户端 c 在点 p 写入 b_p（基线）

   (v,k,1)-BIBD 保证任意两客户端的标记点集合交 <= 1，
   因此各客户端定制模型在水印区域上高度可分。

   **密钥隔离的意义**：见证点权威值不再取自 encoder 参数切片，
   因此
     - 不再与任何客户端的私有切片发生"值空间重叠"（客户端 0 不再
       能独自掌握全部见证点权威值）；
     - 合谋者即使拼齐自己被分到的 encoder 切片，也无法反推 e_p；
     - 未被合谋者标记的点的 e_p 对合谋者完全不可知，因此**无法伪造**
       指向其他客户端集合的签名（陷害）。

2. **固定基线 b_p（Fixed Baseline）**
   见证点区域在所有客户端、所有轮次上共享同一组基线值 b_p，
   且该区域在本地训练中被**冻结**（不参与参数更新）。
   由此：

       L[p] = (m_p / K) * e_p + (1 - m_p / K) * b_p,   m_p = |{c in S : A[p,c]=1}|

   服务器已知 e_p 与 b_p，故可直接反解 m_p，得到"该点被多少个合谋者标记"。
   由于基线固定，信号**不会随轮次衰减**（否则 w_t 会被 e_p 逐渐污染，
   见证点信号在若干轮后消失）。

3. **子集匹配解码（Subset Matching）**
   合谋者平均得到泄漏模型 L = mean({M_c : c ∈ S})。
   服务器对每个客户端都持有其模型 M_c，故只需穷举候选集合 S'：

       S_hat = argmin_{S'} || L - mean({M_c : c ∈ S'}) ||^2

   这是精确的 ML 估计（噪声高斯时），无需预设合谋规模 K。

为什么不用 AND 唯一性
--------------------
泄漏模型是"参数平均"，某点被 m 个合谋者标记时其值为
w_t + (m/K)(e_p - w_t)。m=1 与 m=2 的取值之间不存在判决间隙，
因此 AND 语义在此威胁模型下失效，必须改用计数/匹配类解码。

作者：Medical_FedLearning_Watermark
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SingerParams",
    "singer_difference_set_generic",
    "is_difference_set",
    "build_incidence_matrix",
    "WitnessDesign",
    "build_witness_design",
    "WitnessLayout",
    "build_witness_layout",
    "save_witness_layout",
    "load_witness_layout",
    "find_witness_layout",
    "embed_witness_watermark",
    "embed_witness_watermark_torch",
    "decode_colluders",
    "decode_colluders_by_count",
    "estimate_collusion_size",
    "estimate_collusion_size_by_count",
    "simulate_collusion",
    "generate_witness_secret",
    "derive_witness_values",
    "derive_spreading_pattern",
    "DEFAULT_ENCODER_PARAMS",
    "DEFAULT_WITNESS_AMPLITUDE",
]


# ====================================================================
# 1. 有限域多项式运算（GF(p)）
# ====================================================================

def _p_mul_mod(a: Sequence[int], b: Sequence[int],
               f: Sequence[int], p: int) -> Tuple[int, ...]:
    """GF(p) 上多项式相乘并模 f。

    f 以低次在前排列：f[0] 为常数项，f[N] 为 1（首一）。
    """
    n = len(f) - 1
    conv = [0] * (2 * n)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                conv[i + j] = (conv[i + j] + ai * bj) % p
    for d in range(2 * n - 1, n - 1, -1):
        c = conv[d]
        if c:
            for t in range(n + 1):
                conv[d - n + t] = (conv[d - n + t] - c * f[t]) % p
            conv[d] = 0
    return tuple(conv[:n])


def _p_pow_mod(a: Sequence[int], e: int, f: Sequence[int], p: int) -> Tuple[int, ...]:
    """GF(p) 上 a^e mod f（平方-乘算法）。"""
    r = tuple([1] + [0] * (len(f) - 2))
    b = a
    while e > 0:
        if e & 1:
            r = _p_mul_mod(r, b, f, p)
        b = _p_mul_mod(b, b, f, p)
        e >>= 1
    return r


def _prime_power_decomp(q: int) -> Tuple[int, int]:
    """把 q 分解为 p^a，返回 (p, a)。q 非素数幂时抛错。"""
    if q < 2:
        raise ValueError(f"q={q} 必须 >= 2")
    for p in range(2, q + 1):
        if q % p == 0:
            t, a = q, 0
            while t % p == 0:
                t //= p
                a += 1
            if t == 1:
                return p, a
            break
    raise ValueError(f"q={q} 不是素数幂，无法用 Singer 构造")


#: 已知本原多项式（常数项在前，末位为 1）。
#: key = (p, N)，value = 本原多项式系数（低次在前）。
#: 构造 Singer 差集时以「差集校验」为唯一判据，故此表只需覆盖已验证项。
_KNOWN_PRIMITIVE_POLYS: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {
    (2, 3): [(1, 1, 0, 1), (1, 1, 1, 1)],                          # GF(2^3)  -> q=2
    (3, 3): [(1, 2, 0, 1), (2, 2, 0, 1), (1, 1, 1, 1), (2, 1, 0, 1)],  # GF(3^3) -> q=3
    (2, 6): [(1, 1, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 1, 1),
             (1, 1, 0, 0, 1, 0, 1), (1, 0, 0, 0, 0, 0, 1),
             (1, 1, 1, 0, 0, 0, 1), (1, 1, 0, 1, 0, 1, 1)],        # GF(2^6) -> q=4
}


# ====================================================================
# 2. Singer 差集
# ====================================================================

@dataclass(frozen=True)
class SingerParams:
    """一组 Singer 差集参数。"""
    q: int          # 素数幂
    p: int          # 特征素数
    a: int          # q = p^a
    v: int          # 地面集大小 = q^2 + q + 1
    k: int          # 差集大小 = q + 1
    D: Tuple[int, ...]   # 差集（模 v 的平移等价类）
    poly: Tuple[int, ...]  # 所用本原多项式

    def as_dict(self) -> dict:
        return {
            "q": self.q, "p": self.p, "a": self.a,
            "v": self.v, "k": self.k,
            "D": list(self.D), "poly": list(self.poly),
        }


def singer_difference_set_generic(q: int) -> SingerParams:
    """构造 Singer 差集（支持任意素数幂 q = p^a）。

    在 GF(q^3) = GF(p^{3a}) 中取本原元 alpha，令

        D = { i : Tr_{GF(q^3)/GF(q)}(alpha^i) = 0 }

    则 (v, k, lambda) = (q^2+q+1, q+1, 1) 的差集。

    Args:
        q: 素数幂（如 2, 3, 4, 5, 7, 8, 9...）。

    Returns:
        SingerParams。

    Raises:
        ValueError: q 不是素数幂，或找不到可用的本原多项式。
    """
    p, a = _prime_power_decomp(q)
    n = 3 * a                       # GF(q^3) = GF(p^{3a})
    cands = _KNOWN_PRIMITIVE_POLYS.get((p, n), [])

    for f in cands:
        try:
            v, k, D = _singer_from_poly(q, f)
        except Exception:            # noqa: BLE001
            continue
        if len(D) == k and is_difference_set(v, k, D):
            return SingerParams(q=q, p=p, a=a, v=v, k=k,
                                D=tuple(D), poly=tuple(f))
    raise ValueError(
        f"无法为 q={q} (p={p}, a={a}) 构造 Singer 差集："
        f"GF({p}^{n}) 的已知本原多项式均未产生合法差集。"
        f"请向 _KNOWN_PRIMITIVE_POLYS 补充本原多项式。"
    )


def _singer_from_poly(q: int, f: Sequence[int]) -> Tuple[int, int, List[int]]:
    """用给定本原多项式构造 Singer 差集。"""
    p, a = _prime_power_decomp(q)
    n = 3 * a
    v, k = q * q + q + 1, q + 1
    one = (1,) + (0,) * (n - 1)
    zero = (0,) * n
    x = (0, 1) + (0,) * (n - 2)     # 形式变量 x

    def trace(z):
        # Tr_{GF(q^3)/GF(q)}(z) = z + z^q + z^{q^2}
        z1 = _p_pow_mod(z, q, f, p)
        z2 = _p_pow_mod(z, q * q, f, p)
        return tuple((z[i] + z1[i] + z2[i]) % p for i in range(n))

    D: List[int] = []
    cur = one
    for i in range(v):
        if trace(cur) == zero:
            D.append(i)
        cur = _p_mul_mod(cur, x, f, p)
    return v, k, sorted(D)


def is_difference_set(v: int, k: int, D: Sequence[int]) -> bool:
    """校验 D 是否为 Z_v 上的 (v, k, 1) 差集。

    判据：所有有序非零差恰好各出现一次。
    """
    if len(D) != k or len(set(D)) != k:
        return False
    if any(not (0 <= d < v) for d in D):
        return False
    diffs = []
    for a, b in combinations(D, 2):
        diffs.append((a - b) % v)
        diffs.append((b - a) % v)
    return (len(diffs) == v - 1
            and len(set(diffs)) == v - 1
            and 0 not in diffs)


# ====================================================================
# 3. 关联矩阵
# ====================================================================

def build_incidence_matrix(v: int, n_clients: int, k: int,
                           D: Sequence[int]) -> np.ndarray:
    """由循环差集构造关联矩阵 A ∈ {0,1}^{v×n_clients}。

    第 i 列是差集 D 的一个平移：若 (D + i) ∩ {d} ≠ ∅ 则 A[d, i] = 1。

    性质：
        - 每列恰有 k 个 1（每客户端标记 k 个见证点）
        - 任意两列内积 <= 1（(v,k,1)-BIBD 性质，两客户端标记点交 <= 1）
        - 行权 = 该点被多少客户端标记

    Args:
        v: 地面集大小。
        n_clients: 客户端数（列数）。
        k: 差集大小（每列的 1 的个数）。
        D: Singer 差集。

    Returns:
        A，shape = (v, n_clients)，dtype float64。
    """
    A = np.zeros((v, n_clients), dtype=np.float64)
    for i in range(n_clients):
        for d in D:
            A[(d + i) % v, i] = 1.0
    return A


# ====================================================================
# 4. 见证点设计
# ====================================================================

#: LightAutoencoder 编码器参数量（本项目水印总量来源，见 train_key_matrix.py）
DEFAULT_ENCODER_PARAMS = 9472

#: 见证点权威值的默认幅度。见证点值由密钥派生（不再取自 encoder 切片），
#: 因此其绝对幅度不受 encoder 参数分布约束，需要显式指定一个有意义的量级，
#: 使见证点信号相对于任务梯度噪声足够突出。
DEFAULT_WITNESS_AMPLITUDE = 0.05


# ====================================================================
# 4.5 密钥派生的见证点取值
# ====================================================================

def generate_witness_secret() -> bytes:
    """生成一个新的见证点密钥（服务器持有，不下发给客户端）。"""
    return secrets.token_bytes(32)


def _prf_stream(secret: bytes, label: str, nbytes: int) -> bytes:
    """基于 HMAC-SHA256 的确定性字节流（counter 模式）。"""
    if not secret:
        raise ValueError("secret 不能为空")
    out = bytearray()
    counter = 0
    label_b = label.encode("utf-8")
    while len(out) < nbytes:
        msg = label_b + b"|" + counter.to_bytes(4, "big")
        out += hmac.new(secret, msg, hashlib.sha256).digest()
        counter += 1
    return bytes(out[:nbytes])


def derive_witness_values(secret: bytes,
                          v: int,
                          G: int,
                          amplitude: float = DEFAULT_WITNESS_AMPLITUDE
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """由服务器密钥单向派生见证点的**权威值**与**基线值**。

    权威值 e_p 与基线值 b_p 都是 (v, G) 数组，满足：

        e_p = PRF(secret, "witness/e" || p)
        b_p = PRF(secret, "witness/b" || p)   （默认基线为 0）

    设计要点
    --------
    1. **与 encoder 参数切片完全解耦**：见证点取值不再来自 encoder 参数向量的
       任何区间，因此
         - 不再与客户端 0 的私有切片发生"值空间重叠"，
         - 合谋者即便拼齐自己被分到的 encoder 切片也无法反推 e_p。
    2. **单向性**：e_p 由 PRF 派生，合谋者即使知道某点被自己标记，
       也**无法**推算出其它客户端标记点上的 e_p，因此无法伪造指向
       无辜客户端集合的见证点签名（陷害）。
    3. **非退化**：|e_p| >= 0.5 * amplitude，避免近零值导致该点的信噪比塌陷。
    4. **基线固定**：b_p 在所有客户端、所有轮次上相同，且见证点区域在本地
       训练中被冻结，因此见证点信号不会随轮次衰减。

    Args:
        secret: 服务器密钥（>= 16 字节）。
        v: 见证点数。
        G: 每点占用的模型位置数。
        amplitude: 权威值幅度。

    Returns:
        (e, b)：两个 (v, G) 的 float64 数组。
    """
    if v <= 0 or G <= 0:
        raise ValueError(f"v={v}, G={G} 必须为正")
    if amplitude <= 0:
        raise ValueError(f"amplitude={amplitude} 必须为正")

    raw = _prf_stream(secret, "witness/e", v * G * 4)
    # 用 uint32 解释字节（避免随机字节被解释成 NaN/Inf 的浮点数）
    words = np.frombuffer(raw, dtype=np.uint32)[:v * G].astype(np.float64)
    # 映射到 (-1, 1)：以 2^32 为周期做线性映射
    u = (words / 4294967296.0) * 2.0 - 1.0
    sgn = np.sign(u)
    sgn[sgn == 0.0] = 1.0
    mag = 0.5 + 0.5 * np.abs(u)              # -> [0.5, 1.0]
    e = (sgn * mag * amplitude).reshape(v, G)

    # 基线固定为 0：见证点区域对模型容量的占用可忽略，且使解码公式
    #   L[p] = (m_p / K) * e_p
    # 形式最简（m_p = 该点被多少个合谋者标记）。
    b = np.zeros((v, G), dtype=np.float64)

    # 兜底：确保 e 与 b 处处不同（防止基线非零时出现相等）
    same = np.isclose(e, b)
    if same.any():
        e[same] = amplitude

    return e, b


def derive_spreading_pattern(secret: bytes, n: int, v: int = 0,
                             G: int = 0) -> np.ndarray:
    """构造把每个见证点的 G 个副本**分散**开的确定性映射。

    为什么要分散（扩频）::

        若点 p 的 G 个副本连续排列，则一段连续的攻击（剪枝 / 幅值置零）
        可以整块抹掉某一个点；分散后，连续攻击只能同时影响多个点的
        少量副本，配合解码端的"逐点中位数"即可恢复。

    映射规则（"转置"式交错）
    ----------------------
    位置池长度 n = v * G。把池子看成 v 列、G 行：

        slot(p, j) -> pool index = j * v + perm_point[p]

    其中 ``perm_point`` 是由密钥派生的点序置换。于是

        - 点 p 的 G 个副本位于 pool index = perm_point[p], +v, +2v, ...
          **两两间隔恰好为 v**，因此任意两个副本都不相邻（当 v >= 2），
          且一段长度为 L 的连续攻击最多破坏该点约 L/v 个副本；
        - ``perm_point`` 随机化了点与"列"的对应关系，
          使攻击者无法预知某个点的副本在池子中的大致位置；
        - 最后整体做一次密钥派生的**循环旋转**，避免固定偏移规律。

    Args:
        secret: 服务器密钥。
        n: 位置池长度（= v * G）。
        v: 见证点数；为 0 时退化为随机置换。
        G: 每点副本数；为 0 时退化为随机置换。

    Returns:
        长度 n 的整数数组 ``perm``，其中 ``perm[slot]`` 为该 slot 对应的
        池中位置；内容为 [0, n) 的一个置换。
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    if v > 0 and G > 0 and v * G == n:
        # ---- 转置式交错：保证副本间隔恰好为 v ----
        raw = _prf_stream(secret, "witness/perm", v * 4)
        words = np.frombuffer(raw, dtype=np.uint32)[:v].astype(np.int64)
        order = np.argsort(words, kind="stable")
        perm_point = np.empty(v, dtype=np.int64)
        perm_point[order] = np.arange(v, dtype=np.int64)

        perm = np.empty(n, dtype=np.int64)
        for p in range(v):
            for j in range(G):
                perm[p * G + j] = j * v + int(perm_point[p])

        # 整体循环旋转（密钥派生），保持间隔性质不变
        rot = int(np.frombuffer(_prf_stream(secret, "witness/rot", 4),
                                dtype=np.uint32)[0]) % n
        perm = (perm + rot) % n
        return perm

    # ---- 退化路径：纯随机置换（仅用于测试 / 非标准尺寸）----
    raw = _prf_stream(secret, "witness/spread", n * 4)
    idx = np.frombuffer(raw, dtype=np.uint32)[:n].astype(np.int64)
    order = np.argsort(idx, kind="stable")
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.arange(n, dtype=np.int64)
    return perm

@dataclass
class WitnessDesign:
    """见证点水印的完整设计。

    Attributes:
        n_clients: 客户端数。
        K: 期望抵抗的最大合谋规模（用于限制搜索集合）。
        v: 见证点数（**压缩后**的实际使用点数）。
        k: 每客户端标记的点数（BIBD 块大小）。
        G: 每见证点占用的模型位置数。
        A: 关联矩阵，shape (v, n_clients)。
        singer: 对应的 Singer 差集参数。
            注意 ``singer.v`` 是差集的原始地面集大小
            （q^2+q+1），而 ``v`` 是其中真正被使用的点数
            （已剔除行权为 0 的点），故 ``v <= singer.v``。
        positions: 见证点池在模型中的位置分配。
    """
    n_clients: int
    K: int
    v: int
    k: int
    G: int
    A: np.ndarray
    singer: SingerParams
    positions: List[Tuple[str, int]] = field(default_factory=list)

    # -- 派生属性 ---------------------------------------------------
    @property
    def n_witness_positions(self) -> int:
        """见证点池占用的总位置数 = v * G。"""
        return self.v * self.G

    @property
    def row_weights(self) -> np.ndarray:
        """每个见证点被多少客户端标记。"""
        return self.A.sum(axis=1).astype(int)

    def client_points(self, client_id: int) -> np.ndarray:
        """客户端 client_id 标记的见证点下标。"""
        return np.where(self.A[:, client_id] > 0)[0]

    def summary(self) -> str:
        rw = self.row_weights
        v_raw = self.singer.v
        compressed = (v_raw != self.v)
        return (f"见证点设计: n={self.n_clients} K={self.K} "
                f"q={self.singer.q} v={self.v} k={self.k} G={self.G}\n"
                f"  关联矩阵 rank = {np.linalg.matrix_rank(self.A)}/{self.n_clients}\n"
                f"  行权分布 = {np.bincount(rw, minlength=self.k + 1).tolist()}\n"
                + (f"  地面集压缩: v_raw={v_raw} -> v_used={self.v} "
                   f"(剔除 {v_raw - self.v} 个无人标记的点)\n" if compressed else "")
                + f"  位置池 = {self.n_witness_positions} 个 "
                f"(v={self.v} × G={self.G})")

    def to_dict(self) -> dict:
        return {
            "n_clients": self.n_clients,
            "K": self.K,
            "v": self.v,
            "k": self.k,
            "G": self.G,
            "A": self.A.astype(np.uint8).tolist(),
            "singer": self.singer.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WitnessDesign":
        sp = d["singer"]
        singer = SingerParams(
            q=sp["q"], p=sp["p"], a=sp["a"], v=sp["v"], k=sp["k"],
            D=tuple(sp["D"]), poly=tuple(sp["poly"]),
        )
        A = np.array(d["A"], dtype=np.float64)
        # 容错：以实际矩阵行数为准（旧版布局可能未压缩）
        v = A.shape[0] if A.ndim == 2 else d["v"]
        return cls(
            n_clients=d["n_clients"], K=d["K"], v=v, k=d["k"],
            G=d["G"], A=A, singer=singer,
        )


def build_witness_design(
    n_clients: int,
    K: int = 3,
    encoder_params: int = DEFAULT_ENCODER_PARAMS,
    positions: Optional[List[Tuple[str, int]]] = None,
    q_candidates: Sequence[int] = (2, 3, 4),
    verbose: bool = True,
) -> WitnessDesign:
    """构造见证点水印设计。

    选型策略：在所有 q 候选中挑 v >= n_clients 的最优者，
    优先「行权 >= 2 的点最多」（能区分 >= 2 个合谋者），
    其次「行权方差最小」（各点被标记数均衡）。

    Args:
        n_clients: 客户端数。
        K: 期望抵抗的最大合谋规模。
        encoder_params: 水印总量（编码器参数量），决定位置池大小。
        positions: 可用的模型位置列表 [(param_name, local_idx), ...]；
            若给出，则见证点池取其前 v*G 个。
        q_candidates: 尝试的素数幂（需保证 q^2+q+1 >= n_clients）。
        verbose: 是否打印选型过程。

    Returns:
        WitnessDesign。

    Raises:
        ValueError: 无可行的 Singer 参数（v < n_clients）。
    """
    best = None
    for q in q_candidates:
        try:
            singer = singer_difference_set_generic(q)
        except ValueError:
            continue
        if singer.v < n_clients:
            continue
        A_full = build_incidence_matrix(singer.v, n_clients, singer.k, singer.D)
        if np.linalg.matrix_rank(A_full) < n_clients:
            continue
        # 压缩：丢弃行权 = 0 的点（无人标记，纯浪费）
        covered = np.where(A_full.sum(axis=1) > 0)[0]
        A_c = A_full[covered, :]
        # 打分基于压缩后的实际设计
        rw = A_c.sum(axis=1)
        n_ge2 = int((rw >= 2).sum())
        score = (n_ge2, -float(rw.var()))
        if best is None or score > best[0]:
            best = (score, singer, A_c)

    if best is None:
        raise ValueError(
            f"没有可行的 Singer 参数：需要 v=q^2+q+1 >= {n_clients}，"
            f"但候选 q={list(q_candidates)} 均不满足或关联矩阵不满秩")

    _, singer, A = best
    v, k = A.shape[0], singer.k
    # 每见证点占用的位置数
    G = max(1, encoder_params // (n_clients * v))
    # 总位置数按 encoder_params 兜底（保证不超预算）
    total = v * G
    if total > encoder_params:
        G = encoder_params // v

    if positions is not None:
        need = v * G
        if len(positions) < need:
            raise ValueError(
                f"可用位置不足：需要 {need} 个（v={v} × G={G}），"
                f"实际只有 {len(positions)} 个")
        # 见证点池取前 need 个位置
        design_positions = list(positions[:need])
    else:
        design_positions = []

    design = WitnessDesign(
        n_clients=n_clients, K=K, v=v, k=k, G=G, A=A, singer=singer,
        positions=design_positions,
    )
    if verbose:
        print(f"[*] {design.summary()}")
    return design


# ====================================================================
# 5. 布局（持久化：位置池 + 关联矩阵 + 权威值切片规则）
# ====================================================================

@dataclass
class WitnessLayout:
    """见证点水印的**可持久化布局**。

    与 ``WitnessDesign`` 的区别：Layout 额外记录了见证点取值与位置映射，
    使得嵌入端与解码端可以在不同进程、不同轮次之间稳定复现同一套映射。

    权威值映射规则（v2：密钥派生，与 encoder 切片解耦）
    -----------------------------------------------
    见证点的权威值 e_p 与基线值 b_p **不再取自 encoder 参数向量的任何区间**，
    而是由服务器密钥单向派生（见 ``derive_witness_values``）：

        e_p = PRF(secret, "witness/e" || p)
        b_p = PRF(secret, "witness/b" || p)

    独占切片区**完全不受影响**：客户端 c 的私有权威值仍取自
    ``encoder_params[offset_c : offset_c + len(P_c)]``，
    main.py 的 offset 计算逻辑**完全不变**。

    相对 v1（"见证点复用 encoder 前段"）的收益：

    - 见证点值与任何客户端的私有切片**不再有数值重叠**，
      客户端 0 不再独自掌握全部见证点权威值；
    - 合谋者即便拼齐自己分到的 encoder 切片也无法反推 e_p；
    - 合谋者无法伪造指向无辜客户端的见证点签名（陷害）。

    位置布局
    --------
    - 见证点位置取自"所有未被任何客户端分配"的 conv 位置，
      与独占切片位置**严格不相交**；
    - 每个见证点的 G 个副本通过 ``spread_perm`` **分散**在整个见证点池中，
      避免连续攻击（剪枝 / 幅值置零）整块抹掉某一个点。

    Attributes:
        design: 见证点设计。
        slice_size: 每客户端独占切片长度。
        witness_offset: 保留字段（v1 中见证点权威值区的起点）。
            v2 中见证点值不再来自 encoder 向量，该字段仅用于兼容旧布局。
        slice_offset: 独占切片区在 encoder 参数向量中的起点。
        point_values: (v, G) 的权威值 e_p。
        base_values: (v, G) 的基线值 b_p。
        spread_perm: 见证点池内的分散置换（长度 = v*G）。
        secret_id: 密钥标识（用于日志与校验，不含密钥本身）。
    """
    design: WitnessDesign
    slice_size: int
    witness_offset: int = 0
    slice_offset: int = 0
    point_values: Optional[np.ndarray] = None
    base_values: Optional[np.ndarray] = None
    spread_perm: Optional[np.ndarray] = None
    secret_id: str = ""

    @property
    def n_clients(self) -> int:
        return self.design.n_clients

    @property
    def v(self) -> int:
        return self.design.v

    @property
    def G(self) -> int:
        return self.design.G

    @property
    def slice_end(self) -> int:
        """独占切片区结束位置（= slice_offset + n_clients * slice_size）。"""
        return self.slice_offset + self.n_clients * self.slice_size

    @property
    def A(self) -> np.ndarray:
        return self.design.A

    @property
    def positions(self) -> List[Tuple[str, int]]:
        return self.design.positions

    def encoder_slice_for_point(self, point: int) -> Tuple[int, int]:
        """点 point 的权威值在 encoder 参数向量中的区间 [start, end)。

        .. deprecated::
            v2 起见证点权威值不再来自 encoder 向量，该方法仅用于
            兼容旧布局与旧调用方。
        """
        start = self.witness_offset + point * self.G
        return start, start + self.G

    def point_positions(self, point: int) -> List[Tuple[str, int]]:
        """见证点 point 的 G 个副本在模型中的位置列表。

        若布局带有 ``spread_perm``，副本被分散到池子各处；
        否则退化为连续布局（兼容 v1 布局）。
        """
        n = self.v * self.G
        if self.spread_perm is not None and len(self.spread_perm) == n:
            idx = self.spread_perm[point * self.G:(point + 1) * self.G]
        else:
            idx = np.arange(point * self.G, (point + 1) * self.G)
        out = []
        for i in idx:
            i = int(i)
            if 0 <= i < len(self.positions):
                out.append(self.positions[i])
        return out

    def extract_point_values(self, encoder_params: np.ndarray) -> np.ndarray:
        """取得 (v, G) 的见证点权威值。

        v2 优先使用布局内置的密钥派生值；只有当布局没有内置值时
        （例如加载了 v1 旧布局），才回退到从 encoder 参数向量切片。
        """
        if self.point_values is not None:
            return np.asarray(self.point_values, dtype=np.float64)
        arr = np.asarray(encoder_params, dtype=np.float64).reshape(-1)
        out = np.zeros((self.v, self.G))
        for p in range(self.v):
            s, e = self.encoder_slice_for_point(p)
            seg = arr[s:e]
            if seg.size < self.G:
                seg = np.pad(seg, (0, self.G - seg.size), mode="edge")
            out[p] = seg[:self.G]
        return out

    def extract_base_values(self) -> np.ndarray:
        """取得 (v, G) 的见证点基线值（默认全 0）。"""
        if self.base_values is not None:
            return np.asarray(self.base_values, dtype=np.float64)
        return np.zeros((self.v, self.G), dtype=np.float64)

    def to_dict(self) -> dict:
        d = {
            "design": self.design.to_dict(),
            "slice_size": self.slice_size,
            "witness_offset": self.witness_offset,
            "slice_offset": self.slice_offset,
            "positions": [list(p) for p in self.positions],
            "secret_id": self.secret_id,
            "layout_version": 2,
        }
        if self.point_values is not None:
            d["point_values"] = np.asarray(
                self.point_values, dtype=np.float64).reshape(-1).tolist()
        if self.base_values is not None:
            d["base_values"] = np.asarray(
                self.base_values, dtype=np.float64).reshape(-1).tolist()
        if self.spread_perm is not None:
            d["spread_perm"] = [int(x) for x in self.spread_perm]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WitnessLayout":
        design = WitnessDesign.from_dict(d["design"])
        design.positions = [(p[0], p[1]) for p in d.get("positions", [])]
        v, G = design.v, design.G
        pv = d.get("point_values")
        bv = d.get("base_values")
        sp = d.get("spread_perm")
        return cls(
            design=design,
            slice_size=d["slice_size"],
            witness_offset=d.get("witness_offset", 0),
            slice_offset=d.get("slice_offset", 0),
            point_values=(np.asarray(pv, dtype=np.float64).reshape(v, G)
                          if pv is not None else None),
            base_values=(np.asarray(bv, dtype=np.float64).reshape(v, G)
                         if bv is not None else None),
            spread_perm=(np.asarray(sp, dtype=np.int64) if sp is not None else None),
            secret_id=d.get("secret_id", ""),
        )


def build_witness_layout(
    n_clients: int,
    K: int = 3,
    encoder_params: int = DEFAULT_ENCODER_PARAMS,
    positions: Optional[List[Tuple[str, int]]] = None,
    slice_size: Optional[int] = None,
    secret: Optional[bytes] = None,
    amplitude: float = DEFAULT_WITNESS_AMPLITUDE,
    verbose: bool = True,
) -> WitnessLayout:
    """构造可持久化的见证点布局。

    Args:
        n_clients: 客户端数。
        K: 期望抵抗的最大合谋规模。
        encoder_params: encoder 参数总量（仅用于预算与切片区计算）。
        positions: 共享见证点位置池 [(param_name, local_idx), ...]。
        slice_size: 每客户端独占切片长度；None 时按 encoder_params/n_clients
            自动计算。
        secret: 见证点密钥。**必须提供**，否则见证点值不具备不可伪造性。
            若为 None，会生成一个随机密钥并打印警告（仅适用于测试）。
        amplitude: 见证点权威值幅度。
        verbose: 是否打印摘要。

    Returns:
        WitnessLayout。
    """
    if slice_size is None:
        slice_size = encoder_params // n_clients

    # 先算见证点需要多少位置（v*G）
    design = build_witness_design(n_clients, K=K, encoder_params=encoder_params,
                                  positions=None, verbose=False)
    need = design.v * design.G

    if positions is None or len(positions) < need:
        raise ValueError(
            f"共享见证点位置池不足：需要 {need} 个（v={design.v} × G={design.G}），"
            f"实际 {0 if positions is None else len(positions)} 个。"
            f"请把所有客户端的 positions.json 合并去重后传入。"
        )

    # 位置预算：见证点占用 v*G 个 conv 位置，且必须与独占切片位置不相交
    #（调用方传入的 positions 应为"未被任何客户端分配"的空闲位置）。
    #
    # 注意：独占切片的总量可能略超 encoder_params（equal 策略的余数
    # 分配所致，如 n=10 时 948*2 + 947*8 = 9480 > 9472）。这是**既有
    # 行为**，现有代码用 `enc_global_idx < len(encoder_params)` 兜底。
    # 见证点布局不改变这一行为，也不因此失败。
    if need > encoder_params:
        raise ValueError(
            f"见证点需要 {need} 个模型位置，超过水印总量 {encoder_params}")

    design.positions = list(positions[:need])

    # ---- 见证点取值：由密钥单向派生，与 encoder 参数切片完全解耦 ----
    if secret is None:
        secret = generate_witness_secret()
        if verbose:
            print("[!] 未提供见证点密钥，已生成随机密钥。"
                  "生产环境请通过 --witness_secret 显式指定并妥善保管。")
    if len(secret) < 16:
        raise ValueError(f"见证点密钥长度不足（{len(secret)} < 16 字节）")

    point_values, base_values = derive_witness_values(
        secret, design.v, design.G, amplitude=amplitude)
    spread_perm = derive_spreading_pattern(secret, need, v=design.v, G=design.G)
    secret_id = hashlib.sha256(secret).hexdigest()[:16]

    layout = WitnessLayout(design=design, slice_size=slice_size,
                            witness_offset=0, slice_offset=0,
                            point_values=point_values,
                            base_values=base_values,
                            spread_perm=spread_perm,
                            secret_id=secret_id)
    if verbose:
        print(f"[*] {layout.summary()}")
    return layout


def _layout_summary(self: WitnessLayout) -> str:
    rw = self.design.row_weights
    v_raw = self.design.singer.v
    compressed = (v_raw != self.v)
    derived = self.point_values is not None
    return (f"见证点布局: n={self.n_clients} K={self.design.K} "
            f"q={self.design.singer.q} v={self.v} k={self.design.k} G={self.G}\n"
            f"  关联矩阵 rank = {np.linalg.matrix_rank(self.A)}/{self.n_clients}\n"
            f"  行权分布 = {np.bincount(rw, minlength=self.design.k + 1).tolist()}\n"
            + (f"  地面集压缩: v_raw={v_raw} -> v_used={self.v} "
               f"(剔除 {v_raw - self.v} 个无人标记的点)\n" if compressed else "")
            + (f"  见证点取值: 密钥派生 (secret_id={self.secret_id}), "
               f"与 encoder 切片解耦\n" if derived else
               f"  见证点取值: 从 encoder 参数切片提取 (v1 旧布局, 建议重新生成)\n")
            + f"  独占切片区 = [{self.slice_offset}, {self.slice_end})  "
            f"({self.slice_size}/客户端)\n"
            f"  见证点位置 = {len(self.positions)} 个 (与切片位置不相交, 已分散)")


WitnessLayout.summary = _layout_summary   # type: ignore[attr-defined]


def save_witness_layout(layout: WitnessLayout, path: str) -> str:
    """把布局保存为 JSON。"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout.to_dict(), f, indent=2)
    return path


def load_witness_layout(path: str) -> WitnessLayout:
    """从 JSON 加载布局。"""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"见证点布局文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return WitnessLayout.from_dict(json.load(f))


def find_witness_layout(key_matrix_path: str,
                        model_type: Optional[str] = None,
                        client_num: Optional[int] = None) -> Optional[str]:
    """在 key_matrix 目录下查找 witness_layout.json。

    查找顺序：
        1. ``<key_matrix_path>/witness_layout.json``
        2. ``<key_matrix_path>/client<N>/witness_layout.json``
        3. ``<key_matrix_path>/<model_type>/client<N>/witness_layout.json``

    Args:
        key_matrix_path: key_matrix 根目录或具体子目录。
        model_type: 模型类型（'resnet' / 'alexnet'）。
        client_num: 客户端数。

    Returns:
        布局文件路径；找不到时返回 None。
    """
    if not key_matrix_path:
        return None
    candidates = [os.path.join(key_matrix_path, "witness_layout.json")]
    if client_num is not None:
        candidates.append(os.path.join(key_matrix_path, f"client{client_num}",
                                       "witness_layout.json"))
        if model_type:
            candidates.append(os.path.join(key_matrix_path, model_type,
                                           f"client{client_num}",
                                           "witness_layout.json"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ====================================================================
# 6. 嵌入
# ====================================================================

def embed_witness_watermark(
    model_params: Dict[str, np.ndarray],
    design: WitnessDesign,
    client_id: int,
    encoder_values: np.ndarray,
    point_values: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """把见证点水印嵌入到客户端 client_id 的定制模型。

    规则（对每个见证点 p）：
        - 若 A[p, client_id] == 1：在点 p 的 G 个位置写入权威值 e_p
        - 否则：保持模型原值（即全局模型 w_t）

    Args:
        model_params: 全局模型参数（键为参数名，值为 numpy 数组或 torch 张量）。
        design: 见证点设计。
        client_id: 客户端 ID。
        encoder_values: 权威值来源，形状 (n_clients, v, G) 或 (v, G)。
            长度 v*G 的扁平数组亦可。
        point_values: 可选的逐点权威值覆盖，shape (v, G)。

    Returns:
        新的参数字典（浅拷贝 + 修改），不修改入参。
    """
    v, G = design.v, design.G
    marked = set(design.client_points(client_id).tolist())

    # 解析权威值 -> (v, G)
    ev = _resolve_point_values(encoder_values, design, client_id)

    out = {k: (v_.clone() if hasattr(v_, "clone") else np.array(v_, copy=True))
           for k, v_ in model_params.items()}

    if point_values is not None:
        pv = np.asarray(point_values, dtype=np.float64).reshape(v, G)
    else:
        pv = ev

    # 未标记点写入固定基线（默认 0）。基线在所有客户端、所有轮次相同，
    # 因此"某点被 m 个合谋者标记"可直接由泄漏模型反解：
    #     L[p] = (m/K) * e_p + (1 - m/K) * b_p
    # 若基线为 0，则 L[p] / e_p = m / K。
    bv = np.zeros((v, G), dtype=np.float64)

    for p in range(v):
        val = pv[p] if p in marked else bv[p]
        # 该点的 G 个位置：design.positions 中 [p*G : (p+1)*G]
        for j in range(G):
            pos_idx = p * G + j
            if pos_idx >= len(design.positions):
                break
            param_name, local_idx = design.positions[pos_idx]
            if param_name not in out:
                continue
            flat = out[param_name].reshape(-1)
            if 0 <= local_idx < flat.shape[0]:
                flat[local_idx] = val[j]

    return out


def embed_witness_watermark_torch(
    model_params: Dict[str, "torch.Tensor"],
    layout: WitnessLayout,
    client_id: int,
    encoder_params: "torch.Tensor",
) -> Dict[str, "torch.Tensor"]:
    """把见证点水印嵌入到定制模型（torch 版本，供 main.py 直接调用）。

    与 ``embed_witness_watermark`` 的区别：
      - 输入输出均为 torch 张量（``model_params`` 会被就地修改并返回）
      - 权威值直接从 layout 规定的 encoder 参数区间提取，
        保证嵌入端与解码端使用同一套映射

    Args:
        model_params: 模型参数字典（会被就地修改）。
        layout: 见证点布局。
        client_id: 客户端 ID。
        encoder_params: encoder 参数向量（扁平 tensor）。

    Returns:
        model_params（就地修改后返回）。
    """
    import torch  # 局部导入，避免顶层依赖

    point_values = layout.extract_point_values(
        encoder_params.detach().cpu().numpy().astype(np.float64))
    base_values = layout.extract_base_values()

    marked = set(layout.design.client_points(client_id).tolist())
    G = layout.G

    # v2 布局：未标记点写入**固定基线**（默认 0）。
    # 这样见证点区域在所有客户端、所有轮次上完全一致（只有标记点不同），
    # 于是合谋平均后可直接反解"该点被多少个合谋者标记"，且信号不随轮次衰减。
    # v1 旧布局（point_values 为 None）保持原行为：未标记点沿用模型原值。
    write_baseline = layout.point_values is not None

    # 原地写入叶张量（nn.Parameter）必须关掉梯度，否则会报
    # "a view of a leaf Variable that requires grad is being used in
    #  an in-place operation"。
    with torch.no_grad():
        for p in range(layout.v):
            if p in marked:
                vals = point_values[p]
            elif write_baseline:
                vals = base_values[p]
            else:
                continue

            positions = layout.point_positions(p)
            for j, (param_name, local_idx) in enumerate(positions):
                if j >= G:
                    break
                if param_name not in model_params:
                    continue
                flat = model_params[param_name].view(-1)
                if 0 <= local_idx < flat.numel():
                    flat[local_idx] = float(vals[j])

    return model_params


def _resolve_point_values(encoder_values, design: WitnessDesign,
                           client_id: int) -> np.ndarray:
    """把各种形式的 encoder 权威值统一成 (v, G) 数组。"""
    v, G = design.v, design.G
    arr = np.asarray(encoder_values, dtype=np.float64)

    if arr.ndim == 3:
        # (n_clients, v, G)
        return arr[client_id].reshape(v, G)
    if arr.ndim == 2:
        # (v, G)
        return arr.reshape(v, G)
    if arr.ndim == 1:
        # 扁平：需要 n_clients * v * G 个
        need_all = design.n_clients * v * G
        if arr.size >= need_all:
            return arr[client_id * v * G:(client_id + 1) * v * G].reshape(v, G)
        if arr.size >= v * G:
            return arr[:v * G].reshape(v, G)
        raise ValueError(
            f"encoder_values 长度不足：至少需要 v*G={v * G}，实际 {arr.size}")
    raise ValueError(f"encoder_values 维度不支持: {arr.shape}")


# ====================================================================
# 6. 解码
# ====================================================================

def decode_colluders(
    leaked_model: Dict[str, np.ndarray],
    client_models: Dict[int, Dict[str, np.ndarray]],
    design: WitnessDesign,
    max_colluders: Optional[int] = None,
    position_mask: Optional[List[Tuple[str, int]]] = None,
    verbose: bool = False,
) -> Tuple[List[int], dict]:
    """子集匹配解码：从泄漏模型中识别出合谋的客户端集合。

    原理：泄漏模型 L = mean({M_c : c ∈ S})。
    服务器对每个客户端都持有其模型 M_c，故只需在候选集合中
    找到使残差最小的那一个：

        S_hat = argmin_{S'} || L - mean({M_c : c ∈ S'}) ||^2

    这在高斯噪声下是最大似然估计，且不要求预先知道合谋规模。

    Args:
        leaked_model: 泄漏模型参数字典。
        client_models: {client_id: 模型参数字典}，服务器持有的各客户端模型。
        design: 见证点设计。
        max_colluders: 最大合谋规模；None 时用 design.K。
        position_mask: 只在这些位置上计算残差；None 时用 design.positions，
            若 design.positions 为空则用全部同名参数的全部元素。
        verbose: 是否打印诊断信息。

    Returns:
        (合谋者列表, 诊断信息字典)。
    """
    k_max = max_colluders if max_colluders is not None else design.K
    k_max = max(1, min(k_max, len(client_models)))

    # 提取用于比较的特征向量
    leak_vec = _extract_feature(leaked_model, position_mask, design)
    cand_vecs = {cid: _extract_feature(m, position_mask, design)
                 for cid, m in client_models.items()}

    client_ids = sorted(cand_vecs.keys())
    M = np.stack([cand_vecs[c] for c in client_ids])      # (n, D)
    n = M.shape[0]

    # 向量化残差计算：
    #   ||L - mean(S)||^2 = ||L||^2 - 2/r * <L, sum_S> + 1/r^2 * ||sum_S||^2
    # 用 Gram 矩阵 G = M M^T 与 L·M 向量化计算所有子集，避免逐个重建数组。
    G = M @ M.T                       # (n, n)
    Ldot = M @ leak_vec               # (n,)
    Lnorm = float(leak_vec @ leak_vec)

    best_set, best_err = None, np.inf
    all_errors = []

    for r in range(1, k_max + 1):
        for combo in combinations(range(n), r):
            idx = np.fromiter(combo, dtype=np.intp, count=r)
            sub = G[np.ix_(idx, idx)].sum()
            cross = Ldot[idx].sum()
            # 均方误差（除以 D 只是常数，不影响 argmin）
            err = Lnorm - 2.0 * cross / r + sub / (r * r)
            all_errors.append((combo, err))
            if err < best_err:
                best_err = err
                best_set = tuple(client_ids[i] for i in combo)

    diag = {
        "best_error": best_err,
        "n_candidates": len(all_errors),
        "margin": None,
        "runner_up_error": None,
    }
    if all_errors:
        errs = sorted(e for _, e in all_errors)
        diag["runner_up_error"] = errs[1] if len(errs) > 1 else None
        if diag["runner_up_error"] is not None:
            diag["margin"] = diag["runner_up_error"] - best_err

    if verbose:
        print(f"[*] 解码: 合谋者 = {list(best_set) if best_set else []} "
              f"(残差={best_err:.6e}, 候选数={len(all_errors)})")

    return (list(best_set) if best_set else []), diag


# ====================================================================
# 7.5 基于"计数"的鲁棒解码（v2 推荐路径）
# ====================================================================

def _extract_point_matrix(model: Dict[str, np.ndarray],
                          layout: "WitnessLayout",
                          agg: str = "median") -> np.ndarray:
    """从模型参数中提取 (v, G) 的见证点取值矩阵。

    每个见证点 p 有 G 个**分散**的副本。对这 G 个副本做稳健聚合
    （默认中位数），使得局部的连续攻击（剪枝 / 幅值置零）最多破坏
    少数副本，仍能恢复该点的取值。

    Args:
        model: 模型参数字典（numpy 数组或 torch 张量）。
        layout: 见证点布局。
        agg: 'median'（默认，抗局部攻击）或 'mean'。

    Returns:
        (v, G) 的 float64 数组；缺失的位置填 NaN。
    """
    v, G = layout.v, layout.G
    out = np.full((v, G), np.nan, dtype=np.float64)

    for p in range(v):
        for j, (param_name, local_idx) in enumerate(layout.point_positions(p)):
            if j >= G:
                break
            if param_name not in model:
                continue
            arr = model[param_name]
            if hasattr(arr, "detach"):          # torch tensor
                arr = arr.detach().cpu().numpy()
            flat = np.asarray(arr, dtype=np.float64).reshape(-1)
            if 0 <= local_idx < flat.shape[0]:
                out[p, j] = flat[local_idx]

    return out


def _aggregate_points(point_matrix: np.ndarray, agg: str = "median") -> np.ndarray:
    """把 (v, G) 的副本矩阵聚合成 (v,) 的逐点取值。

    缺失（NaN）的副本被忽略；若某点所有副本都缺失，则返回 NaN。

    .. note::
        每个见证点的 G 个副本携带的是**同一个点的不同幅度** e_p[j]，
        因此逐点直接取中位数会混合不同 j 的幅度。
        正确的做法是**先按 (p, j) 逐副本归一化成比值，再对 j 维聚合**
        （见 `decode_colluders_by_count`）。本函数仅用于调试/展示。
    """
    v = point_matrix.shape[0]
    out = np.full(v, np.nan, dtype=np.float64)
    for p in range(v):
        row = point_matrix[p]
        row = row[~np.isnan(row)]
        if row.size == 0:
            continue
        out[p] = np.median(row) if agg == "median" else float(row.mean())
    return out


def decode_colluders_by_count(
    leaked_model: Dict[str, np.ndarray],
    client_models: Dict[int, Dict[str, np.ndarray]],
    layout: "WitnessLayout",
    max_colluders: Optional[int] = None,
    agg: str = "median",
    max_avg_error: float = 0.25,
    min_margin: float = 1e-6,
    verbose: bool = False,
) -> Tuple[List[int], dict]:
    """基于"见证点计数"的鲁棒合谋归因（v2 推荐解码路径）。

    原理
    ----
    见证点 p 的权威值为 e_p、基线为 b_p（v2 中 b_p = 0，且见证点区域被冻结，
    因此该区域在所有客户端、所有轮次上完全确定）。若合谋集合为 S（|S| = K），
    则泄漏模型在点 p 上的取值是

        L[p] = (m_p / K) * e_p + (1 - m_p / K) * b_p,
        m_p = |{c in S : A[p, c] = 1}|

    当 b_p = 0 时，**比值** r_p = L[p] / e_p = m_p / K 与 K 无关。
    对任意候选集合 S'（大小 r），其在该点上的预测比值为

        r'_p = |{c in S' : A[p, c] = 1}| / r

    因此归因退化为一个**计数匹配**问题：

        S_hat = argmin_{S'} sum_p (r_p - r'_p)^2

    相比 v1 的"原始参数残差"匹配，计数匹配有三点优势：

    1. **归一化**：除以 e_p 消除了不同点幅度差异的影响；
    2. **鲁棒**：每个点的 G 个分散副本先做中位数聚合，
       局部攻击最多破坏少数副本；
    3. **可弃权**：若最优子集的平均残差仍显著大于噪声水平，
       说明泄漏模型不是任何合谋的平均，返回空列表（不硬凑一个结论）。

    Args:
        leaked_model: 泄漏模型参数字典。
        client_models: {client_id: 模型参数字典}，服务器持有的各客户端模型。
        layout: 见证点布局（提供权威值、基线值与位置映射）。
        max_colluders: 最大搜索规模；None 时用 layout.design.K。
        agg: 点内副本聚合方式（'median' / 'mean'）。
        max_avg_error: 平均残差上限。超过则认为泄漏模型并非任何合谋集合
            的平均，返回空列表（弃权），避免"硬凑一个结论"。
        min_margin: 最优与次优的最小差距；小于该值同样弃权。
        verbose: 是否打印诊断信息。

    Returns:
        (合谋者列表, 诊断信息字典)。
    """
    k_max = layout.design.K if max_colluders is None else int(max_colluders)
    k_max = max(1, min(k_max, len(client_models)))

    # v1 旧布局没有内置的密钥派生值，无法做归一化计数解码
    if layout.point_values is None:
        return [], {"enabled": False,
                    "reason": "布局为 v1（见证点值来自 encoder 切片），"
                              "不支持计数解码；请重新生成 witness_layout.json"}

    e = layout.extract_point_values(np.zeros(1))   # v2: 布局内置
    b = layout.extract_base_values()

    # ---- 逐副本 (p, j) 归一化成"比值" ----
    # 每个副本携带 e[p][j]（标记点）或 b[p][j]（未标记点），
    # 幅度各不相同，但**比值** r = (L - b) / (e - b) 与 j 无关：
    #     r = m_p / K
    # 因此可以逐副本归一化后再对 j 维做稳健聚合（对局部攻击鲁棒）。
    leak_m = _extract_point_matrix(leaked_model, layout, agg)   # (v, G)
    denom = e - b
    safe_denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_copy = (leak_m - b) / safe_denom
    per_copy = np.where(np.abs(denom) < 1e-12, np.nan, per_copy)

    # 对 j 维聚合（中位数）：少数副本被攻击掉时仍能恢复该点的比值
    r_obs_full = np.full(layout.v, np.nan, dtype=np.float64)
    for p in range(layout.v):
        row = per_copy[p]
        row = row[~np.isnan(row)]
        if row.size == 0:
            continue
        r_obs_full[p] = np.median(row) if agg == "median" else float(row.mean())

    valid = ~np.isnan(r_obs_full)
    if valid.sum() < 3:
        return [], {"enabled": False, "reason": "见证点有效点不足",
                    "best_error": None, "n_points": int(valid.sum())}

    r_obs = r_obs_full[valid]

    client_ids = sorted(client_models.keys())
    if not client_ids:
        return [], {"enabled": False, "reason": "没有候选客户端", "best_error": None}

    # A 的列按 client_id 索引；取出与候选客户端对应的列
    full_A = layout.A
    col_idx = []
    for cid in client_ids:
        if cid < 0 or cid >= full_A.shape[1]:
            return [], {"enabled": False,
                        "reason": f"client_id {cid} 超出关联矩阵范围",
                        "best_error": None}
        col_idx.append(cid)
    A = full_A[valid, :][:, col_idx]

    best_set, best_err = None, np.inf
    runner_err = np.inf
    best_scale = 1.0
    n_candidates = 0
    all_errs = []
    # 每个规模的最小残差（V 形曲线），用于自动推断合谋规模
    per_size: Dict[int, float] = {}
    # 每个规模的最优子集与残差（用于简约修正）
    best_per_size: Dict[int, Tuple[tuple, float, float]] = {}

    for r in range(1, k_max + 1):
        size_best = np.inf
        size_combo = None
        size_scale = 1.0
        for combo in combinations(range(len(client_ids)), r):
            idx = list(combo)
            # 预测计数：该点在候选集合中被标记的客户端数
            pred_count = A[:, idx].sum(axis=1)
            pred = pred_count / float(r)
            # 尺度精修：允许 pred 整体缩放（吸收"见证点未完全冻结"
            # 或幅度偏差带来的系统性偏移），最优尺度 s = <r_obs,pred>/<pred,pred>
            denom_s = float(np.sum(pred * pred))
            if denom_s < 1e-12:
                continue
            s = float(np.sum(r_obs * pred) / denom_s)
            err = float(np.sum((r_obs - s * pred) ** 2))
            n_candidates += 1
            all_errs.append((combo, err))
            if err < size_best:
                size_best = err
                size_combo = tuple(client_ids[i] for i in combo)
                size_scale = s
            if err < best_err:
                runner_err = best_err
                best_err = err
                best_set = tuple(client_ids[i] for i in combo)
                best_scale = s
            elif err < runner_err:
                runner_err = err
        per_size[r] = size_best if np.isfinite(size_best) else float("inf")
        if size_combo is not None:
            best_per_size[r] = (size_combo, size_best, size_scale)

    if best_set is None:
        return [], {"enabled": True, "reason": "无候选子集", "best_error": None}

    # ---- 简约修正（Occam）----
    # 与 estimate_collusion_size_by_count 同样的理由：见证点比值噪声
    # 可能抹平残差曲线的谷底，导致 argmin 高估合谋规模。
    # 若某个更小的规模 r' 的最优子集残差与全局最优相差不超过容差，
    # 就优先采用 r'（能用更少合谋者解释的，不选更复杂的解释）。
    finite = {r: e for r, e in per_size.items() if np.isfinite(e)}
    if finite:
        parsimony_r = _parsimony_size(finite, min(finite, key=lambda r: finite[r]))
        if (parsimony_r is not None and parsimony_r in best_per_size
                and parsimony_r < len(best_set)):
            cand_combo, cand_err, cand_scale = best_per_size[parsimony_r]
            # 更新 runner_err：次优应当是"排除所选解后"的最优
            others = [e for r, (c, e, _) in best_per_size.items()
                      if r != parsimony_r]
            runner_err = min(others) if others else np.inf
            best_set = cand_combo
            best_err = cand_err
            best_scale = cand_scale

    # 弃权判定：
    #   (1) 平均残差过大 -> 泄漏模型不是任何合谋集合的平均（证据不足）
    #   (2) 最优与次优差距不够 -> 无法区分候选合谋者
    margin = (runner_err - best_err) if np.isfinite(runner_err) else np.inf
    n_pts = int(valid.sum())
    avg_err = best_err / max(n_pts, 1)

    abstained = False
    reason = None
    if avg_err > max_avg_error:
        abstained = True
        reason = f"平均残差过大 ({avg_err:.4f} > {max_avg_error})"
    elif np.isfinite(margin) and margin <= min_margin:
        abstained = True
        reason = f"最优与次优差距过小 (margin={margin:.4f})"

    found = [] if abstained else list(best_set)
    diag = {
        "enabled": True,
        "best_error": best_err,
        "avg_error": avg_err,
        "scale": best_scale,
        "margin": margin,
        "runner_up_error": runner_err if np.isfinite(runner_err) else None,
        "n_candidates": n_candidates,
        "n_points": n_pts,
        "abstained": abstained,
        "reason": reason,
        "observed_ratios": r_obs.tolist(),
        "per_size_error": per_size,
    }
    if verbose:
        print(f"[*] 计数解码: 合谋者 = {found} (avg_err={avg_err:.4f}, "
              f"margin={margin:.4f}, 有效点={n_pts}"
              + (f", 已弃权: {reason}" if abstained else "") + ")")
    return found, diag


def estimate_collusion_size_by_count(
    leaked_model: Dict[str, np.ndarray],
    layout: "WitnessLayout",
    max_colluders: Optional[int] = None,
    agg: str = "median",
) -> Tuple[Optional[int], Dict[int, float]]:
    """在**计数域**推断合谋规模 K。

    对每个候选规模 r，遍历所有大小为 r 的客户端子集，取残差最小者，
    记为 E(r)。真正的合谋规模对应 E(r) 的最小值（V 形曲线）。

    相比 v1 的 `estimate_collusion_size`（在原始参数残差上做 V 形），
    本函数在归一化后的计数域上做，因此：
      - 不受见证点幅度差异影响；
      - 对见证点区域的噪声 / 漂移更鲁棒。

    Args:
        leaked_model: 泄漏模型参数字典。
        layout: 见证点布局。
        max_colluders: 最大搜索规模；None 时用 layout.design.K。
        agg: 点内副本聚合方式。

    Returns:
        (推断的 K 或 None, {规模: 最小残差} 字典)。
    """
    if layout.point_values is None:
        return None, {}

    k_max = layout.design.K if max_colluders is None else int(max_colluders)
    k_max = max(1, min(k_max, layout.n_clients))

    e = layout.extract_point_values(np.zeros(1))
    b = layout.extract_base_values()
    leak_m = _extract_point_matrix(leaked_model, layout, agg)
    denom = e - b
    safe_denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_copy = (leak_m - b) / safe_denom
    per_copy = np.where(np.abs(denom) < 1e-12, np.nan, per_copy)

    r_obs_full = np.full(layout.v, np.nan, dtype=np.float64)
    for p in range(layout.v):
        row = per_copy[p]
        row = row[~np.isnan(row)]
        if row.size == 0:
            continue
        r_obs_full[p] = np.median(row) if agg == "median" else float(row.mean())

    valid = ~np.isnan(r_obs_full)
    if valid.sum() < 3:
        return None, {}
    r_obs = r_obs_full[valid]
    A = layout.A[valid, :]

    per_size: Dict[int, float] = {}
    for r in range(1, k_max + 1):
        best = np.inf
        for combo in combinations(range(layout.n_clients), r):
            pred = A[:, list(combo)].sum(axis=1) / float(r)
            denom_s = float(np.sum(pred * pred))
            if denom_s < 1e-12:
                continue
            s = float(np.sum(r_obs * pred) / denom_s)
            err = float(np.sum((r_obs - s * pred) ** 2))
            if err < best:
                best = err
        per_size[r] = best if np.isfinite(best) else float("inf")

    finite = {r: e for r, e in per_size.items() if np.isfinite(e)}
    if not finite:
        return None, per_size
    best_r = min(finite, key=lambda r: finite[r])

    # ---- 简约修正（Occam）：在"足够好"的候选里取最小的规模 ----
    # 为什么需要：
    #   见证点比值 r_obs = m_p/K + 噪声/e_p，而 |e_p| 最小仅 0.025，
    #   漂移稍大就会让比值噪声达到信号量级（m_p/K ∈ [0,1]）的同一水平。
    #   此时残差曲线 E(r) 的谷底会被抹平：E(3) 可能略低于 E(2)，
    #   直接取 argmin 就会**高估**合谋规模，随后解码在错误的规模里
    #   搜索，把"实际合谋者 + 一个无辜者"的子集排在真解前面。
    #
    #   简约规则：只要某个更小的规模 r' 的残差与最优值相差不超过
    #   阈值，就认为 r' 已经能解释观测，取其中最小者。
    #   这符合奥卡姆剃刀：能用更少合谋者解释的，不选更复杂的解释。
    parsimony_r = _parsimony_size(finite, best_r)
    if parsimony_r is not None and parsimony_r < best_r:
        best_r = parsimony_r

    return best_r, per_size


#: 简约修正的残差容差（相对最优值的比例）。
#: 用**相对**而非绝对容差，是因为残差 E(r) 的绝对量级随规模 r 变化
#: （候选点更多 -> 残差累加更多），固定绝对阈值无法同时适配
#: K=2 与 K=3：绝对容差偏大时会把 K=3 压成 K=2（低估规模）。
_SIZE_PARSIMONY_REL_TOL = 0.15

#: 简约修正的残差容差（绝对值下限，用于残差整体极小时的兜底）。
_SIZE_PARSIMONY_ABS_TOL = 0.0


def _parsimony_size(finite: Dict[int, float],
                    best_r: int,
                    rel_tol: float = _SIZE_PARSIMONY_REL_TOL,
                    abs_tol: float = _SIZE_PARSIMONY_ABS_TOL) -> Optional[int]:
    """在残差曲线上做简约修正，返回"足够好"的最小规模。

    Args:
        finite: {规模: 最小残差}（只含有限值）。
        best_r: 残差最小的规模。
        rel_tol: 相对容差（相对最优残差）。
        abs_tol: 绝对容差下限。

    Returns:
        简约修正后的规模；若无需修正则返回 ``best_r`` 本身。
    """
    if not finite:
        return None
    emin = min(finite.values())
    thresh = emin + max(abs_tol, rel_tol * emin)
    candidates = [r for r in finite if finite[r] <= thresh]
    if not candidates:
        return best_r
    return min(candidates)


def _extract_feature(model: Dict[str, np.ndarray],
                     position_mask: Optional[List[Tuple[str, int]]],
                     design: WitnessDesign) -> np.ndarray:
    """从模型参数中提取用于匹配的特征向量。"""
    if position_mask is not None:
        vals = []
        for param_name, local_idx in position_mask:
            if param_name not in model:
                continue
            flat = np.asarray(model[param_name]).reshape(-1)
            if 0 <= local_idx < flat.shape[0]:
                vals.append(float(flat[local_idx]))
        if vals:
            return np.asarray(vals, dtype=np.float64)

    if design.positions:
        vals = []
        for param_name, local_idx in design.positions:
            if param_name not in model:
                continue
            flat = np.asarray(model[param_name]).reshape(-1)
            if 0 <= local_idx < flat.shape[0]:
                vals.append(float(flat[local_idx]))
        if vals:
            return np.asarray(vals, dtype=np.float64)

    # 退路：按参数名排序，把所有数值拼起来
    parts = []
    for name in sorted(model.keys()):
        arr = np.asarray(model[name], dtype=np.float64).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(1)


# ====================================================================
# 7. 仿真
# ====================================================================

def estimate_collusion_size(
    leaked_model: Dict[str, np.ndarray],
    client_models: Dict[int, Dict[str, np.ndarray]],
    design: WitnessDesign,
    max_colluders: int = 5,
    position_mask: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[int, float]:
    """在不知道合谋规模的前提下推断合谋者数量 K。

    原理：对每个候选规模 r，计算该规模下的**最优匹配残差**
    （取所有大小为 r 的子集里残差最小者）。残差随 r 呈 V 形：

        r < K : 欠拟合，残差高
        r = K : 恰好匹配，残差最小
        r > K : 过度稀释，残差回升

    因此取 argmin_r 即可得到 K 的估计。

    返回 (推断的 K, 该 K 下的最优残差)。
    """
    leak_vec = _extract_feature(leaked_model, position_mask, design)
    cand_vecs = {cid: _extract_feature(m, position_mask, design)
                 for cid, m in client_models.items()}
    client_ids = sorted(cand_vecs.keys())
    max_r = max(1, min(max_colluders, len(client_ids)))

    M = np.stack([cand_vecs[c] for c in client_ids])
    G = M @ M.T
    Ldot = M @ leak_vec
    Lnorm = float(leak_vec @ leak_vec)

    per_r = {}
    for r in range(1, max_r + 1):
        best = np.inf
        for combo in combinations(range(len(client_ids)), r):
            idx = np.fromiter(combo, dtype=np.intp, count=r)
            err = Lnorm - 2.0 * Ldot[idx].sum() / r + G[np.ix_(idx, idx)].sum() / (r * r)
            if err < best:
                best = err
        per_r[r] = best

    # V 形曲线：取残差最小的规模
    best_r = min(per_r, key=lambda r: per_r[r])
    return best_r, per_r[best_r]


def simulate_collusion(
    client_models: Dict[int, Dict[str, np.ndarray]],
    colluders: Sequence[int],
    design: WitnessDesign,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """模拟合谋：把合谋者的模型做参数平均。

    Args:
        client_models: {client_id: 模型参数字典}。
        colluders: 合谋者 ID 列表。
        design: 见证点设计。
        noise_scale: 平均后附加的高斯噪声标准差（相对水印尺度）。
        rng: 随机数生成器。

    Returns:
        泄漏模型参数字典。
    """
    rng = rng or np.random.default_rng(0)
    colluders = list(colluders)
    if not colluders:
        raise ValueError("colluders 不能为空")

    keys = list(client_models[colluders[0]].keys())
    out = {}
    for k in keys:
        acc = None
        for c in colluders:
            arr = np.asarray(client_models[c][k], dtype=np.float64)
            acc = arr.copy() if acc is None else acc + arr
        acc /= len(colluders)
        if noise_scale > 0:
            acc = acc + noise_scale * rng.normal(size=acc.shape)
        out[k] = acc
    return out
