"""
抗合谋水印（Collusion-Resistant Watermark, CRW）
=================================================

针对"多个恶意客户端将各自收到的定制模型做参数平均，以削弱水印并阻止溯源"
这一威胁模型，提供**精确合谋归因**能力：识别出到底是哪 K 个客户端合谋。

核心思想
--------
1. **共享见证点池（Shared Witness Pool）**
   所有客户端共享同一批"见证点"位置池。点 p 的权威值 e_p 取自
   自编码器编码器参数（保持 encoder 水印语义）。

   BIBD 关联矩阵 A ∈ {0,1}^{v×n} 决定谁在哪个点写入权威值：

       A[p, c] = 1  ->  客户端 c 在点 p 写入 e_p
       A[p, c] = 0  ->  客户端 c 在点 p 写入全局模型原值 w_t[p]

   (v,k,1)-BIBD 保证任意两客户端的标记点集合交 <= 1，
   因此各客户端定制模型在水印区域上高度可分。

2. **子集匹配解码（Subset Matching）**
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

import json
import os
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
    "embed_witness_watermark",
    "decode_colluders",
    "estimate_collusion_size",
    "simulate_collusion",
    "DEFAULT_ENCODER_PARAMS",
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


@dataclass
class WitnessDesign:
    """见证点水印的完整设计。

    Attributes:
        n_clients: 客户端数。
        K: 期望抵抗的最大合谋规模（用于限制搜索集合）。
        v: 见证点数。
        k: 每客户端标记的点数（BIBD 块大小）。
        G: 每见证点占用的模型位置数。
        A: 关联矩阵，shape (v, n_clients)。
        singer: 对应的 Singer 差集参数。
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
        return (f"见证点设计: n={self.n_clients} K={self.K} "
                f"q={self.singer.q} v={self.v} k={self.k} G={self.G}\n"
                f"  关联矩阵 rank = {np.linalg.matrix_rank(self.A)}/{self.n_clients}\n"
                f"  行权分布 = {np.bincount(rw, minlength=self.k + 1).tolist()}\n"
                f"  位置池 = {self.n_witness_positions} 个 "
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
        return cls(
            n_clients=d["n_clients"], K=d["K"], v=d["v"], k=d["k"],
            G=d["G"], A=np.array(d["A"], dtype=np.float64), singer=singer,
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
        A = build_incidence_matrix(singer.v, n_clients, singer.k, singer.D)
        if np.linalg.matrix_rank(A) < n_clients:
            continue
        rw = A.sum(axis=1)
        n_ge2 = int((rw >= 2).sum())
        score = (n_ge2, -float(rw.var()))
        if best is None or score > best[0]:
            best = (score, singer, A)

    if best is None:
        raise ValueError(
            f"没有可行的 Singer 参数：需要 v=q^2+q+1 >= {n_clients}，"
            f"但候选 q={list(q_candidates)} 均不满足或关联矩阵不满秩")

    _, singer, A = best
    v, k = singer.v, singer.k

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

    与 ``WitnessDesign`` 的区别：Layout 额外记录了权威值在
    encoder 参数向量中的**切片来源**，使得嵌入端与解码端可以
    在不同进程、不同轮次之间稳定复现同一套映射。

    权威值映射规则
    --------------
    encoder 参数向量共 ``encoder_params`` 个元素。它被划分为：

        [ 独占切片区 [0, n_clients*slice_size) ]   <- 现有机制，不变
        [ 见证点权威值区 [0, v*G) ]               <- 与上者**值空间重叠**

    - 独占切片区：客户端 c 的私有权威值取自
      ``encoder_params[offset_c : offset_c + len(P_c)]``。
      现有 main.py 的 offset 计算逻辑**完全不变**。
    - 见证点权威值区：见证点 p 的 G 个权威值取自
      ``encoder_params[p*G : (p+1)*G]``，
      **所有标记点 p 的客户端共享同一份**（共享见证点，
      是抗合谋能力的来源）。

    为什么允许值空间重叠？
    ------------------
    encoder 参数总量恰好等于 n 个独占切片所需总量（9472 = 948*2 + 947*8），
    **没有空闲区间**可供见证点独占。因此见证点复用 encoder 参数的
    前 v*G 个元素。

    这是安全的，因为两种水印的**位置空间完全不相交**：
    见证点位置取自"所有未被任何客户端分配"的 conv 位置，
    而独占切片位置与之无交集。因此：

    - 所有权认证只检查客户端 c 的切片位置 P_c，
      不会把共享的见证点值误判为 c 的证据；
    - 见证点解码只检查见证点位置，
      不会受某客户端私有切片值的影响。

    Attributes:
        design: 见证点设计。
        slice_size: 每客户端独占切片长度。
        witness_offset: 见证点权威值区在 encoder 参数向量中的起点。
        slice_offset: 独占切片区在 encoder 参数向量中的起点。
    """
    design: WitnessDesign
    slice_size: int
    witness_offset: int = 0
    slice_offset: int = 0

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
        """点 point 的权威值在 encoder 参数向量中的区间 [start, end)。"""
        start = self.witness_offset + point * self.G
        return start, start + self.G

    def extract_point_values(self, encoder_params: np.ndarray) -> np.ndarray:
        """从 encoder 参数向量中提取 (v, G) 的权威值矩阵。"""
        arr = np.asarray(encoder_params, dtype=np.float64).reshape(-1)
        out = np.zeros((self.v, self.G))
        for p in range(self.v):
            s, e = self.encoder_slice_for_point(p)
            seg = arr[s:e]
            if seg.size < self.G:
                seg = np.pad(seg, (0, self.G - seg.size), mode="edge")
            out[p] = seg[:self.G]
        return out

    def to_dict(self) -> dict:
        return {
            "design": self.design.to_dict(),
            "slice_size": self.slice_size,
            "witness_offset": self.witness_offset,
            "slice_offset": self.slice_offset,
            "positions": [list(p) for p in self.positions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WitnessLayout":
        design = WitnessDesign.from_dict(d["design"])
        design.positions = [(p[0], p[1]) for p in d.get("positions", [])]
        return cls(design=design, slice_size=d["slice_size"],
                   witness_offset=d.get("witness_offset", 0),
                   slice_offset=d.get("slice_offset", 0))


def build_witness_layout(
    n_clients: int,
    K: int = 3,
    encoder_params: int = DEFAULT_ENCODER_PARAMS,
    positions: Optional[List[Tuple[str, int]]] = None,
    slice_size: Optional[int] = None,
    verbose: bool = True,
) -> WitnessLayout:
    """构造可持久化的见证点布局。

    Args:
        n_clients: 客户端数。
        K: 期望抵抗的最大合谋规模。
        encoder_params: encoder 参数总量。
        positions: 共享见证点位置池 [(param_name, local_idx), ...]。
        slice_size: 每客户端独占切片长度；None 时按 encoder_params/n_clients
            自动计算。
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

    # encoder 参数向量的预算分配：
    #   [ 独占切片区 [0, n*slice_size) ]  —— 现有机制，offset 不变
    #   [ 见证点权威值区 [0, v*G) ]      —— 复用前段（值空间重叠）
    # 位置空间则完全隔离：见证点位置来自"未被任何客户端分配"的位置。
    #
    # 注意：独占切片的总量可能略超 encoder_params（equal 策略的余数
    # 分配所致，如 n=10 时 948*2 + 947*8 = 9480 > 9472）。这是**既有
    # 行为**，现有代码用 `enc_global_idx < len(encoder_params)` 兜底。
    # 见证点布局不改变这一行为，也不因此失败。
    need = design.v * design.G
    if need > encoder_params:
        raise ValueError(
            f"见证点权威值区需要 {need} 个 encoder 参数，超过总量 "
            f"{encoder_params}")

    design.positions = list(positions[:need])
    layout = WitnessLayout(design=design, slice_size=slice_size,
                            witness_offset=0, slice_offset=0)
    if verbose:
        print(f"[*] {layout.summary()}")
    return layout


def _layout_summary(self: WitnessLayout) -> str:
    rw = self.design.row_weights
    return (f"见证点布局: n={self.n_clients} K={self.design.K} "
            f"q={self.design.singer.q} v={self.v} k={self.design.k} G={self.G}\n"
            f"  关联矩阵 rank = {np.linalg.matrix_rank(self.A)}/{self.n_clients}\n"
            f"  行权分布 = {np.bincount(rw, minlength=self.design.k + 1).tolist()}\n"
            f"  见证点权威值区 = [{self.witness_offset}, "
            f"{self.witness_offset + self.v * self.G})  (与切片区值空间重叠)\n"
            f"  独占切片区 = [{self.slice_offset}, {self.slice_end})  "
            f"({self.slice_size}/客户端)\n"
            f"  见证点位置 = {len(self.positions)} 个 (与切片位置不相交)")


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
    marked = design.client_points(client_id)

    # 解析权威值 -> (v, G)
    ev = _resolve_point_values(encoder_values, design, client_id)

    out = {k: (v_.clone() if hasattr(v_, "clone") else np.array(v_, copy=True))
           for k, v_ in model_params.items()}

    if point_values is not None:
        pv = np.asarray(point_values, dtype=np.float64).reshape(v, G)
    else:
        pv = ev

    for p in marked:
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
                flat[local_idx] = pv[p, j]

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

    marked = layout.design.client_points(client_id)
    G = layout.G

    for p in marked:
        for j in range(G):
            pos_idx = p * G + j
            if pos_idx >= len(layout.positions):
                break
            param_name, local_idx = layout.positions[pos_idx]
            if param_name not in model_params:
                continue
            flat = model_params[param_name].view(-1)
            if 0 <= local_idx < flat.numel():
                flat[local_idx] = float(point_values[p, j])

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
