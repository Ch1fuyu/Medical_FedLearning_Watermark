"""
BIBD-ACC 浮点水印码本生成脚本（严格版）
==============================================================
目标：构造 n 列的 0/1 码本 C in {0,1}^{v x n}，满足 K-弹性 AND 唯一性：
      任意两个不同的、大小不超过 K 的客户端集合 S != T，
      其码向量逐位 AND 的结果必须不同：
          AND_{i in S} C[:, i]  !=  AND_{i in T} C[:, i]

---------------------------------------------------------------
1) 参数求解
   (v, k, 1)-BIBD 的用户容量公式为   n = (v^2 - v) / (k^2 - k)，  k = K + 1。
   给定 n 与 K（即 k = K+1），解一元二次方程
        v^2 - v - n (k^2 - k) = 0
   取正整数解 v。若精确解不存在，则搜索最小可行参数 (v, k)。

   一致性说明（脚本内置检查）：
     差集条件要求 v - 1 = k(k-1)。代入容量公式得
        n = (v^2 - v) / (k^2 - k) = v  <=>  v = n
   因此"精确解"存在当且仅当  n = K(K+1) + 1  （此时 v = n，n 客户端
   恰好用满 Singer 差集给出的全部 v 个平移块）。其它 n 走最小可行参数搜索。

---------------------------------------------------------------
2) BIBD 关联矩阵构造（循环差集 / Singer 差集）
   构造大小为 k 的集合 D subset Z_v，使得 D 中任意两个元素的有序差
   （模 v）恰好覆盖 1, 2, ..., v-1 各一次（D 为 (v, k, 1)-差集）。
   构造关联矩阵 M in {0,1}^{v x n}：第 i 列在位置 {(d+i) mod v : d in D}
   处为 1，其余为 0；每列恰好 k 个 1。
   这是 (v, k, 1)-BIBD 的关联矩阵：任意两列的交不超过 1，每对元素
   恰好共同出现在一个块中。

   构造后必须严格验证：
     (a) 每列恰好 k 个 1；
     (b) 任意两列的内积 <= 1（任意两块的交不超过 1 个元素）；
     (c) 列数 >= n（取前 n 列作为码本）。

   差集来源优先级：
     1) Singer 差集（q 为素数，v = q^2+q+1, k = q+1）—— 精确构造；
     2) 暴力搜索（小 v）；
     3) 随机爬山搜索（近似差集：任意两块的交 <= 1 仍严格成立）；
     4) 通用贪心打包（任意 v, k：任两块交 <= 1）。

---------------------------------------------------------------
3) 码本生成（关键约定）
   C = M[:, 0:n] in {0,1}^{v x n}，码向量直接取关联矩阵的列，不取补。

---------------------------------------------------------------
4) AND 唯一性验证（严格执行）
   枚举所有大小 1..K 的客户端子集，用哈希表检查是否存在两个"不同集合"
   得到同一个 AND 向量；冲突会被完整报告（含反例集合对）。

   结构性事实（脚本自动诊断）：
     * (v,k,1)-BIBD 的 AND 唯一性只在 K = 1 时严格成立；
       在 K >= 2 时，投影平面上同一点处有多条线（例如 PG(2,3) 的每点
       4 条线，C(4,2) = 6 对线交于同一点），故 AND(线对) 发生碰撞。
     * 原因：AND(S) 只反映"公共块"，其分辨力受 BIBD 的点-块关联结构限制。
     * 要在 K >= 2 时严格成立，必须额外加入"高阶签名行"，即
       对每个 K 元集合 Q 加一行，客户端 i 在该行为 1 当且仅当 i in Q。
       这部分单独即可保证唯一性（证明见 generate_signature_rows）。

   模式：
     'bibd'   —— 只用 BIBD 构造（保留经典 BIBD 结构，AND 唯一性仅 K=1 成立，
                 脚本会照实报错并给出反例）；
     'hybrid' —— BIBD 骨架 + K 元子集签名行（推荐：既有 BIBD 的"每列恰 k
                 个 1 / 两两内积 <= 1"结构，又严格满足 K-弹性 AND 唯一性）；
     'exact'  —— 纯 K 元子集地面集（最简洁，理论下界行数 = C(n,K)）；
     'auto'   —— 优先 hybrid，不可用时回退 exact。

---------------------------------------------------------------
5) 浮点映射
   0 -> mu - delta，1 -> mu + delta（mu/delta 可为标量或长度 v 的数组）。

---------------------------------------------------------------
命令行示例：
    python utils/bibd_acc_codebook.py --n_clients 13 --K 3 --mode hybrid \
        --mu 0.0 --delta 0.01 --out save/bibd_codebook.npz
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from math import comb, isqrt

import numpy as np


# ====================================================================
# 1. 参数求解
# ====================================================================
def solve_bibd_params(n_clients: int, K: int):
    """
    精确解 n = (v^2 - v) / (k^2 - k)，k = K + 1。
    即解一元二次方程 v^2 - v - n(k^2 - k) = 0 的正整数根。

    返回 (v, k)；不存在精确解时返回 None。

    注：由于差集条件 v - 1 = k(k-1)，精确解存在当且仅当
        n = k(k-1) + 1 = K(K+1) + 1（此时 v = n）。
    """
    if n_clients <= 0:
        raise ValueError("n_clients 必须为正整数")
    if K < 0:
        raise ValueError("K 必须 >= 0")

    k = K + 1
    denom = k * k - k
    if denom <= 0:                       # K = 0 -> k = 1，任意 v 均可，v = n
        return n_clients, k

    disc = 1 + 4 * n_clients * denom
    s = isqrt(disc)
    if s * s != disc:
        return None
    if (1 + s) % 2 != 0:
        return None

    v = (1 + s) // 2
    if v * v - v == n_clients * denom:
        # 循环差集只能给出 v 个平移块，故要求 v >= n_clients 才能取前 n 列。
        if v >= n_clients:
            return v, k
    return None


def find_feasible_params(n_clients: int, K: int, max_search: int = 500):
    """
    精确解不存在时，搜索最小可行参数 (v, k)：
        * k 从 K+1 向上搜索；
        * 优先取差集可行值 v = k(k-1)+1（此时 v 个平移块，容量 = v）；
          若 v >= n_clients 即满足"列数 >= n"；
        * 否则退回通用打包：取 v = max(n_clients, k(k-1)+1)，贪心选取
          n_clients 个 k-块，两两交 <= 1。

    返回 dict：{v, k, n_capacity, n_clients, exact, source}
    """
    for k in range(K + 1, K + 1 + max_search):
        denom = k * k - k
        if denom <= 0:
            return {"v": n_clients, "k": k, "n_capacity": n_clients,
                    "n_clients": n_clients, "exact": False, "source": "trivial"}

        # (A) 差集情形：v = k(k-1) + 1，块数 = v
        v_ds = denom + 1
        if v_ds >= n_clients:
            return {"v": v_ds, "k": k, "n_capacity": v_ds,
                    "n_clients": n_clients, "exact": False, "source": "difference_set"}

        # (B) 通用打包：v 个点上的 k-块，两两交 <= 1，最多可取
        #     floor(C(v,2) / C(k,2)) 个块；取 v 使其 >= n_clients
        v = max(n_clients, v_ds)
        while True:
            pair_budget = (v * (v - 1) // 2) // max(1, (k * (k - 1) // 2))
            if pair_budget >= n_clients:
                break
            v += 1
        return {"v": v, "k": k, "n_capacity": pair_budget,
                "n_clients": n_clients, "exact": False, "source": "packing"}

    raise ValueError("无法找到可行参数，请增大 K 或减小客户端数量")


# ====================================================================
# 2. 差集构造
# ====================================================================
# ------------------------------------------------------------------------
# 2.1 有限域 GF(q^3) 工具（用于 Singer 差集；q 为素数）
# ------------------------------------------------------------------------
def _gf_find_irreducible_cubic(q: int):
    """
    在 GF(q) 上找一个首一不可约三次多项式 f(x) = x^3 + a x^2 + b x + c（c != 0）。
    三次多项式不可约 <=> 无根。返回 (c, b, a)。
    """
    for a in range(q):
        for b in range(q):
            for c in range(1, q):
                has_root = False
                for r in range(q):
                    val = (c + b * r + a * r * r + pow(r, 3, q)) % q
                    if val == 0:
                        has_root = True
                        break
                if not has_root:
                    return c, b, a
    return None


class _GF3:
    """GF(q^3) = GF(q)[x]/(f)，元素表示为 (a0, a1, a2) = c0 + c1 x + c2 x^2。"""

    def __init__(self, q: int, f):
        self.q = q
        self.f = f                       # (c, b, a): x^3 + a x^2 + b x + c
        self._na = (-f[2]) % q
        self._nb = (-f[1]) % q
        self._nc = (-f[0]) % q

    def mul(self, x, y):
        q = self.q
        conv = [0] * 5
        for i in range(3):
            xi = x[i]
            if xi == 0:
                continue
            for j in range(3):
                conv[i + j] = (conv[i + j] + xi * y[j]) % q
        for d in (4, 3):
            coef = conv[d]
            if coef:
                conv[d - 1] = (conv[d - 1] + coef * self._na) % q
                conv[d - 2] = (conv[d - 2] + coef * self._nb) % q
                conv[d - 3] = (conv[d - 3] + coef * self._nc) % q
                conv[d] = 0
        return (conv[0], conv[1], conv[2])

    def pow(self, x, e):
        result = (1, 0, 0)
        base = x
        while e > 0:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result

    def trace(self, x):
        """Tr_{GF(q^3)/GF(q)}(x) = x + x^q + x^{q^2}。"""
        xq = self.pow(x, self.q)
        xq2 = self.pow(xq, self.q)
        return sum(((x[i] + xq[i] + xq2[i]) % self.q) for i in range(3)) % self.q

    def find_primitive_element(self):
        """寻找 GF(q^3)^* 的本原元（阶 q^3 - 1）。"""
        order = self.q ** 3 - 1
        primes = []
        m, p = order, 2
        while p * p <= m:
            if m % p == 0:
                primes.append(p)
                while m % p == 0:
                    m //= p
            p += 1
        if m > 1:
            primes.append(m)

        cands = [(c0, c1, c2)
                 for c0 in range(self.q)
                 for c1 in range(self.q)
                 for c2 in range(self.q)
                 if not (c0 == c1 == c2 == 0)]
        cands.sort(key=lambda e: sum(e))
        for g in cands:
            ok = True
            for p in primes:
                if self.pow(g, order // p) == (1, 0, 0):
                    ok = False
                    break
            if ok:
                return g
        return None


def singer_difference_set(q: int):
    """
    Singer 差集（q 为素数）：构造 (v, k, 1) = (q^2+q+1, q+1, 1) 差集 D ⊂ Z_v。

    原理：取 GF(q^3) 的本原元 α，V = GF(q^3)^* / GF(q)^* 是阶
    v = q^2+q+1 的循环群；projectivized 直线
        B_d = { e in V : Tr(d e) = 0 }
    在 V 中恰有 q+1 个点，且任意两条直线恰交于一点  =>  D 是 (v,q+1,1) 差集。
    实现上 D = { i in [0, v) : Tr(α^i) = 0 }。

    返回 (v, k, sorted(list(D)))；失败返回 None。
    """
    if not _is_prime(q):
        return None
    f = _gf_find_irreducible_cubic(q)
    if f is None:
        return None
    gf = _GF3(q, f)
    alpha = gf.find_primitive_element()
    if alpha is None:
        return None

    v = q * q + q + 1
    k = q + 1
    D = []
    cur = (1, 0, 0)
    for i in range(v):
        if gf.trace(cur) == 0:
            D.append(i)
        cur = gf.mul(cur, alpha)

    if len(D) != k:
        return None
    if not is_difference_set(v, k, D):
        return None
    return v, k, sorted(D)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# ------------------------------------------------------------------------
# 2.2 差集性质与通用搜索
# ------------------------------------------------------------------------
def is_difference_set(v: int, k: int, D) -> bool:
    """
    检查 D 是否为 (v, k, 1)-差集：D 中任意两个元素的有序差（模 v）
    恰好覆盖 1, 2, ..., v-1 各一次。
    """
    D = list(D)
    if len(D) != k:
        return False
    diffs = []
    for a, b in combinations(D, 2):
        diffs.append((a - b) % v)
        diffs.append((b - a) % v)
    if len(diffs) != v - 1:
        return False
    return len(set(diffs)) == v - 1 and 0 not in diffs


def brute_force_difference_set(v: int, k: int):
    """小参数暴力搜索 (v, k, 1)-差集。"""
    if k < 2 or v > 64:
        return None
    for D in combinations(range(v), k):
        if is_difference_set(v, k, D):
            return list(D)
    return None


def heuristic_difference_set(v: int, k: int, rng=None, iters: int = 20000):
    """
    随机局部搜索近似差集（爬山法）：最小化重复差值的个数。
    成功后任意两块交 <= 1 严格成立（"部分差集"）。
    """
    rng = rng if rng is not None else np.random.default_rng(12345)
    if k < 2 or k > v:
        return None

    def score(D):
        seen = set()
        bad = 0
        for a, b in combinations(D, 2):
            for d in ((a - b) % v, (b - a) % v):
                if d in seen:
                    bad += 1
                else:
                    seen.add(d)
        return bad

    best, best_score = None, None
    restarts = max(1, iters // 1000)
    for restart in range(restarts):
        D = [int(x) for x in rng.choice(v, size=k, replace=False)]
        cur = score(D)
        for _ in range(max(1, iters // restarts)):
            i = int(rng.integers(k))
            j = int(rng.integers(v))
            if j in D:
                continue
            old = D[i]
            D[i] = j
            new = score(D)
            if new <= cur:
                cur = new
            else:
                D[i] = old
            if cur == 0:
                return sorted(D)
        if best_score is None or cur < best_score:
            best_score, best = cur, list(D)
    if best_score == 0:
        return sorted(best)
    return None


def find_difference_set(v: int, k: int, verbose: bool = True):
    """
    寻找 (v, k, 1)-差集。优先级：
      1) Singer 构造（当 v = q^2+q+1 且 q 为素数）
      2) 暴力搜索（v 较小）
      3) 随机爬山搜索
    返回 sorted list D；失败返回 None。
    """
    q = _solve_quadratic_q(v)
    if q is not None and q + 1 == k:
        res = singer_difference_set(q)
        if res is not None:
            if verbose:
                print(f"    差集来源: Singer 构造 (q={q}) -> D={res[2]}")
            return res[2]

    D = brute_force_difference_set(v, k)
    if D is not None:
        if verbose:
            print(f"    差集来源: 暴力搜索 -> D={D}")
        return D

    D = heuristic_difference_set(v, k)
    if D is not None:
        if verbose:
            print(f"    差集来源: 随机爬山 -> D={D}")
        return D
    return None


def _solve_quadratic_q(v: int):
    """求 v = q^2 + q + 1 的整数 q（不存在返回 None）。"""
    disc = 4 * v - 3
    s = isqrt(disc)
    if s * s != disc:
        return None
    if (s - 1) % 2 != 0:
        return None
    q = (s - 1) // 2
    if q * q + q + 1 == v:
        return q
    return None


# ====================================================================
# 3. 关联矩阵构造与验证
# ====================================================================
def build_cyclic_bibd_matrix(v: int, n_blocks: int, k: int, D):
    """
    由循环差集 D 构造关联矩阵 M in {0,1}^{v x n_blocks}：
    第 i 列在位置 {(d+i) mod v : d in D} 处为 1，其余为 0。
    每列恰好 k 个 1。列数取 min(v, n_blocks) 个平移块。
    """
    n_use = min(v, n_blocks)
    M = np.zeros((v, n_use), dtype=np.int8)
    for i in range(n_use):
        for d in D:
            M[(d + i) % v, i] = 1
    return M


def build_packing_matrix(v: int, n_blocks: int, k: int, seed: int = 0):
    """
    通用贪心打包：在 v 个点上选取 n_blocks 个 k-块，任意两块交 <= 1。
    （等价于 (v, k, 1)-packing / 部分 BIBD）
    """
    rng = np.random.default_rng(seed)
    selected, used_pairs = [], set()
    order = list(rng.permutation(v))
    attempts = 0
    max_attempts = 200 * n_blocks + 1000
    while len(selected) < n_blocks and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(v))
        cand = list(rng.choice(v, size=k, replace=False))
        pairs = set(combinations(sorted(int(x) for x in cand), 2))
        if pairs & used_pairs:
            continue
        selected.append(sorted(int(x) for x in cand))
        used_pairs |= pairs
    if len(selected) < n_blocks:
        return None
    M = np.zeros((v, n_blocks), dtype=np.int8)
    for j, blk in enumerate(selected):
        M[blk, j] = 1
    return M


def validate_bibd_matrix(M: np.ndarray, k: int, n_needed: int):
    """
    严格验证关联矩阵：
      (a) 每列恰好 k 个 1；
      (b) 任意两列的内积 <= 1（任意两块的交不超过 1 个元素）；
      (c) 列数 >= n_needed。
    返回 (ok, report dict)。
    """
    v, ncols = M.shape
    col_sums = M.sum(axis=0)
    bad_cols = np.where(col_sums != k)[0]

    inter = (M.astype(np.int64).T @ M.astype(np.int64))
    np.fill_diagonal(inter, 0)
    max_inter = int(inter.max()) if inter.size else 0
    bad_pairs = np.argwhere(inter > 1) if inter.size else []

    ok = (len(bad_cols) == 0) and (max_inter <= 1) and (ncols >= n_needed)
    report = {
        "ok": bool(ok),
        "v": int(v),
        "n_cols": int(ncols),
        "k": int(k),
        "n_needed": int(n_needed),
        "col_weight_all_k": bool(len(bad_cols) == 0),
        "bad_col_indices": bad_cols[:20].tolist(),
        "max_pairwise_inner_product": max_inter,
        "pairwise_inter_le_1": bool(max_inter <= 1),
        "num_bad_column_pairs": int(len(bad_pairs)),
        "bad_column_pairs_examples": bad_pairs[:10].tolist(),
    }
    return bool(ok), report


# ====================================================================
# 4. 码本生成
# ====================================================================
def generate_bibd_codebook(n_clients: int, K: int, verbose: bool = True):
    """
    按 BIBD 路线生成 0/1 码本：
      1) 求 (v, k)（优先精确解，否则最小可行参数）；
      2) 找 (v,k,1)-差集 D；
      3) 构造关联矩阵 M in {0,1}^{v x n}；
      4) 严格验证 M（每列恰 k 个 1 / 两两内积 <= 1 / 列数 >= n）；
      5) 取前 n 列作为码本 C（直接取列，不取补）。
    返回 (codebook, info)。
    """
    params = solve_bibd_params(n_clients, K)
    exact = params is not None
    if exact:
        v, k = params
        n_cap = n_clients
        if verbose:
            print(f"[*] 精确参数: v={v}, k={k}, K={K}, n={n_clients} "
                  f"(n = K(K+1)+1 = {K*(K+1)+1})")
    else:
        p = find_feasible_params(n_clients, K)
        v, k, n_cap = p["v"], p["k"], p["n_capacity"]
        if verbose:
            print(f"[!] 精确解不存在 -> 最小可行参数: v={v}, k={k}, "
                  f"容量={n_cap} >= n={n_clients} (来源: {p['source']})")

    M = None
    if v >= n_clients and k >= 2:
        if verbose:
            print(f"[*] 构造 (v,k,1)=({v},{k},1) 差集 ...")
        D = find_difference_set(v, k, verbose=verbose)
        if D is not None:
            M = build_cyclic_bibd_matrix(v, n_clients, k, D)
            if M.shape[1] < n_clients:
                M = None
        elif verbose:
            print(f"    未找到 ({v},{k},1)-差集，改用通用贪心打包")

    if M is None:
        if verbose:
            print(f"[*] 使用通用贪心打包构造 (v,k) = ({v},{k}) 的部分 BIBD")
        M = build_packing_matrix(v, n_clients, k)
        if M is None:
            raise RuntimeError(f"BIBD 构造失败 (v={v}, k={k}, n={n_clients})")
        D = None

    ok, rep = validate_bibd_matrix(M, k, n_clients)
    if verbose:
        print(f"[*] 关联矩阵验证: 每列恰好 {k} 个 1 = {rep['col_weight_all_k']}, "
              f"最大两列内积 = {rep['max_pairwise_inner_product']} "
              f"(<=1: {rep['pairwise_inter_le_1']}), "
              f"列数 {rep['n_cols']} >= {n_clients}")
    if not ok:
        raise RuntimeError(f"BIBD 关联矩阵未通过验证: {rep}")

    codebook = M[:, :n_clients].copy()
    info = {
        "v": int(codebook.shape[0]),
        "k": int(k),
        "K": int(K),
        "n_clients": int(n_clients),
        "n_capacity": int(n_cap),
        "exact_params": bool(exact),
        "difference_set": list(D) if D is not None else None,
        "matrix_report": rep,
        "construction": "BIBD (cyclic difference set / packing)",
    }
    return codebook, info


# ====================================================================
# 5. AND 唯一性验证（严格执行）
# ====================================================================
def _pack_bits(cb: np.ndarray) -> np.ndarray:
    """把 (v, n) 的 0/1 矩阵按列打包成位向量（每列一个 bytes 块）。"""
    return np.packbits(cb.astype(np.uint8), axis=0)


def _pack_to_words(cb: np.ndarray) -> np.ndarray:
    """
    把 (v, n) 矩阵的每一列打包为 uint64 词数组，返回 shape (n, nwords)。
    AND 运算在 uint64 上按位进行，比逐位布尔运算快很多。
    """
    v, n = cb.shape
    packed = _pack_bits(cb)                       # (ceil(v/8), n)
    nbytes = packed.shape[0]
    pad = (-nbytes) % 8
    if pad:
        packed = np.vstack([packed, np.zeros((pad, n), dtype=np.uint8)])
    # 保证 C 连续后再 view 为 uint64
    flat = np.ascontiguousarray(packed.T)         # (n, nbytes)
    return flat.view(np.uint64)                   # (n, nwords)


def verify_and_uniqueness(codebook: np.ndarray, K: int,
                          max_examples: int = 5, verbose: bool = True):
    """
    严格检查：任意 1..K 个码向量的逐位 AND 是否唯一。
    同一集合重复出现不算冲突；只有"不同集合"产生相同 AND 才算失败。

    返回 (ok, conflicts)；conflicts 为 [(set1, set2), ...] 反例列表。
    """
    v, n = codebook.shape
    if K < 1:
        return True, []

    seen = {}                    # AND 向量(bytes) -> 客户端集合(frozenset)
    conflicts = []
    cb = codebook.astype(np.uint8)
    packed64 = _pack_to_words(cb)                  # (n, nwords)

    for r in range(1, K + 1):
        if r > n:
            break
        for idxs in combinations(range(n), r):
            word = np.bitwise_and.reduce(packed64[list(idxs)], axis=0)
            key = word.tobytes()
            cur = frozenset(idxs)
            prev = seen.get(key)
            if prev is None:
                seen[key] = cur
            elif prev != cur:
                conflicts.append((sorted(prev), sorted(cur)))
                if len(conflicts) >= max_examples:
                    if verbose:
                        print(f"[!] AND 唯一性失败（已发现 {len(conflicts)} 组反例）")
                    return False, conflicts
    if verbose:
        print(f"[*] AND 唯一性 (<= {K} 合谋者): "
              f"{'通过' if not conflicts else '失败'}，"
              f"共检查 {len(seen)} 个 AND 向量")
    return (not conflicts), conflicts


def diagnose_and_uniqueness(codebook: np.ndarray, K: int, verbose: bool = True):
    """
    分析 AND 冲突的成因，给出结构性结论（例如：多个 K 元集合交集同为空）。
    返回 dict。
    """
    v, n = codebook.shape
    cb = codebook.astype(np.uint8)
    info = {"v": int(v), "n": int(n), "K": int(K)}

    empty_collisions = []
    point_collisions = {}
    for r in range(1, K + 1):
        if r > n:
            break
        for idxs in combinations(range(n), r):
            and_vec = np.bitwise_and.reduce(cb[:, idxs], axis=1)
            w = int(and_vec.sum())
            if w == 0:
                empty_collisions.append(idxs)
            elif w == 1:
                p = int(np.argmax(and_vec))
                point_collisions.setdefault(p, []).append(idxs)

    info["num_empty_intersection_sets"] = len(empty_collisions)
    info["num_points_with_multiple_sets"] = sum(
        1 for lst in point_collisions.values() if len(lst) > 1)
    if verbose:
        print(f"    诊断: 交集为空的集合数 = {info['num_empty_intersection_sets']}, "
              f"存在多集合共享同一点的点数 = {info['num_points_with_multiple_sets']}")
        if empty_collisions:
            print(f"    示例(交集全空): {empty_collisions[:3]}")
    return info


# ====================================================================
# 6. 签名行（保证 K-弹性 AND 唯一性的关键部件）
# ====================================================================
def generate_signature_rows(n_clients: int, K: int):
    """
    生成 K 元子集签名行：对每个 K 元集合 Q ∈ C([n], K) 生成一行，
    客户端 i 在该行为 1 当且仅当 i ∈ Q。

    证明（AND 唯一性，对 |S|, |T| <= K）：
      AND(S) 在签名行部分 = { Q ∈ C([n],K) : S ⊆ Q }。
      若 AND(S) = AND(T)，且存在 s ∈ S \ T：
        因 |T| <= K 且 n >= K+1，可取某个 Q ⊇ T 且 s ∉ Q
        （把 T 补成 K 元集合时优先补入 s 以外、且 Q 不含 s 的元素；
          若 T 已含 K 个元素且 T ⊆ ...，则 s ∉ T 且 s ∉ Q 可行）。
        则 Q ∈ AND(T) = AND(S) => S ⊆ Q => s ∈ Q，矛盾。
      故 S ⊆ T，对称得 T ⊆ S，故 S = T。
    返回 (sig_rows, row_index_dict)。
    """
    if n_clients < K + 1:
        raise ValueError(
            f"签名行构造要求 n_clients >= K+1 (当前 n={n_clients}, K={K})")

    rows = list(combinations(range(n_clients), K))
    index = {row: i for i, row in enumerate(rows)}
    sig = np.zeros((len(rows), n_clients), dtype=np.int8)
    for r, Q in enumerate(rows):
        for i in Q:
            sig[r, i] = 1
    return sig, index


# ====================================================================
# 7. 纯精确构造（K 元子集地面集）
# ====================================================================
def generate_exact_codebook(n_clients: int, K: int, verbose: bool = True,
                             max_rows: int = 5_000_000):
    """
    精确构造：地面集 = 客户端的所有 K 元子集（行），客户端 i 的码向量 =
    所有"包含 i 的 K 元子集"所在行为 1。

    AND 唯一性证明见 generate_signature_rows。
    返回 (codebook, info)。
    """
    if K < 1:
        raise ValueError("精确构造要求 K >= 1")
    if n_clients < K + 1:
        raise ValueError(f"精确构造要求 n_clients >= K+1 (当前 n={n_clients}, K={K})")

    n_rows = comb(n_clients, K)
    if n_rows > max_rows:
        raise ValueError(f"精确构造地面集过大 (C(n,K)={n_rows} > {max_rows})")

    codebook, _ = generate_signature_rows(n_clients, K)

    info = {
        "v": int(n_rows),
        "k": int(K),
        "K": int(K),
        "n_clients": int(n_clients),
        "n_capacity": int(n_clients),
        "exact_params": False,
        "construction": "exact K-subset ground set",
        "ground_set_size": int(n_rows),
    }
    if verbose:
        print(f"[*] 精确构造: 地面集 = C({n_clients},{K}) = {n_rows} 行, "
              f"每列 {comb(n_clients - 1, K - 1)} 个 1")
    return codebook, info


# ====================================================================
# 8. 混合构造：BIBD 骨架 + 签名行
# ====================================================================
def generate_hybrid_codebook(n_clients: int, K: int, verbose: bool = True):
    """
    混合构造 = BIBD 关联矩阵（骨架：每列恰 k 个 1、两两内积 <= 1）
              ++ K 元子集签名行（保证 K-弹性 AND 唯一性）。

    BIBD 骨架提供经典的"块结构"，签名行提供高阶分辨力；
    两者按位 AND 拼接，签名部分单独就足以唯一确定合谋集合。
    返回 (codebook, info)。
    """
    if n_clients < K + 1:
        raise ValueError(f"混合构造要求 n_clients >= K+1 (当前 n={n_clients}, K={K})")

    try:
        bibd_cb, bibd_info = generate_bibd_codebook(n_clients, K, verbose=verbose)
    except Exception as exc:                        # noqa: BLE001
        if verbose:
            print(f"[!] BIBD 骨架不可用 ({exc})，退化为纯精确构造")
        return generate_exact_codebook(n_clients, K, verbose=verbose)

    sig, _ = generate_signature_rows(n_clients, K)
    codebook = np.vstack([bibd_cb, sig])

    info = dict(bibd_info)
    info.update({
        "v": int(codebook.shape[0]),
        "v_bibd": int(bibd_cb.shape[0]),
        "v_signature": int(sig.shape[0]),
        "construction": "hybrid (BIBD skeleton + K-subset signature rows)",
    })
    if verbose:
        print(f"[*] 混合构造: BIBD 骨架 {bibd_cb.shape[0]} 行 + "
              f"签名行 {sig.shape[0]} 行 = {codebook.shape[0]} 行")
    return codebook, info


# ====================================================================
# 6c. 随机大码本模式（行数远小于 C(n,K)，高概率满足 AND 唯一性）
# ====================================================================
def generate_random_codebook(n_clients: int, K: int, n_rows: int = None,
                            density: float = 0.5, seed: int = 0,
                            verbose: bool = True,
                            max_attempts: int = 50):
    """
    随机大码本：每行是一个随机 0/1 向量（每个位置以 density 概率为 1）。

    AND 唯一性的概率论论证：
      对固定的两个不同集合 S != T，取 s in S\\T（对称情形同��）。
      AND(S) 在第 s 位为 1 要求 s 的码向量该位为 1 且 S 中所有客户端该位为 1；
      AND(T) 在第 s 位为 0 要求 T 中至少一个客户端该位为 0。
      两者的"差异事件"相互独立，冲突概率约为
          p_conflict ~ 2^(-|S \\ T|) 的量级，实际对大码本极小。
      因此 n_rows 足够大时，AND 唯一性以压倒性概率成立。

    行数选择：默认取一个经验值，保证 (a) 行数远小于 C(n,K)，
    (b) 冲突概率可忽略。实际生成后会严格验证，若失败会自动加大行数重试。

    返回 (codebook, info)。
    """
    rng = np.random.default_rng(seed)

    if n_rows is None:
        # 经验行数：max(8n, 32*C(n,K) 的对数级) —— 取 C(n,K)^0.5 量级即可
        n_rows = max(8 * n_clients, int(np.ceil(4 * np.sqrt(comb(n_clients, K)))))

    cb = None
    for attempt in range(max_attempts):
        cand = (rng.random((n_rows, n_clients)) < density).astype(np.int8)
        # 避免全 0 列（全 0 列会让 AND 恒为 0，失去分辨力）
        for j in range(n_clients):
            if cand[:, j].sum() == 0:
                cand[rng.integers(n_rows), j] = 1
        ok, _ = verify_and_uniqueness(cand, K, verbose=False)
        if ok:
            cb = cand
            break
        # 冲突则加行重试
        n_rows = int(n_rows * 1.5) + 8
        if verbose:
            print(f"    随机码本出现 AND 冲突，扩容至 {n_rows} 行重试 "
                  f"(第 {attempt + 1} 次)")

    if cb is None:
        return None, None

    info = {
        "v": int(cb.shape[0]),
        "k": int(K),
        "K": int(K),
        "n_clients": int(n_clients),
        "n_capacity": int(n_clients),
        "exact_params": False,
        "construction": f"random large codebook (density={density})",
        "density": float(density),
        "seed": int(seed),
    }
    if verbose:
        print(f"[*] 随机大码本: {cb.shape[0]} 行 x {cb.shape[1]} 列, "
              f"密度={density}, AND 唯一性通过")
    return cb, info


# ====================================================================
# 9. 浮点映射
# ====================================================================
def map_to_float(codebook: np.ndarray, mu=0.0, delta=0.01):
    """
    0/1 -> 两级浮点：1 -> mu + delta，0 -> mu - delta。
    mu/delta 可为标量或长度 v 的数组。
    """
    v = codebook.shape[0]
    mu_arr = np.full(v, float(mu)) if np.isscalar(mu) else np.asarray(mu, dtype=float).ravel()
    delta_arr = np.full(v, float(delta)) if np.isscalar(delta) else np.asarray(delta, dtype=float).ravel()
    if mu_arr.size != v or delta_arr.size != v:
        raise ValueError("mu/delta 长度必须等于码本行数 v")
    if np.any(delta_arr <= 0):
        raise ValueError("delta 必须为正")

    float_cb = np.where(codebook == 1,
                        mu_arr[:, None] + delta_arr[:, None],
                        mu_arr[:, None] - delta_arr[:, None])
    return float_cb, mu_arr, delta_arr


# ====================================================================
# 9b. 推荐配置（结合水印预算）
# ====================================================================
#: LightAutoencoder 编码器参数量（本项目水印总量的来源，见 train_key_matrix.py）
DEFAULT_ENCODER_PARAMS = 9472


def recommend_config(n_clients: int, K: int = 3,
                     encoder_params: int = DEFAULT_ENCODER_PARAMS):
    """
    根据客户端数与水印预算给出推荐的码本配置。

    水印预算来源（train_key_matrix.py）：
      total_watermark_size = encoder_params
      每个客户端分到 watermark_size = encoder_params // n_clients 个参数位置，
      这些位置就是该客户端能承载的浮点水印个数，也就是码本的"行数"上限。

    返回 dict：{n_clients, K, max_rows, mode, v_expected, note}
    """
    max_rows = encoder_params // n_clients
    need_sig = comb(n_clients, K) if K >= 1 else 0
    # BIBD 骨架行数（v = k(k-1)+1，k = K+1）
    k = K + 1
    v_bibd = k * (k - 1) + 1

    if need_sig + v_bibd <= max_rows:
        mode = "hybrid"
        v_expected = v_bibd + need_sig
        note = "严格保证 AND 唯一性（BIBD 骨架 + K 元子集签名行）"
    else:
        mode = "random"
        v_expected = max_rows
        note = (f"预算不足（C(n,K)={need_sig} 行放不下），"
                f"改用随机大码本，AND 唯一性靠概率保证")

    return {
        "n_clients": int(n_clients),
        "K": int(K),
        "max_rows": int(max_rows),
        "mode": mode,
        "v_expected": int(v_expected),
        "note": note,
    }


# ====================================================================
# 10. 主入口
# ====================================================================
def generate_codebook(n_clients: int, K: int, mode: str = "auto",
                      mu=0.0, delta=0.01, verbose: bool = True,
                      max_rows: int = None, density: float = 0.5,
                      seed: int = 0):
    """
    生成浮点水印码本。

    mode:
        'bibd'   —— 只用 BIBD 构造（保留经典结构；AND 唯一性仅 K=1 成立，
                    K>=2 时照实报错并给出反例）；
        'hybrid' —— BIBD 骨架 + K 元子集签名行（严格保证 AND 唯一性，
                    行数 = v_bibd + C(n,K)，在 n=20,K=3 时达 1160 行）；
        'exact'  —— 纯 K 元子集地面集（行数 = C(n,K)，理论下界）；
        'random' —— 随机大码本（行数远小于 C(n,K)，AND 唯一性靠概率保证，
                    适合水印预算受限的场景）；
        'auto'   —— 按预算自动选择：预算够 -> hybrid；不够 -> random。

    max_rows: 码本行数上限（一般设为每客户端可用的水印位置数）。
              给定后会挑选行数最小的可行模式。
    返回 (float_codebook, info)；info 含 binary_codebook / mu / delta 等。
    """
    if mode not in ("auto", "bibd", "exact", "hybrid", "random"):
        raise ValueError("mode 必须是 'auto' | 'bibd' | 'exact' | 'hybrid' | 'random'")

    codebook, info, used = None, None, None

    # ---------- 预算感知：auto 模式按 max_rows 选模式 ----------
    if mode == "auto" and max_rows is not None:
        need_exact = comb(n_clients, K) if K >= 1 else 0
        if need_exact <= max_rows:
            mode = "hybrid"
            if verbose:
                print(f"[*] 预算 {max_rows} 行足够容纳 hybrid "
                      f"(需 {need_exact} 行签名)")
        else:
            mode = "random"
            if verbose:
                print(f"[*] 预算 {max_rows} 行不足 hybrid (需 {need_exact} 行), "
                      f"改用 random 模式")

    if mode in ("auto", "hybrid", "bibd"):
        try:
            if mode == "bibd":
                codebook, info = generate_bibd_codebook(n_clients, K, verbose=verbose)
                used = "bibd"
            else:
                codebook, info = generate_hybrid_codebook(n_clients, K, verbose=verbose)
                used = "hybrid"
        except Exception as exc:                     # noqa: BLE001
            if mode == "bibd":
                raise
            if verbose:
                print(f"[!] {mode} 构造不可用 ({exc})，回退到精确构造")
            codebook, info = None, None

        if codebook is not None:
            if max_rows is not None and codebook.shape[0] > max_rows:
                if verbose:
                    print(f"[!] {used} 码本 {codebook.shape[0]} 行 "
                          f"超出预算 {max_rows} 行")
                if mode == "bibd":
                    raise RuntimeError(
                        f"BIBD 码本 {codebook.shape[0]} 行超出预算 {max_rows} 行")
                codebook, info = None, None
            else:
                ok, conflicts = verify_and_uniqueness(codebook, K, verbose=verbose)
                info["and_unique"] = bool(ok)
                info["and_conflicts"] = conflicts
                if not ok:
                    if verbose:
                        print("[!] 码本不满足 K-弹性 AND 唯一性，反例：")
                        for a, b in conflicts[:3]:
                            print(f"      AND({a}) == AND({b})")
                        diagnose_and_uniqueness(codebook, K, verbose=verbose)
                    if mode == "bibd":
                        raise RuntimeError(
                            f"BIBD 码本不满足 K={K} 的 AND 唯一性。这是 (v,k,1)-BIBD "
                            f"的固有性质：同一点处的多条线两两 AND 相同，"
                            f"故仅 K=1 严格成立。请使用 mode='hybrid' 或 'random'。"
                        )
                    codebook, info = None, None

    if codebook is None and mode != "random":
        if mode == "bibd":
            raise RuntimeError("BIBD 路线失败")
        codebook, info = generate_exact_codebook(n_clients, K, verbose=verbose)
        used = "exact"
        ok, conflicts = verify_and_uniqueness(codebook, K, verbose=verbose)
        info["and_unique"] = bool(ok)
        info["and_conflicts"] = conflicts
        if not ok:
            raise RuntimeError("精确构造竟然未通过 AND 唯一性验证（内部错误）")
        if max_rows is not None and codebook.shape[0] > max_rows:
            # exact 也超预算 -> 退到 random
            if verbose:
                print(f"[!] exact 码本 {codebook.shape[0]} 行超出预算 "
                      f"{max_rows} 行，改用 random 模式")
            codebook, info = None, None

    if codebook is None:
        # random 模式：行数受预算约束
        n_rows = max_rows if max_rows is not None else None
        cb, ri = generate_random_codebook(n_clients, K, n_rows=n_rows,
                                          density=density, seed=seed,
                                          verbose=verbose)
        if cb is None:
            raise RuntimeError(
                f"random 模式在预算 {max_rows} 行内无法满足 AND 唯一性")
        codebook, info = cb, ri
        used = "random"
        ok, conflicts = verify_and_uniqueness(codebook, K, verbose=verbose)
        info["and_unique"] = bool(ok)
        info["and_conflicts"] = conflicts

    info["construction_used"] = used
    float_cb, mu_arr, delta_arr = map_to_float(codebook, mu, delta)
    info["binary_codebook"] = codebook
    info["mu"] = mu_arr
    info["delta"] = delta_arr
    info["float_codebook"] = float_cb
    return float_cb, info


# ====================================================================
# 11. 合谋模拟 + 归因
# ====================================================================
def simulate_collusion(float_cb: np.ndarray, info: dict, colluders,
                       threshold_ratio: float = 0.5, verbose: bool = True):
    """
    模拟等权平均合谋，并从浮点合谋模型恢复合谋者集合。
    判决：mean_j > mu_j + ratio*delta_j 判 1，否则 0；
    再在码本中搜索大小 <= K 的组合做 AND 匹配。
    """
    mu, delta = info["mu"], info["delta"]
    cb = info["binary_codebook"].astype(np.uint8)
    K = info["K"]
    n_clients = cb.shape[1]
    colluders = list(colluders)

    # --- 入参校验：给出明确错误信息，避免裸 IndexError ---
    if not colluders:
        raise ValueError("colluders 不能为空")
    if len(set(colluders)) != len(colluders):
        raise ValueError(f"colluders 含重复客户端: {colluders}")
    out_of_range = [i for i in colluders if not (0 <= i < n_clients)]
    if out_of_range:
        raise IndexError(
            f"客户端索引越界: {out_of_range}，合法范围为 [0, {n_clients - 1}]"
            f"（当前码本 n_clients={n_clients}）")
    if len(colluders) > K:
        raise ValueError(
            f"合谋者数量 {len(colluders)} 超过码本可归因上限 K={K}")

    colluded = np.mean(float_cb[:, colluders], axis=1)
    thresh = mu + threshold_ratio * delta
    obs_bits = (colluded > thresh).astype(np.uint8)

    if verbose:
        print(f"\n[合谋] 客户端 {colluders}")
        print(f"  合谋均值(前5块): {np.round(colluded[:5], 4)}")
        print(f"  判决码位(前40): {obs_bits[:40].tolist()}")

    candidates = []
    for r in range(1, min(K, cb.shape[1]) + 1):
        for idxs in combinations(range(cb.shape[1]), r):
            and_vec = np.bitwise_and.reduce(cb[:, idxs], axis=1)
            if np.array_equal(and_vec, obs_bits):
                candidates.append(idxs)

    success = (len(candidates) == 1 and set(candidates[0]) == set(colluders))
    if verbose:
        print(f"  候选: {candidates}")
        print(f"  归因: {'成功' if success else '失败/不唯一'}")
    return success, candidates


# ====================================================================
# 12. 保存 / 加载
# ====================================================================
def save_codebook(path: str, float_cb: np.ndarray, info: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = {k: v for k, v in info.items()
            if k not in ("binary_codebook", "mu", "delta", "float_codebook")}
    np.savez_compressed(
        path,
        float_codebook=float_cb.astype(np.float32),
        binary_codebook=info["binary_codebook"].astype(np.uint8),
        mu=info["mu"].astype(np.float32),
        delta=info["delta"].astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False, default=str),
    )
    return path


# ====================================================================
# 13. 命令行
# ====================================================================
def main():
    ap = argparse.ArgumentParser(description="BIBD-ACC 浮点水印码本生成")
    ap.add_argument("--n_clients", type=int, default=13)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--mode", type=str, default="auto",
                    choices=["auto", "bibd", "exact", "hybrid", "random"])
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=None,
                    help="码本行数上限（一般=每客户端可用水印位置数）；"
                         "给定后 auto 模式会自动挑选行数最小的可行方案")
    ap.add_argument("--density", type=float, default=0.5,
                    help="random 模式的 1 的密度")
    ap.add_argument("--recommend", action="store_true",
                    help="只打印 n=5/10/20 的推荐配置表后退出")
    ap.add_argument("--encoder_params", type=int, default=DEFAULT_ENCODER_PARAMS,
                    help="水印总量（编码器参数量），用于计算每客户端的行数预算")
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.recommend:
        print(f"{'n':>4} {'K':>3} {'行数预算':>9} {'C(n,K)':>9} "
              f"{'推荐模式':>9} {'预期行数':>9}  说明")
        print("-" * 88)
        for n in (5, 10, 20):
            for K in (1, 2, 3):
                rec = recommend_config(n, K=K,
                                       encoder_params=args.encoder_params)
                print(f"{n:>4} {K:>3} {rec['max_rows']:>9} "
                      f"{comb(n, K):>9} {rec['mode']:>9} "
                      f"{rec['v_expected']:>9}  {rec['note']}")
        return

    # 未显式给出行数预算时，按水印总量自动推导
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = args.encoder_params // args.n_clients
        if args.mode == "auto":
            print(f"[*] 自动推导行数预算: {args.encoder_params} / "
                  f"{args.n_clients} = {max_rows}")

    float_cb, info = generate_codebook(
        n_clients=args.n_clients, K=args.K, mode=args.mode,
        mu=args.mu, delta=args.delta, verbose=True,
        max_rows=max_rows, density=args.density, seed=args.seed)

    print(f"\n[结果]")
    print(f"  构造方式   : {info['construction_used']}")
    print(f"  v (块数)   : {info['v']}")
    print(f"  k          : {info['k']}")
    print(f"  码本 shape : {float_cb.shape}")
    print(f"  AND 唯一性 : {'通过' if info.get('and_unique') else '失败'}")
    print("  前 4 行 (每行一个参数块):")
    print(np.round(float_cb[:4], 4))

    if args.out:
        save_codebook(args.out, float_cb, info)
        print(f"\n[*] 已保存: {args.out}")

    # 合谋归因演示
    if args.n_clients >= 2:
        simulate_collusion(float_cb, info, colluders=(0, 1))


if __name__ == "__main__":
    main()
