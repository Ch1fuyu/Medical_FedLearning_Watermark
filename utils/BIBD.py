"""
浮点水印抗合谋码本生成脚本（单一模式）

> 说明：本文件是历史版本，核心实现已迁移到
>       utils/bibd_acc_codebook.py（Singer 差集 + 严格 AND 唯一性验证）。
>       为保持向后兼容，这里转发到新实现。

输入:
    n_clients : 客户端总数
    K         : 要抵抗的最大合谋者数
    mu        : 每块基准值（标量或长度 v 的数组）
    delta     : 每块嵌入强度（标量或长度 v 的数组）

输出:
    float_codebook : ndarray (v, n_clients)
        每列一个客户端的浮点水印向量；
        第 j 行对应第 j 个参数块，值为 mu_j ± delta_j。

码本逻辑:
    按 (v, k, 1)-BIBD 构造 0/1 码本 (k = K+1)，
    再把 0/1 映射为两级浮点值: 1 -> mu+delta, 0 -> mu-delta。
    任意 <= K 个客户端的逐位 AND 唯一，因此合谋平均后
    "共同为 1 的块保持高值" 可唯一反推合谋者集合。

    注：纯 (v,k,1)-BIBD 的 AND 唯一性仅在 K=1 时严格成立（投影平面上
    同一点处多条线的 AND 相同）。K>=2 时默认使用 hybrid 构造
    （BIBD 骨架 + K 元子集签名行），严格满足 K-弹性 AND 唯一性。
"""

import os
import sys

import numpy as np

# 允许 "python utils/BIBD.py" 直接运行（项目根不在 sys.path 时回退）
try:
    from utils.bibd_acc_codebook import (  # noqa: F401
        generate_codebook,
        map_to_float,
        simulate_collusion as simulate_collusion_float,
        verify_and_uniqueness,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.bibd_acc_codebook import (  # noqa: F401
        generate_codebook,
        map_to_float,
        simulate_collusion as simulate_collusion_float,
        verify_and_uniqueness,
    )


def generate_float_codebook(n_clients, K, mu=0.0, delta=0.01, verbose=True,
                            mode="auto"):
    """
    生成浮点水印码本（转发到 utils.bibd_acc_codebook.generate_codebook）。

    参数:
        n_clients : 客户端总数
        K         : 最大合谋者数
        mu        : 每块基准值；标量或长度 v 的数组
        delta     : 每块嵌入强度；标量或长度 v 的数组
        mode      : 'auto' | 'hybrid' | 'exact' | 'bibd'

    返回:
        float_cb : ndarray (v, n_clients)，每列一个客户端的浮点水印
        info     : dict，含 v, k, n_clients, K, n_capacity, binary_codebook
    """
    return generate_codebook(n_clients=n_clients, K=K, mode=mode,
                             mu=mu, delta=delta, verbose=verbose)


# ============================================================
# 1. 参数求解
# ============================================================
def solve_bibd_params(n_clients, K):
    """解 n = (v^2 - v) / (k^2 - k), k = K+1。返回 (v, k) 或 None。"""
    from utils.bibd_acc_codebook import solve_bibd_params as _f
    return _f(n_clients, K)


def find_feasible_params(n_clients, K, max_search=500):
    """精确解不存在时，搜索最小可行参数。"""
    from utils.bibd_acc_codebook import find_feasible_params as _f
    p = _f(n_clients, K, max_search)
    return p["v"], p["k"], p["n_capacity"]


# ============================================================
# 2. BIBD 构造
# ============================================================
def construct_bibd_projective_plane(q):
    """构造 (q^2+q+1, q+1, 1)-BIBD 的关联矩阵（循环差集法）。"""
    from utils.bibd_acc_codebook import (singer_difference_set,
                                         build_cyclic_bibd_matrix)
    res = singer_difference_set(q)
    if res is None:
        return None
    v, k, D = res
    return build_cyclic_bibd_matrix(v, v, k, D)


def construct_bibd_greedy(v, k, n_blocks):
    """通用贪心构造 (v, k, 1)-packing：任两块交 <= 1。"""
    from utils.bibd_acc_codebook import build_packing_matrix
    return build_packing_matrix(v, n_blocks, k)


# ============================================================
# 3. 验证
# ============================================================
def verify_and_uniqueness_legacy(binary_cb, K):
    """历史接口：检查任意 1..K 个码向量的逐位 AND 是否唯一。"""
    return verify_and_uniqueness(binary_cb, K, verbose=False)


# ============================================================
# 4. 示例
# ============================================================
if __name__ == "__main__":
    n_clients, K = 10, 3
    float_cb, info = generate_float_codebook(
        n_clients=n_clients, K=K, mu=0.0, delta=0.01, verbose=True)

    print(f"\n浮点码本 shape = {float_cb.shape}")
    print("前 4 行 (每行一个参数块):")
    print(np.round(float_cb[:4], 4))

    simulate_collusion_float(float_cb, info, colluders=(0, 1))
    simulate_collusion_float(float_cb, info, colluders=(0, n_clients - 1))
    simulate_collusion_float(float_cb, info, colluders=(0, 1, n_clients - 1))