"""
水印区域冻结工具

用途
----
在「完全冻结水印区域」的实验模式下，每个客户端在本地训练时，
其**自有水印区域 P_k** 完全不参与参数更新：

1. 反向传播后把 P_k 位置的梯度置零（避免动量 / Adam 二阶矩累积带来的漂移）；
2. 优化器 step 之后把 P_k 的参数值还原为训练开始时的快照
   （抵消 weight decay、DP 噪声等一切非梯度来源的改动）。

由于 P_k 在整个本地训练期间保持恒定，服务器下发的权威水印值
会被原封不动地带回，客户端之间也天然保持可区分。

见证点区域（抗合谋水印）
-----------------------
抗合谋分支的见证点区域是**跨客户端共享**的一批位置，其取值由
服务器密钥派生并且**在所有客户端、所有轮次上固定**（标记点写权威值
e_p，未标记点写基线 b_p）。因此见证点区域对所有客户端都应被冻结：

- 冻结后，合谋平均的结果可以直接反解"该点被多少个合谋者标记"；
- 不冻结的话，任务梯度 + DP 噪声会把见证点冲掉，信号在若干轮后消失。

索引说明
--------
`KeyMatrixManager.load_positions(client_id)` 返回的 `(param_name, local_idx)`
是 **参数内部局部索引**（由 train_key_matrix.py 生成），因此可以直接
按参数名定位到 nn.Parameter 的扁平视图，无需再做全局偏移换算。

见证点位置来自 `witness_layout.json` 的 `positions` 字段，同样是
`(param_name, local_idx)` 形式。
"""

import os
from typing import Dict, Optional, Tuple

import torch

from utils.key_matrix_utils import KeyMatrixManager


class WatermarkFreezer:
    """按客户端冻结其自有水印区域 P_k（可选同时冻结见证点区域）。"""

    def __init__(self, model, key_matrix_path, args=None, layout=None):
        self.model = model
        self.args = args
        self.key_manager = KeyMatrixManager(key_matrix_path, args=args)
        self.client_num = getattr(self.key_manager, 'client_num', 0)
        # {client_id: {param_name: LongTensor(indices)}}
        self._index_cache = {}

        # ---- 见证点布局（可选）----
        # 分支隔离：见证点只属于抗合谋分支（key_matrix_profile == 'collusion'）。
        # 默认分支（传统水印基线）即便目录下存在 witness_layout.json，
        # 也不冻结见证点区域——保持与 main.py 的嵌入逻辑一致。
        self.layout = layout
        self._witness_index_map: Optional[Dict[str, torch.Tensor]] = None
        if self.layout is None:
            self.layout = self._load_layout(key_matrix_path)
        if self.layout is not None:
            self._witness_index_map = self._build_witness_indices()

    # ------------------------------------------------------------------
    # 布局加载
    # ------------------------------------------------------------------
    def _witness_allowed(self) -> bool:
        """当前分支是否允许使用见证点。

        默认分支（传统水印基线）不使用见证点；抗合谋分支才启用。
        若调用方未传 args、或 args 未设置 key_matrix_profile，
        则保守地允许（便于测试与脚本直接构造）。
        """
        if self.args is None:
            return True
        profile = getattr(self.args, 'key_matrix_profile', None)
        if profile is None:
            return True
        return profile == 'collusion'

    def _load_layout(self, key_matrix_path):
        """在 key_matrix 目录下查找 witness_layout.json（找不到返回 None）。"""
        if not self._witness_allowed():
            return None
        try:
            from utils.collusion_resistant import find_witness_layout
            path = find_witness_layout(key_matrix_path)
            if not path:
                return None
            from utils.collusion_resistant import load_witness_layout
            return load_witness_layout(path)
        except Exception:
            return None

    def _build_witness_indices(self) -> Dict[str, torch.Tensor]:
        """见证点池的**全部**位置索引（所有客户端共享，因此与 client_id 无关）。

        见证点区域在所有客户端上取值固定（标记点 e_p / 未标记点 b_p），
        任何一个客户端都不应改动它，因此统一冻结整片池子。
        """
        index_map: Dict[str, list] = {}
        for param_name, local_idx in self.layout.positions:
            index_map.setdefault(param_name, []).append(int(local_idx))
        return {
            name: torch.tensor(sorted(set(idxs)), dtype=torch.long)
            for name, idxs in index_map.items()
        }

    @property
    def has_witness(self) -> bool:
        """是否加载到了见证点布局。"""
        return self.layout is not None and bool(self._witness_index_map)

    # ------------------------------------------------------------------
    # 索引准备
    # ------------------------------------------------------------------
    def client_indices(self, client_id):
        """返回该客户端水印区域索引表：{param_name: LongTensor(局部索引)}"""
        if client_id in self._index_cache:
            return self._index_cache[client_id]

        index_map = {}
        for param_name, local_idx in self.key_manager.load_positions(client_id):
            index_map.setdefault(param_name, []).append(int(local_idx))

        index_map = {
            name: torch.tensor(sorted(set(idxs)), dtype=torch.long)
            for name, idxs in index_map.items()
        }
        self._index_cache[client_id] = index_map
        return index_map

    def witness_indices(self) -> Dict[str, torch.Tensor]:
        """返回见证点区域的索引表（所有客户端相同）。"""
        if self._witness_index_map is None:
            return {}
        return self._witness_index_map

    def _named_params(self):
        return dict(self.model.named_parameters())

    # ------------------------------------------------------------------
    # 冻结相关操作
    # ------------------------------------------------------------------
    def snapshot(self, client_id, include_witness=True):
        """
        记录当前水印区域参数值（作为「训练后还原」的基准）。

        Args:
            client_id: 客户端 ID。
            include_witness: 是否同时快照见证点区域。

        Returns:
            {param_name: (values_clone, indices)}
        """
        tables = [self.client_indices(client_id)]
        if include_witness and self.has_witness:
            tables.append(self.witness_indices())

        params = self._named_params()
        snapshot = {}
        for index_map in tables:
            for name, idx in index_map.items():
                param = params.get(name)
                if param is None:
                    continue
                flat = param.detach().view(-1)
                valid = idx[idx < flat.numel()]
                if valid.numel() == 0:
                    continue
                if name in snapshot:
                    # 与已有快照取并集
                    prev_vals, prev_idx = snapshot[name]
                    merged = torch.unique(torch.cat([prev_idx.to(valid.device), valid]))
                    vals = flat[merged].clone()
                    snapshot[name] = (vals, merged)
                else:
                    snapshot[name] = (flat[valid].clone(), valid.to(flat.device))
        return snapshot

    def zero_grads(self, client_id, include_witness=True):
        """把水印区域位置的梯度置零（必须在 backward 之后、optimizer.step 之前调用）"""
        tables = [self.client_indices(client_id)]
        if include_witness and self.has_witness:
            tables.append(self.witness_indices())

        params = self._named_params()
        for index_map in tables:
            for name, idx in index_map.items():
                param = params.get(name)
                if param is None or param.grad is None:
                    continue
                flat_grad = param.grad.view(-1)
                valid = idx[idx < flat_grad.numel()]
                if valid.numel() == 0:
                    continue
                flat_grad[valid] = 0.0

    def restore(self, snapshot):
        """把水印区域参数还原为快照值（必须在 optimizer.step / DP 噪声之后调用）"""
        if not snapshot:
            return
        params = self._named_params()
        with torch.no_grad():
            for name, (values, idx) in snapshot.items():
                param = params.get(name)
                if param is None:
                    continue
                flat = param.detach().view(-1)
                valid = idx[idx < flat.numel()]
                if valid.numel() == 0:
                    continue
                flat[valid] = values.to(flat.device)

    def witness_positions(self):
        """返回见证点区域的模型位置列表 [(param_name, local_idx), ...]。"""
        if self.layout is None:
            return []
        return list(self.layout.positions)
