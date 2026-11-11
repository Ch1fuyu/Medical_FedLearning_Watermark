"""抗合谋评测配置表（合谋规模 K -> 密钥矩阵规格）的解析与校验。

背景
----
抗合谋归因的搜索空间上界原本是

    k_search_cap = layout.design.K + 2

其中 ``layout.design.K`` 来自 ``witness_layout.json``（生成密钥矩阵时
由 ``train_key_matrix.py --collusion_K`` 决定）。如果实际仿真的合谋规模
``collusion_size`` 超出 ``design.K + 2``，真解不在搜索空间内，
归因会**静默失败**（输出空列表），很容易被误读成"方案缺陷"。

本模块把"合谋规模 K -> 密钥矩阵规格（client_num / model_name）"做成
**显式配置表**（默认 ``config/collusion_presets.json``），使得：

- ``main.py`` 在抗合谋模式下按 ``--collusion_size`` 自动定位密钥矩阵目录，
  不需要用户手工对齐 ``--client_num`` / ``--model_name``；
- 配置缺失或不合规时**直接退出并说明原因**，而不是跑出一堆错误结果。

关于 design.K
-------------
见证点设计的实质参数（地面集 v、每点标记数 k、副本数 G、关联矩阵 A）
**只取决于客户端数 n_clients**，与 ``design.K`` 无关；``design.K`` 只是
一个"设计容量"标注。因此：

- 配置表的条目只需声明 ``client_num`` / ``model_name``；
- ``design.K`` 以磁盘上的布局文件为准（权威来源）；
- 运行时搜索上界取 ``max(design.K + 余量, 实际合谋规模)``，
  即"设计容量不足时自动放宽 + 明确告警"，而不是静默失败。

威胁模型
--------
本模块只做"配置解析 + 目录定位 + 一致性校验"，不涉及任何水印逻辑。
"""

import json
import os
from typing import Dict, Optional, Tuple

#: 默认配置表路径（相对项目根目录）。
DEFAULT_PRESET_PATH = os.path.join('config', 'collusion_presets.json')

#: 归因搜索上限相对 design.K 的余量：k_search_cap = design.K + margin。
#: main.py 沿用同样的余量，保证"配置表声明的 design_K"与"运行时真正
#: 搜索的范围"一致。
SEARCH_MARGIN = 2


class CollusionPresetError(RuntimeError):
    """抗合谋配置表解析 / 校验失败。

    继承 ``RuntimeError``，便于上层统一捕获并给出可读的错误信息。
    """


class CollusionSpec:
    """某个合谋规模 K 对应的密钥矩阵规格。

    Attributes:
        size: 合谋规模 K。
        client_num: 客户端数。
        model_name: 模型类型（'resnet' / 'alexnet'）。
        design_K: 布局的设计容量（**从磁盘布局读取**，配置表可覆盖）。
        source: 配置来源描述（用于日志）。
    """

    __slots__ = ('size', 'client_num', 'model_name', 'design_K', 'source')

    def __init__(self, size: int, client_num: int, model_name: str,
                 design_K: Optional[int] = None, source: str = ""):
        self.size = int(size)
        self.client_num = int(client_num)
        self.model_name = str(model_name)
        self.design_K = int(design_K) if design_K is not None else None
        self.source = source

    def copy(self) -> "CollusionSpec":
        """返回一份拷贝。

        ``validate_layout`` 会把磁盘上的 design.K 写回规格，
        因此调用方不应直接改动配置表里的共享对象。
        """
        return CollusionSpec(self.size, self.client_num, self.model_name,
                             self.design_K, self.source)

    @property
    def search_cap(self) -> Optional[int]:
        """归因搜索的合谋规模上界（= design.K + margin）。

        ``design_K`` 未知（尚未读取布局）时返回 None。
        """
        if self.design_K is None:
            return None
        return self.design_K + SEARCH_MARGIN

    def covers_size(self) -> bool:
        """该规格的搜索范围是否覆盖它所声明的合谋规模 K。

        ``design_K`` 未知时返回 True（留待布局校验阶段判断）。
        """
        cap = self.search_cap
        if cap is None:
            return True
        return self.size <= cap

    def key_matrix_dirname(self) -> str:
        """密钥矩阵子目录名（相对 key_matrix_dir）。"""
        return os.path.join(self.model_name, f'client{self.client_num}')

    def __repr__(self) -> str:
        return (f"CollusionSpec(size={self.size}, client_num={self.client_num}, "
                f"model_name={self.model_name!r}, design_K={self.design_K})")


def _load_json_source(raw: str, source_label: str) -> dict:
    """从 JSON 字符串或文件路径加载配置。"""
    text = raw.strip()
    if text.startswith('{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise CollusionPresetError(
                f"抗合谋配置表 JSON 解析失败 ({source_label}): {e}") from e

    if not os.path.exists(text):
        raise CollusionPresetError(
            f"抗合谋配置表文件不存在: {text} ({source_label})")
    try:
        with open(text, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CollusionPresetError(
            f"抗合谋配置表 JSON 解析失败 ({text}): {e}") from e


def load_presets(raw: Optional[str] = None,
                 default_path: str = DEFAULT_PRESET_PATH) -> Dict[int, CollusionSpec]:
    """加载并校验抗合谋配置表。

    Args:
        raw: JSON 字符串或文件路径；None 时使用 ``default_path``。
        default_path: 默认配置表路径。

    Returns:
        {合谋规模 K: CollusionSpec}

    Raises:
        CollusionPresetError: 配置不存在、格式错误或内容非法。
    """
    if raw:
        data = _load_json_source(raw, "命令行")
        source = "命令行"
    else:
        if not os.path.exists(default_path):
            raise CollusionPresetError(
                f"抗合谋配置表不存在: {default_path}\n"
                f"请创建该文件, 或用 --collusion_size_configs 指定其他配置表。")
        data = _load_json_source(default_path, default_path)
        source = default_path

    if not isinstance(data, dict):
        raise CollusionPresetError(
            f"抗合谋配置表必须是 JSON 对象 ({source}), "
            f"实际类型 {type(data).__name__}")

    specs: Dict[int, CollusionSpec] = {}
    for key, cfg in data.items():
        if key.startswith('_'):
            continue  # 注释字段
        try:
            size = int(key)
        except (TypeError, ValueError):
            raise CollusionPresetError(
                f"抗合谋配置表的键必须是整数 (合谋规模 K), "
                f"实际为 {key!r} ({source})") from None

        if not isinstance(cfg, dict):
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 必须是对象 ({source})")

        missing = [f for f in ('client_num', 'model_name') if f not in cfg]
        if missing:
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 缺少字段 {missing} ({source})")

        declared_k = cfg.get('design_K')
        if declared_k is not None:
            try:
                declared_k = int(declared_k)
            except (TypeError, ValueError):
                raise CollusionPresetError(
                    f"抗合谋配置表条目 {size} 的 design_K 类型非法: "
                    f"{declared_k!r} ({source})") from None
            if declared_k < 1:
                raise CollusionPresetError(
                    f"抗合谋配置表条目 {size} 的 design_K 必须 >= 1 "
                    f"(实际 {declared_k}, {source})")

        try:
            spec = CollusionSpec(
                size=size,
                client_num=int(cfg['client_num']),
                model_name=str(cfg['model_name']),
                design_K=declared_k,
                source=source,
            )
        except (TypeError, ValueError) as e:
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 的字段类型非法: {e} ({source})") from e

        if spec.client_num < 2:
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 的 client_num 必须 >= 2 "
                f"(实际 {spec.client_num}, {source})")
        if spec.size > spec.client_num:
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 的合谋规模超过客户端数 "
                f"{spec.client_num} ({source})")

        specs[size] = spec

    if not specs:
        raise CollusionPresetError(
            f"抗合谋配置表为空 ({source}), 至少需要一条 K -> 规格 的映射")

    return specs


def resolve_spec(presets: Dict[int, CollusionSpec],
                 size: int) -> CollusionSpec:
    """按合谋规模取出规格；缺失时给出可操作的错误信息。

    返回的是**规格的拷贝**：``validate_layout`` 会把磁盘上的 design.K
    写回规格，调用方不应因此污染配置表里的共享对象。

    Raises:
        CollusionPresetError: 配置表中没有该 K。
    """
    if size in presets:
        return presets[size].copy()

    available = ', '.join(str(k) for k in sorted(presets))
    raise CollusionPresetError(
        f"抗合谋配置表中没有合谋规模 K={size} 的条目。\n"
        f"   已配置的 K: {available}\n"
        f"   配置表来源: {next(iter(presets.values())).source}\n"
        f"   处理方式（二选一）:\n"
        f"     1) 改用已配置的规模, 例如 --collusion_size {min(presets)};\n"
        f"     2) 在配置表中添加 K={size} 的条目, 并先生成对应密钥矩阵:\n"
        f"        python train_key_matrix.py --model_type <model_name> "
        f"--client_num <client_num> \\\n"
        f"            --profile collusion --collusion_K <design_K>")


def resolve_key_matrix_path(base_dir: str, spec: CollusionSpec) -> str:
    """由规格拼出密钥矩阵目录的路径。

    Args:
        base_dir: 密钥矩阵根目录（如 ``save/key_matrix_collusion``）。
        spec: 规格。

    Returns:
        ``<base_dir>/<model_name>/client<client_num>``。
    """
    return os.path.join(base_dir, spec.key_matrix_dirname()).replace('\\', '/')


def validate_layout(layout, spec: CollusionSpec) -> Optional[str]:
    """校验磁盘上的见证点布局与配置表声明是否一致。

    Args:
        layout: 已加载的 WitnessLayout（可为 None）。
        spec: 规格。

    Returns:
        若布局的设计容量不足以覆盖该 K，返回一条告警文案（None 表示无告警）。

    Raises:
        CollusionPresetError: 布局缺失 / 版本过旧 / design.K 与配置表冲突。
    """
    if layout is None:
        raise CollusionPresetError(
            f"密钥矩阵目录 {resolve_key_matrix_path('save/key_matrix_collusion', spec)} "
            f"下缺少 witness_layout.json, 无法进行抗合谋归因。\n"
            f"   请重新生成: python train_key_matrix.py --model_type "
            f"{spec.model_name} --client_num {spec.client_num} \\\n"
            f"       --profile collusion --collusion_K "
            f"{spec.design_K if spec.design_K else spec.size}")

    layout_k = getattr(getattr(layout, 'design', None), 'K', None)
    if layout_k is None:
        return None
    layout_k = int(layout_k)

    # 配置表声明的 design_K 与磁盘布局不一致 -> 配置过期
    if spec.design_K is not None and spec.design_K != layout_k:
        raise CollusionPresetError(
            f"见证点布局的 design.K={layout_k} 与配置表声明的 "
            f"design_K={spec.design_K} 不一致 (合谋规模 K={spec.size})。\n"
            f"   配置表已过期, 请二选一:\n"
            f"     1) 用该布局的实际 design.K={layout_k} 更新配置表条目;\n"
            f"     2) 重新生成密钥矩阵使 design_K 变为 {spec.design_K}: "
            f"python train_key_matrix.py --model_type {spec.model_name} "
            f"--client_num {spec.client_num} \\\n"
            f"          --profile collusion --collusion_K {spec.design_K}")

    # 把磁盘上的 design.K 作为权威值写回规格
    spec.design_K = layout_k

    cap = spec.search_cap
    if cap is not None and not spec.covers_size():
        return (f"布局的 design.K={layout_k} 推出的归因搜索上界 {cap} "
                f"小于合谋规模 K={spec.size}，运行时已自动放宽到 "
                f"{spec.size}（见证点计数解码对任意 K <= 客户端数 都成立）。"
                f"若希望设计容量与评测规模一致，请把该 K 的 design_K 提到 "
                f">= {spec.size - SEARCH_MARGIN} 并重新生成密钥矩阵。")

    return None


def resolve_collusion_spec(size: int,
                           presets_raw: Optional[str] = None,
                           default_path: str = DEFAULT_PRESET_PATH
                           ) -> Tuple[CollusionSpec, Dict[int, CollusionSpec]]:
    """一步到位：加载配置表 -> 按 K 取规格。

    Args:
        size: 合谋规模 K。
        presets_raw: JSON 字符串或文件路径；None 时用默认路径。
        default_path: 默认配置表路径。

    Returns:
        (规格, 完整配置表)

    Raises:
        CollusionPresetError: 配置缺失 / 不合法 / 没有该 K。
    """
    presets = load_presets(presets_raw, default_path)
    return resolve_spec(presets, size), presets
