"""抗合谋评测配置表（合谋规模 K + 模型类型 -> 密钥矩阵规格）的解析与校验。

背景
----
抗合谋归因的搜索空间上界原本是

    k_search_cap = layout.design.K + 2

其中 ``layout.design.K`` 来自 ``witness_layout.json``（生成密钥矩阵时
由 ``train_key_matrix.py --collusion_K`` 决定）。如果实际仿真的合谋规模
``collusion_size`` 超出 ``design.K + 2``，真解不在搜索空间内，
归因会**静默失败**（输出空列表），很容易被误读成"方案缺陷"。

本模块把"合谋规模 K + 模型类型 -> 密钥矩阵规格（client_num）"做成
**显式配置表**（默认 ``config/collusion_presets.json``），使得：

- ``main.py`` 在抗合谋模式下按 ``(--collusion_size, --model_name)``
  自动定位密钥矩阵目录，不需要用户手工对齐 ``--client_num``；
- 配置缺失或不合规时**直接退出并说明原因**，而不是跑出一堆错误结果。

配置表格式
----------
::

    {
      "<K>": {
        "<model_name>": {"client_num": 10, "design_K": 3},
        ...
      },
      ...
    }

``design_K`` 可选；省略时以磁盘上的布局文件为准。

关于 design.K
-------------
见证点设计的实质参数（地面集 v、每点标记数 k、副本数 G、关联矩阵 A）
**只取决于客户端数 n_clients**，与 ``design.K`` 无关；``design.K`` 只是
一个"设计容量"标注。因此：

- 配置表的条目只需声明 ``client_num``；
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

#: 允许的模型类型（与 main.py / train_key_matrix.py 的 choices 保持一致）。
KNOWN_MODELS = ('resnet', 'alexnet')


class CollusionPresetError(RuntimeError):
    """抗合谋配置表解析 / 校验失败。

    继承 ``RuntimeError``，便于上层统一捕获并给出可读的错误信息。
    """


class CollusionSpec:
    """某个 (合谋规模 K, 模型类型) 对应的密钥矩阵规格。

    Attributes:
        size: 合谋规模 K。
        model_name: 模型类型（'resnet' / 'alexnet'）。
        client_num: 客户端数。
        design_K: 布局的设计容量（**从磁盘布局读取**，配置表可覆盖）。
        source: 配置来源描述（用于日志）。
    """

    __slots__ = ('size', 'client_num', 'model_name', 'design_K', 'dataset',
                 'source')

    def __init__(self, size: int, client_num: int, model_name: str,
                 design_K: Optional[int] = None, source: str = "",
                 dataset: Optional[str] = None):
        self.size = int(size)
        self.client_num = int(client_num)
        self.model_name = str(model_name)
        self.design_K = int(design_K) if design_K is not None else None
        self.dataset = str(dataset) if dataset else None
        self.source = source

    def copy(self) -> "CollusionSpec":
        """返回一份拷贝。

        ``validate_layout`` 会把磁盘上的 design.K 写回规格，
        因此调用方不应直接改动配置表里的共享对象。
        """
        return CollusionSpec(self.size, self.client_num, self.model_name,
                             self.design_K, self.source, self.dataset)

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


def _parse_model_entry(size: int, model_name: str, cfg: dict,
                       source: str) -> CollusionSpec:
    """解析单条 (K, model) 规格。"""
    if not isinstance(cfg, dict):
        raise CollusionPresetError(
            f"抗合谋配置表条目 K={size} / model={model_name} 必须是对象 "
            f"({source})")

    if 'client_num' not in cfg:
        raise CollusionPresetError(
            f"抗合谋配置表条目 K={size} / model={model_name} "
            f"缺少字段 ['client_num'] ({source})")

    declared_k = cfg.get('design_K')
    if declared_k is not None:
        try:
            declared_k = int(declared_k)
        except (TypeError, ValueError):
            raise CollusionPresetError(
                f"抗合谋配置表条目 K={size} / model={model_name} 的 design_K "
                f"类型非法: {declared_k!r} ({source})") from None
        if declared_k < 1:
            raise CollusionPresetError(
                f"抗合谋配置表条目 K={size} / model={model_name} 的 design_K "
                f"必须 >= 1 (实际 {declared_k}, {source})")

    try:
        spec = CollusionSpec(
            size=size,
            client_num=int(cfg['client_num']),
            model_name=model_name,
            design_K=declared_k,
            source=source,
            dataset=cfg.get('dataset'),
        )
    except (TypeError, ValueError) as e:
        raise CollusionPresetError(
            f"抗合谋配置表条目 K={size} / model={model_name} 的字段类型非法: "
            f"{e} ({source})") from e

    if spec.client_num < 2:
        raise CollusionPresetError(
            f"抗合谋配置表条目 K={size} / model={model_name} 的 client_num "
            f"必须 >= 2 (实际 {spec.client_num}, {source})")
    if spec.size > spec.client_num:
        raise CollusionPresetError(
            f"抗合谋配置表条目 K={size} / model={model_name} 的合谋规模超过"
            f"客户端数 {spec.client_num} ({source})")

    return spec


def load_presets(raw: Optional[str] = None,
                 default_path: str = DEFAULT_PRESET_PATH
                 ) -> Dict[Tuple[int, str], CollusionSpec]:
    """加载并校验抗合谋配置表。

    Args:
        raw: JSON 字符串或文件路径；None 时使用 ``default_path``。
        default_path: 默认配置表路径。

    Returns:
        {(合谋规模 K, 模型类型): CollusionSpec}

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

    specs: Dict[Tuple[int, str], CollusionSpec] = {}
    for key, models_cfg in data.items():
        if key.startswith('_'):
            continue  # 注释字段
        try:
            size = int(key)
        except (TypeError, ValueError):
            raise CollusionPresetError(
                f"抗合谋配置表的键必须是整数 (合谋规模 K), "
                f"实际为 {key!r} ({source})") from None

        if not isinstance(models_cfg, dict):
            raise CollusionPresetError(
                f"抗合谋配置表条目 {size} 必须按模型类型分组 "
                f"(例如 \"{size}\": {{\"resnet\": {{...}}}}), 实际类型 "
                f"{type(models_cfg).__name__} ({source})")

        for model_name, cfg in models_cfg.items():
            if model_name.startswith('_'):
                continue
            if model_name not in KNOWN_MODELS:
                raise CollusionPresetError(
                    f"抗合谋配置表条目 K={size} 的模型类型 {model_name!r} "
                    f"未知 (支持: {', '.join(KNOWN_MODELS)}, {source})")
            spec = _parse_model_entry(size, model_name, cfg, source)
            specs[(size, model_name)] = spec

    if not specs:
        raise CollusionPresetError(
            f"抗合谋配置表为空 ({source}), 至少需要一条 (K, 模型) -> 规格 的映射")

    return specs


def resolve_spec(presets: Dict[Tuple[int, str], CollusionSpec],
                 size: int,
                 model_name: str) -> CollusionSpec:
    """按 (合谋规模, 模型类型) 取出规格；缺失时给出可操作的错误信息。

    返回的是**规格的拷贝**：``validate_layout`` 会把磁盘上的 design.K
    写回规格，调用方不应因此污染配置表里的共享对象。

    Raises:
        CollusionPresetError: 配置表中没有该 (K, 模型)。
    """
    if (size, model_name) in presets:
        return presets[(size, model_name)].copy()

    # 该 K 下有哪些模型可用？
    models_for_size = sorted(m for (s, m) in presets if s == size)
    all_models = sorted({m for (_, m) in presets})
    all_sizes = sorted({s for (s, _) in presets})

    lines = [f"抗合谋配置表中没有 (合谋规模 K={size}, 模型={model_name}) 的条目。"]
    if models_for_size:
        lines.append(f"   K={size} 已配置的模型: {', '.join(models_for_size)}")
    else:
        lines.append(f"   已配置的 K: {', '.join(str(s) for s in all_sizes)}")
        lines.append(f"   已配置的模型: {', '.join(all_models)}")
    lines.append(f"   配置表来源: {next(iter(presets.values())).source}")
    lines.append("   处理方式（三选一）:")
    lines.append(f"     1) 改用已配置的规模, 例如 --collusion_size "
                 f"{min(all_sizes)};")
    if models_for_size:
        lines.append(f"     2) 改用该 K 已配置的模型, 例如 --model_name "
                     f"{models_for_size[0]};")
    lines.append(f"     3) 在配置表中添加 K={size} / {model_name} 的条目, "
                 f"并先生成对应密钥矩阵:")
    lines.append(f"        python train_key_matrix.py --model_type {model_name} "
                 f"--client_num <client_num> \\")
    lines.append(f"            --profile collusion --collusion_K <design_K> "
                 f"--dataset <dataset>")

    raise CollusionPresetError("\n   ".join(lines))


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

    校验项：
      1. 布局存在；
      2. 布局的 design.K 与配置表声明一致（不一致 -> 配置过期，退出）；
      3. 布局生成时用的数据集与配置表声明一致（不一致 -> 退出）。

    Args:
        layout: 已加载的 WitnessLayout（可为 None）。
        spec: 规格。

    Returns:
        若布局的设计容量不足以覆盖该 K，返回一条告警文案（None 表示无告警）。

    Raises:
        CollusionPresetError: 布局缺失 / design.K 或 dataset 与配置表冲突。
    """
    if layout is None:
        raise CollusionPresetError(
            f"密钥矩阵目录 {resolve_key_matrix_path('save/key_matrix_collusion', spec)} "
            f"下缺少 witness_layout.json, 无法进行抗合谋归因。\n"
            f"   请重新生成: python train_key_matrix.py --model_type "
            f"{spec.model_name} --client_num {spec.client_num} \\\n"
            f"       --profile collusion --collusion_K "
            f"{spec.design_K if spec.design_K else spec.size} "
            f"--dataset <dataset>")

    layout_k = getattr(getattr(layout, 'design', None), 'K', None)
    if layout_k is None:
        return None
    layout_k = int(layout_k)

    # 配置表声明的 design_K 与磁盘布局不一致 -> 配置过期
    if spec.design_K is not None and spec.design_K != layout_k:
        raise CollusionPresetError(
            f"见证点布局的 design.K={layout_k} 与配置表声明的 "
            f"design_K={spec.design_K} 不一致 "
            f"(合谋规模 K={spec.size}, 模型={spec.model_name})。\n"
            f"   配置表已过期, 请二选一:\n"
            f"     1) 用该布局的实际 design.K={layout_k} 更新配置表条目;\n"
            f"     2) 重新生成密钥矩阵使 design_K 变为 {spec.design_K}: "
            f"python train_key_matrix.py --model_type {spec.model_name} "
            f"--client_num {spec.client_num} \\\n"
            f"          --profile collusion --collusion_K {spec.design_K} "
            f"--dataset <dataset>")

    # 把磁盘上的 design.K 作为权威值写回规格
    spec.design_K = layout_k

    cap = spec.search_cap
    if cap is not None and not spec.covers_size():
        return (f"布局的 design.K={layout_k} 推出的归因搜索上界 {cap} "
                f"小于合谋规模 K={spec.size}，运行时已自动放宽到 "
                f"{spec.size}（见证点计数解码对任意 K <= 客户端数 都成立）。"
                f"若希望设计容量与评测规模一致，请把该条目的 design_K 提到 "
                f">= {spec.size - SEARCH_MARGIN} 并重新生成密钥矩阵。")

    return None


def validate_dataset(spec: CollusionSpec, key_matrix_path: str) -> Optional[str]:
    """校验密钥矩阵生成时用的数据集与配置表声明是否一致。

    密钥矩阵目录 ``<base>/<model>/client<N>`` **不含数据集维度**，
    而卷积层形状随 ``in_channels`` 变化（chestmnist=1, cifar10=3），
    因此 chestmnist 与 cifar10 的密钥矩阵**不能混用**。

    Args:
        spec: 规格（``spec.dataset`` 为配置表声明的数据集）。
        key_matrix_path: 密钥矩阵目录。

    Returns:
        若数据集不一致，返回一条告警文案；否则返回 None。

    Raises:
        CollusionPresetError: 数据集不一致（属于配置错误，必须退出）。
    """
    if not spec.dataset:
        return None

    import json
    info_path = os.path.join(key_matrix_path, 'key_matrix_info.json')
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001
        return None

    saved_ds = info.get('dataset')
    if not saved_ds or saved_ds == spec.dataset:
        return None

    raise CollusionPresetError(
        f"密钥矩阵的数据集与配置表声明不一致:\n"
        f"   密钥矩阵目录: {key_matrix_path}\n"
        f"   该目录是用 {saved_ds} 生成的, 配置表声明的是 {spec.dataset}\n"
        f"   原因: 密钥矩阵目录不含数据集维度, 而卷积层形状随 in_channels "
        f"变化 (chestmnist=1, cifar10/cifar100=3), 两者不能混用。\n"
        f"   处理方式（二选一）:\n"
        f"     1) 为 {spec.dataset} 重新生成密钥矩阵:\n"
        f"        python train_key_matrix.py --model_type {spec.model_name} \\\n"
        f"            --client_num {spec.client_num} --profile collusion \\\n"
        f"            --collusion_K {spec.design_K} --dataset {spec.dataset}\n"
        f"     2) 若要同时保留两种数据集, 为其中一种指定独立的根目录与"
        f"配置表 (--key_matrix_dir + --collusion_size_configs), 并把配置表"
        f"里该条目的 dataset 改成对应数据集。")


def resolve_collusion_spec(size: int,
                           model_name: str,
                           presets_raw: Optional[str] = None,
                           default_path: str = DEFAULT_PRESET_PATH
                           ) -> Tuple[CollusionSpec, Dict[Tuple[int, str], CollusionSpec]]:
    """一步到位：加载配置表 -> 按 (K, 模型) 取规格。

    Args:
        size: 合谋规模 K。
        model_name: 模型类型。
        presets_raw: JSON 字符串或文件路径；None 时用默认路径。
        default_path: 默认配置表路径。

    Returns:
        (规格, 完整配置表)

    Raises:
        CollusionPresetError: 配置缺失 / 不合法 / 没有该 (K, 模型)。
    """
    presets = load_presets(presets_raw, default_path)
    return resolve_spec(presets, size, model_name), presets
