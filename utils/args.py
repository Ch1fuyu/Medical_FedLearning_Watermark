import argparse
import os
import sys
from typing import Dict, Any


# ====================================================================
#  实验配置中心
# ====================================================================
#  用法：直接改下面的值，然后运行 ``python main.py`` 即可。
#
#  优先级：**命令行参数 > 本配置块 > argparse 默认值**。
#  也就是说配置块只填充"命令行没有显式给出"的项，
#  临时想试个别的值，依然可以直接敲命令行覆盖。
#
#  生效范围：只有 :data:`_CONFIG_TARGET_SCRIPTS` 里的入口脚本会应用本配置块。
#  消融实验（ablation_experiment.py 等）与攻击脚本（pruning_attack.py /
#  finetune_attack.py）各自有独立的命令行参数，不受影响。
#
#  典型场景：跑 AlexNet + CIFAR10 + 10 客户端 + K=3 的抗合谋实验，
#  把下面这些改好，直接 ``python main.py``。
# --------------------------------------------------------------------

#: 只有这些入口脚本会应用 EXPERIMENT_CONFIG。
_CONFIG_TARGET_SCRIPTS = ('main.py',)

EXPERIMENT_CONFIG: Dict[str, Any] = {
    # ---------------- 数据集与模型 ----------------
    'dataset': 'cifar10',          # chestmnist / cifar10 / cifar100
    'model_name': 'alexnet',       # alexnet / resnet
    'num_classes': None,           # None = 按 dataset 预设
    'in_channels': None,           # None = 按 dataset 预设

    # ---------------- 联邦学习 ----------------
    'epochs': 3,                   # 通信轮数 R
    'local_ep': 1,                 # 每客户端本地轮数 E
    'batch_size': 128,             # 本地批大小 B
    'client_num': 10,              # 客户端数（抗合谋模式下由配置表决定）
    'frac': 1.0,                   # 每轮参与比例 C
    'iid': True,                   # IID 数据划分

    # ---------------- 优化器 ----------------
    'optim': 'adam',               # adam / sgd
    'lr': 1e-3,
    'wd': 1e-5,
    'use_lr_scheduler': False,
    'dropout_rate': 0.5,           # 仅 AlexNet 分类层

    # ---------------- 抗合谋评测（见证点）----------------
    # 这是本分支的主实验。换数据集时，密钥矩阵目录与配置表要同步改
    # （密钥矩阵目录不含数据集维度，卷积层形状随 in_channels 变化）。
    'collusion_only': True,        # 一键切到抗合谋评测模式
    'collusion_size': 3,           # 仿真的合谋规模 K
    'collusion_leak_stage': 'post_training',   # pre_training / post_training
    'collusion_attack_mode': 'none',            # none / gaussian_noise
    'collusion_size_configs': 'config/collusion_presets_cifar10.json',
    'key_matrix_dir': 'save/key_matrix_collusion_cifar10',
    'key_matrix_profile': 'collusion',
    'enable_witness': True,        # 启用见证点水印
    'freeze_watermark_region': True,   # 冻结水印区域（抗合谋必须开）

    # ---------------- 差分隐私 ----------------
    'dp': True,
    'sigma': 0.05,

    # ---------------- 单客户端泄漏追踪 ----------------
    # 与抗合谋是两种互斥的威胁模型，跑抗合谋时保持关闭。
    'enable_leakage_tracking': False,
    'disable_single_client_leak': True,
    'leak_interval': 1,
    'leak_attack_mode': 'gaussian_noise',
}
# ====================================================================


def _explicit_cli_options(parser) -> set:
    """收集命令行里**显式给出**的选项名。

    用于判断某个参数是"用户从命令行指定"还是"只有默认值"，
    从而决定要不要被 :data:`EXPERIMENT_CONFIG` 覆盖。

    Args:
        parser: 已构建好的 ArgumentParser。

    Returns:
        显式给出的选项名集合（如 ``{'--dataset', '--epochs'}``）。
    """
    known = set()
    for action in parser._actions:
        known.update(action.option_strings)

    given = set()
    for token in sys.argv[1:]:
        name = token.split('=', 1)[0]
        if name in known:
            given.add(name)
    return given


def _config_applies_to_current_script() -> bool:
    """判断当前运行的入口脚本是否应该应用 :data:`EXPERIMENT_CONFIG`。

    依据是 ``sys.argv[0]`` 的文件名（basename）。这样消融实验
    （``ablation_experiment.py`` 等）与攻击脚本（``pruning_attack.py`` /
    ``finetune_attack.py``）不会因为共用 ``parser_args()`` 而被污染。

    Returns:
        True 表示当前脚本应当应用配置块。
    """
    if not _CONFIG_TARGET_SCRIPTS:
        return True
    try:
        script = os.path.basename(sys.argv[0] or '')
    except Exception:  # noqa: BLE001
        return False
    return script in _CONFIG_TARGET_SCRIPTS


def _apply_experiment_config(parser, args) -> list:
    """把 :data:`EXPERIMENT_CONFIG` 应用到未被命令行显式指定的参数上。

    规则：
      - 配置块里的键若在命令行出现过（包括 ``--no_xxx`` 这类反向开关），
        **不覆盖**，命令行优先；
      - 其余情况用配置块的值覆盖 argparse 默认值；
      - 当前脚本不在 :data:`_CONFIG_TARGET_SCRIPTS` 里时**完全不应用**。

    Args:
        parser: 已构建好的 ArgumentParser。
        args: 已解析的 Namespace（原地修改）。

    Returns:
        被配置块实际改动的参数名列表。
    """
    if not _config_applies_to_current_script():
        return []

    explicit = _explicit_cli_options(parser)

    # dest -> 该参数的所有选项名（含 --no_xxx 反向开关）
    dest_options = {}
    for action in parser._actions:
        dest_options.setdefault(action.dest, set()).update(action.option_strings)

    applied = []
    for key, value in EXPERIMENT_CONFIG.items():
        if not hasattr(args, key):
            continue
        if dest_options.get(key, {f'--{key}'}) & explicit:
            continue  # 命令行显式指定，优先级更高
        setattr(args, key, value)
        applied.append(key)

    return applied


class ArgConfigError(SystemExit):
    """参数配置不合法。

    继承 ``SystemExit``，因此在 ``parser_args()`` 里抛出后，
    Python 会以非零退出码结束程序并打印消息——这正是"配置不对就
    直接退出并说明"所需要的语义（而不是抛一个没人处理的异常，
    或者带着错误配置继续跑出一堆无效结果）。
    """

    def __init__(self, message: str):
        super().__init__(f"\n❌ 配置错误：{message}\n")


def _check_model_compatibility(args, spec) -> None:
    """校验密钥矩阵与本次实验的模型结构是否匹配。

    **为什么必须校验**：卷积层形状随 ``in_channels`` 变化
    （chestmnist=1, cifar10/cifar100=3），而密钥矩阵里的位置索引是
    按具体形状生成的。若用 chestmnist 生成的密钥矩阵跑 CIFAR10，
    第一层卷积的形状对不上，水印会嵌入到错误的位置或直接被跳过。

    判定依据（按可靠性排序）：
      1. 直接比较 ``client_0/key_matrix.pth`` 里各卷积张量的形状与
         当前模型同名参数是否一致——这是**最准确**的依据，因为
         这些张量就是水印掩码实际作用的对象；
      2. 老版本没有 ``model_signature`` 字段时，退化为比较
         ``key_matrix_info.json`` 里的卷积参数量。

    注意：**不比较总参数量**。总参数量含分类层（FC），而 FC 的形状
    随 ``num_classes`` 变化却与水印位置无关（密钥矩阵只覆盖卷积层），
    用总参数量判断会产生误报。

    不匹配时直接退出并说明如何重新生成。
    """
    import json

    info_path = os.path.join(args.key_matrix_path, 'key_matrix_info.json')
    if not os.path.exists(info_path):
        return  # 目录不存在的情况已由调用方报错

    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001
        return

    # 当前实验的模型
    try:
        from models.alexnet import AlexNet
        from models.resnet import resnet18
        if args.model_name == 'resnet':
            model = resnet18(num_classes=args.num_classes,
                             in_channels=args.in_channels,
                             input_size=args.input_size)
        else:
            model = AlexNet(args.in_channels, args.num_classes,
                            input_size=args.input_size,
                            dropout_rate=getattr(args, 'dropout_rate', 0.5))
        named = dict(model.named_parameters())
    except Exception:  # noqa: BLE001
        return

    # ---- 依据 1：直接比较 key_matrix.pth 里的卷积张量形状 ----
    km_path = os.path.join(args.key_matrix_path, 'client_0', 'key_matrix.pth')
    if os.path.exists(km_path):
        try:
            import torch
            km = torch.load(km_path, map_location='cpu', weights_only=False)
            diffs = []
            for name, tensor in km.items():
                if name not in named:
                    continue
                if tuple(named[name].shape) != tuple(tensor.shape):
                    diffs.append((name, tuple(tensor.shape),
                                  tuple(named[name].shape)))
            if diffs:
                raise ArgConfigError(
                    f"密钥矩阵与本次实验的模型结构不匹配:\n"
                    f"   密钥矩阵目录: {args.key_matrix_path}\n"
                    f"   该目录的卷积层形状是按另一个 in_channels 生成的\n"
                    f"   本次实验: {args.model_name} + in_channels="
                    f"{args.in_channels} (dataset={args.dataset})\n"
                    f"   形状差异: {diffs[:3]}\n"
                    f"   处理方式: 为本次实验重新生成密钥矩阵\n"
                    f"     python train_key_matrix.py --model_type "
                    f"{args.model_name} \\\n"
                    f"         --client_num {spec.client_num} --profile "
                    f"collusion \\\n"
                    f"         --collusion_K {spec.design_K} "
                    f"--dataset {args.dataset}")
            return
        except ArgConfigError:
            raise
        except Exception:  # noqa: BLE001
            pass

    # ---- 依据 2：model_signature（新版生成器写入）----
    saved_sig = info.get('model_signature')
    if saved_sig:
        current_sig = {
            name: list(param.shape)
            for name, param in named.items()
            if 'conv' in name.lower() and 'weight' in name.lower()
            and len(param.shape) == 4
        }
        if saved_sig != current_sig:
            diff = [(n, saved_sig.get(n), c)
                    for n, c in current_sig.items() if saved_sig.get(n) != c]
            raise ArgConfigError(
                f"密钥矩阵与本次实验的模型结构不匹配:\n"
                f"   密钥矩阵目录: {args.key_matrix_path}\n"
                f"   本次实验: {args.model_name} + in_channels={args.in_channels} "
                f"(dataset={args.dataset})\n"
                f"   形状差异: {diff[:3]}\n"
                f"   处理方式: 为本次实验重新生成密钥矩阵\n"
                f"     python train_key_matrix.py --model_type {args.model_name} \\\n"
                f"         --client_num {spec.client_num} --profile collusion \\\n"
                f"         --collusion_K {spec.design_K} --dataset {args.dataset}")
        return

    # ---- 依据 3（最弱）：旧版本无签名信息，跳过校验 ----
    # 无法在不加载 key_matrix.pth 的情况下判断，而上面的依据 1 已经
    # 覆盖了这种情况（旧目录同样有 key_matrix.pth）。


def _resolve_collusion_spec(args) -> None:
    """按 (合谋规模 K, 模型类型) 从配置表解析密钥矩阵规格，并写回 args。

    行为：
      - 加载抗合谋配置表（``--collusion_size_configs`` 或默认路径）；
      - 按 ``(args.collusion_size, args.model_name)`` 取规格；
      - 用规格里的 ``client_num`` 定位密钥矩阵目录；
      - 校验磁盘上的见证点布局与配置表声明一致（design.K、布局存在性）。

    注意：**不覆盖** ``args.model_name``——模型类型是本次实验的显式选择，
    配置表只是"该 (K, 模型) 用哪套密钥矩阵"的映射表。

    任何一步失败都会抛出 :class:`ArgConfigError` 直接终止运行。
    """
    from .collusion_presets import (
        CollusionPresetError,
        resolve_collusion_spec,
        resolve_key_matrix_path,
        validate_dataset,
        validate_layout,
    )

    try:
        spec, presets = resolve_collusion_spec(
            args.collusion_size, args.model_name, args.collusion_size_configs)

        # 用配置表里的规格覆盖客户端数（配置表是权威来源）
        args.client_num = spec.client_num
        args.collusion_K = spec.design_K
        args.collusion_spec = spec
        args.collusion_presets = presets

        # 定位密钥矩阵目录
        args.key_matrix_path = resolve_key_matrix_path(args.key_matrix_dir, spec)
        if not os.path.isdir(args.key_matrix_path):
            raise CollusionPresetError(
                f"配置表指定的密钥矩阵目录不存在: {args.key_matrix_path}\n"
                f"   (K={spec.size}, model_name={spec.model_name}, "
                f"client_num={spec.client_num})\n"
                f"   请先生成（--dataset 必须与评测用的数据集一致，"
                f"因为 conv 层形状随 in_channels 变化）:\n"
                f"     python train_key_matrix.py --model_type "
                f"{spec.model_name} --client_num {spec.client_num} \\\n"
                f"         --profile collusion --collusion_K {spec.design_K} "
                f"--dataset <dataset>")

        # 校验布局与配置表一致
        layout = None
        try:
            from .collusion_resistant import find_witness_layout, load_witness_layout
            layout_path = find_witness_layout(args.key_matrix_path)
            if layout_path:
                layout = load_witness_layout(layout_path)
        except Exception as e:  # noqa: BLE001
            raise CollusionPresetError(
                f"见证点布局加载失败 ({args.key_matrix_path}): {e}") from e
        warning = validate_layout(layout, spec)

        # 校验密钥矩阵生成时用的数据集与配置表声明一致
        # （密钥矩阵目录不含数据集维度，chestmnist 与 cifar10 会共用）
        validate_dataset(spec, args.key_matrix_path)

        # 校验配置表声明的数据集与本次实验是否一致
        # （密钥矩阵目录不含数据集维度，一套目录只能对应一个数据集）
        if spec.dataset and args.dataset and spec.dataset != args.dataset:
            raise CollusionPresetError(
                f"配置表声明的数据集与本次实验不一致:\n"
                f"   配置表条目 (K={spec.size}, {spec.model_name}) 声明 "
                f"dataset={spec.dataset}\n"
                f"   本次实验: dataset={args.dataset}\n"
                f"   原因: 密钥矩阵目录 <base>/<model>/client<N> 不含数据集维度，"
                f"而卷积层形状随 in_channels 变化 "
                f"(chestmnist=1, cifar10/cifar100=3)，两者不能混用。\n"
                f"   处理方式（二选一）:\n"
                f"     1) 把本次实验的数据集改成 {spec.dataset};\n"
                f"     2) 为 {args.dataset} 单独准备密钥矩阵目录与配置表，"
                f"并用 --key_matrix_dir + --collusion_size_configs 指定:\n"
                f"        python train_key_matrix.py --model_type "
                f"{spec.model_name} --client_num {spec.client_num} \\\n"
                f"            --profile collusion --collusion_K "
                f"{spec.design_K} --dataset {args.dataset} \\\n"
                f"            --save_dir save/key_matrix_collusion_{args.dataset}")

        # 校验密钥矩阵与本次实验的模型结构是否匹配
        # （卷积层形状随 in_channels 变化：chestmnist=1, cifar10=3）
        _check_model_compatibility(args, spec)

        cap = spec.search_cap
        print(f'📋 抗合谋配置: K={spec.size}, model={spec.model_name} -> '
              f'client_num={spec.client_num}, '
              f'dataset={spec.dataset or args.dataset}, '
              f'design_K={spec.design_K} '
              f'(归因搜索上限 {cap if cap is not None else "自适应"})')
        print(f'✅ 找到密钥矩阵目录: {args.key_matrix_path}')
        if warning:
            print(f'⚠️  {warning}')

    except CollusionPresetError as e:
        raise ArgConfigError(str(e)) from e


def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= 基础配置参数 ========================
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')

    # ========================= 数据集和模型参数 ========================
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['chestmnist', 'cifar10', 'cifar100'], help='name of dataset')
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['alexnet', 'resnet'], help='model architecture name')
    parser.add_argument('--num_classes', default=None, type=int, help='number of classes')
    parser.add_argument('--in_channels', type=int, default=None, help='input channels')
    parser.add_argument('--preset', type=str, default=None, help='experiment preset name for one-click config')
    parser.add_argument('--override', type=str, default=None, help='comma-separated key=value overrides')

    # ========================= 联邦学习核心参数 ========================
    parser.add_argument('--epochs', type=int, default=150, help='total communication rounds')
    parser.add_argument('--local_ep', type=int, default=2, help='local epochs per client: E')
    parser.add_argument('--batch_size', type=int, default=128, help='local batch size: B')
    parser.add_argument('--client_num', type=int, default=5, help='number of clients: K')
    parser.add_argument('--frac', type=float, default=1, help='fraction of participating clients: C')
    parser.add_argument('--iid', action='store_true', default=True, help='IID data distribution')

    # ========================= 优化器参数 ========================
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for local updates')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (L2 regularization)')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False, help='use cosine annealing learning rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate for AlexNet classifier')

    # ========================= 训练控制参数 ========================
    parser.add_argument('--log_interval', default=1, type=int, help='evaluation interval')
    parser.add_argument('--patience', type=int, default=0, help='early stopping patience (number of rounds without improvement)')
    parser.add_argument('--baseline_mode', action='store_true', default=False, help='run baseline training without watermark')

    # ========================= 损失函数参数 ========================
    parser.add_argument('--class_weights', action='store_true', default=False, help='use class weights for imbalanced dataset')

    # ========================= 多重损失函数参数 (MultiLoss) ========================
    parser.add_argument('--use_drift_reg', action='store_true', default=True, help='enable drift penalty in watermark regularization ablation')
    parser.add_argument('--no_drift_reg', action='store_false', dest='use_drift_reg', help='disable drift penalty in watermark regularization ablation')
    parser.add_argument('--use_margin_reg', action='store_true', default=True, help='enable margin penalty in watermark regularization ablation')
    parser.add_argument('--no_margin_reg', action='store_false', dest='use_margin_reg', help='disable margin penalty in watermark regularization ablation')
    parser.add_argument('--margin_ratio', type=float, default=0.001, help='margin threshold ratio relative to non-watermark gradients')
    parser.add_argument('--drift_lambda', type=float, default=1.0, help='weight for drift penalty term')
    parser.add_argument('--margin_lambda', type=float, default=1.0, help='weight for margin penalty term')
    parser.add_argument('--multiloss_alpha_early', type=float, default=5e-5, help='alpha value for early training phase')
    parser.add_argument('--multiloss_alpha_late', type=float, default=1e-4, help='alpha value for late training phase')
    parser.add_argument('--watermark_grad_target_ratio', type=float, default=0.3, help='target ratio of watermark gradient to non-watermark gradient; set to 0 to disable watermark gradient capping')

    # ========================= 水印与密钥矩阵配置 ========================
    parser.add_argument('--enable_watermark', action='store_true', default=True, help='enable watermark embedding')
    parser.add_argument('--watermark_mode', type=str, default='enhanced', choices=['enhanced', 'normal'], help='watermark embedding mode')
    parser.add_argument('--use_key_matrix', action='store_true', default=True, help='use saved key matrices to embed watermark after local training')
    parser.add_argument('--key_matrix_dir', type=str, default=None,
                        help='directory containing generated key matrices (default: decided by --key_matrix_profile)')
    parser.add_argument('--key_matrix_profile', type=str, default='default',
                        choices=['default', 'collusion'],
                        help='key matrix profile: default=save/key_matrix (traditional watermark baseline), '
                             'collusion=save/key_matrix_collusion (collusion-resistant branch). '
                             'Using separate directories avoids mismatch between key_matrix and witness_layout '
                             'when switching between branches.')

    # ========================= 模型泄漏追踪配置 ========================
    parser.add_argument('--enable_leakage_tracking', action='store_true', default=False, help='enable model leakage tracking simulation and detection')
    parser.add_argument('--no_source_tracing', action='store_false', dest='enable_source_tracing', help='disable leakage source tracing while keeping customized model distribution')
    parser.add_argument('--disable_single_client_leak', action='store_true', default=False,
                        help='完全关闭单客户端泄漏仿真（预训练/后训练/恶意客户端选择），只保留合谋泄漏')
    parser.add_argument('--leak_interval', type=int, default=1, help='leak simulation interval (every N rounds), set to 0 to disable')
    parser.add_argument('--leak_attack_mode', type=str, default='gaussian_noise', choices=['none', 'gaussian_noise'], help='attack mode after model leakage')
    parser.add_argument('--leak_noise_weak', type=float, default=0.01, help='weak noise scale for level 1 (ratio of mean watermark value)')
    parser.add_argument('--leak_noise_medium', type=float, default=0.05, help='medium noise scale for level 2 (ratio of mean watermark value)')
    parser.add_argument('--leak_noise_strong', type=float, default=0.1, help='strong noise scale for level 3 (ratio of mean watermark value)')

    # ========================= 抗合谋水印（见证点）配置 ========================
    parser.add_argument('--collusion_only', action='store_true', default=False,
                        help='抗合谋评测模式：一键关闭单客户端泄漏仿真与追踪、'
                             '开启合谋检测、切换到 collusion 密钥矩阵目录。'
                             '等价于 --disable_single_client_leak --no_source_tracing '
                             '--enable_collusion_detection --key_matrix_profile collusion')
    parser.add_argument('--enable_witness', action='store_true', default=True,
                        help='enable collusion-resistant witness point watermark (shared witness pool)')
    parser.add_argument('--no_enable_witness', action='store_false', dest='enable_witness',
                        help='disable witness point watermark (fall back to exclusive-slice watermark only)')
    parser.add_argument('--collusion_K', type=int, default=2,
                        help='maximum collusion size the witness design resists '
                             '(设计容量; 实际生效的是 witness_layout.json 中的 design.K)')
    parser.add_argument('--enable_collusion_detection', action='store_true', default=False,
                        help='enable collusion attribution: identify which K clients colluded')
    parser.add_argument('--collusion_size', type=int, default=2,
                        help='number of colluding clients to simulate (>=2)')
    parser.add_argument('--collusion_size_configs', type=str, default=None,
                        help='抗合谋评测配置表 (JSON 字符串或文件路径)。'
                             '把"合谋规模 K + 模型类型 -> 密钥矩阵规格"做成显式配置，'
                             '运行 main.py 时据此自动定位密钥矩阵目录; '
                             '若该配置缺失或不合规则直接退出并说明原因。'
                             '格式: {"<K>": {"<model_name>": {"client_num": 10, '
                             '"design_K": 3}, ...}, ...}')
    parser.add_argument('--collusion_leak_stage', type=str, default='pre_training',
                        choices=['pre_training', 'post_training'],
                        help='collusion stage: average customized models (pre) or trained models (post)')
    parser.add_argument('--collusion_attack_mode', type=str, default='none',
                        choices=['none', 'gaussian_noise'],
                        help='attack applied to the colluded model')

    # ========================= 见证点密钥 / 冻结 ========================
    parser.add_argument('--witness_secret', type=str, default=None,
                        help='见证点密钥（十六进制字符串或任意文本）。'
                             '见证点权威值由该密钥单向派生，'
                             '不再取自 encoder 参数切片，'
                             '因此合谋者无法反推或伪造见证点签名。'
                             '生产环境务必显式指定并妥善保管；'
                             '不指定时每次生成的见证点水印互不兼容。')
    parser.add_argument('--witness_amplitude', type=float, default=None,
                        help='见证点权威值幅度（默认 0.05）')
    parser.add_argument('--witness_secret_file', type=str, default=None,
                        help='从文件读取见证点密钥（推荐，避免密钥进入命令行历史）')
    parser.add_argument('--freeze_watermark_region', action='store_true',
                        default=False,
                        help='冻结水印区域：本地训练期间客户端自有水印区域 P_k '
                             '与抗合谋见证点区域都不参与参数更新，'
                             '保证服务器下发的权威值被原封不动地带回，'
                             '并保证见证点信号不随轮次衰减。')

    # ========================= 差分隐私参数 ========================
    parser.add_argument('--dp', action='store_true', default=False, help='enable differential privacy')
    parser.add_argument('--sigma', type=float, default=0.1, help='Gaussian noise standard deviation')

    # ========================= 保存路径参数 ========================
    parser.add_argument('--save_excel_dir', type=str, default='save/excel', help='directory to save Excel metrics')
    parser.add_argument('--save_model_dir', type=str, default='save', help='directory to save model files')
    parser.add_argument('--log_file', type=str, default='./logs/console.logs', help='log file path')
    parser.add_argument('--data_root', type=str, default='./data', help='dataset root directory for downloads and cache')

    # ========================= 模型存档格式参数 ========================
    parser.add_argument('--save_full_weights', action='store_true', default=True,
                        help='save full network weights in the checkpoint (required by pruning/finetune attacks)')
    parser.add_argument('--no_save_full_weights', action='store_false', dest='save_full_weights',
                        help='save only watermark-region params in the checkpoint (much smaller, NOT loadable by attacks)')
    parser.add_argument('--compress_model', action='store_true', default=False,
                        help='compress saved weights to float16 (saves space, but pruning attacks need exact 0/non-zero values)')
    parser.add_argument('--no_compress_model', action='store_false', dest='compress_model',
                        help='save weights in float32 without compression')

    args = parser.parse_args()

    # ========================= 应用实验配置中心 =========================
    # 命令行显式给出的参数优先；其余用 utils.args.EXPERIMENT_CONFIG 覆盖。
    # 这样"改配置块 -> 直接 python main.py"与"命令行临时覆盖"两种用法都能用。
    _applied = _apply_experiment_config(parser, args)
    if _applied:
        print('⚙️  实验配置中心 (utils/args.py: EXPERIMENT_CONFIG) 已应用: '
              + ', '.join(sorted(_applied)))

    from .key_matrix_utils import get_key_matrix_path
    import os

    # ========================= 抗合谋评测模式联动 =========================
    # --collusion_only 一键切换到"只做合谋攻击"的评测配置：
    #   1. 关闭单客户端泄漏仿真（预训练/后训练/恶意客户端选取）
    #   2. 关闭单客户端泄漏源追踪
    #   3. 开启合谋检测与归因
    #   4. 密钥矩阵切到 collusion 专用目录
    # 单客户端泄漏与合谋泄漏是两种互斥的威胁模型：前者的泄漏模型是
    # 单个模型，可用余弦相似度溯源；后者是多个模型的平均，只能用
    # 见证点子集匹配归因。两者不应在同一实验里混跑。
    if args.collusion_only:
        args.disable_single_client_leak = True
        args.enable_source_tracing = False
        args.enable_collusion_detection = True
        # 显式指定 --key_matrix_dir 时，仍需切到 collusion profile：
        # profile 决定"是否启用见证点"，而不是目录名。若不切，
        # 见证点会被禁用，抗合谋归因无从谈起。
        # （目录本身用显式传入的值，不被覆盖。）
        if args.key_matrix_profile == 'default':
            args.key_matrix_profile = 'collusion'
        # 抗合谋分支必须冻结水印区域（含见证点），否则：
        #   - 任务梯度 / DP 噪声会把见证点冲掉，信号在若干轮后消失；
        #   - 见证点区域会随 w_t 漂移，"固定基线"的前提不成立。
        args.freeze_watermark_region = True
        print('🔒 抗合谋评测模式：已关闭单客户端泄漏仿真与追踪，'
              '开启合谋检测，已自动开启水印区域冻结')

    # 密钥矩阵目录：优先 --key_matrix_dir，否则按 profile 决定
    # 不同实验分支用互不干扰的目录，避免切换分支时
    # key_matrix 与 witness_layout 不匹配（save/ 未纳入 git 版本管理）。
    if args.key_matrix_dir is None:
        args.key_matrix_dir = ('save/key_matrix_collusion'
                               if args.key_matrix_profile == 'collusion'
                               else 'save/key_matrix')

    # 先按 profile + 命令行参数推导一个"默认"路径；抗合谋模式下
    # 下面的配置表解析会用更精确的规格覆盖它。
    args.key_matrix_path = get_key_matrix_path(args.key_matrix_dir,
                                                args.model_name,
                                                args.client_num)

    # ========================= 分支隔离：见证点只属于抗合谋分支 =========================
    # 见证点（抗合谋水印）只在 key_matrix_profile == 'collusion' 时启用。
    # 默认分支（传统水印基线）完全不使用见证点：既不嵌入、也不冻结。
    # 这样两条分支彻底分离，来回切换不会互相污染。
    if args.key_matrix_profile != 'collusion':
        if args.enable_witness:
            print('ℹ️  默认分支（传统水印基线）不启用见证点水印；'
                  '抗合谋分支请用 --collusion_only 或 --key_matrix_profile collusion')
        args.enable_witness = False

    # ========================= 抗合谋配置表联动 =========================
    # 抗合谋模式下，"合谋规模 K -> 密钥矩阵规格"由显式配置表决定
    # （默认 config/collusion_presets.json）。这样可以：
    #   1. 自动定位密钥矩阵目录（不需要手工对齐 --client_num）；
    #   2. 保证 design.K 与 K 匹配（否则归因会静默失败）；
    #   3. 配置缺失或不合法时**直接退出并说明**，而不是跑出错结果。
    #
    # 注意放在"分支隔离"之后：只有 collusion 分支才需要见证点布局。
    # 真正的调用点在下方（数据集预设生效之后），因为模型兼容性校验
    # 需要知道 in_channels / input_size。

    # ========================= 见证点密钥 =========================
    # 支持从文件读取密钥（避免密钥出现在命令行历史 / 进程列表中）。
    # 未显式指定时，读取抗合谋分支的**默认长期密钥文件**；
    # 文件不存在则自动创建，保证"密钥长期不变"。
    if getattr(args, 'witness_secret_file', None):
        try:
            with open(args.witness_secret_file, 'r', encoding='utf-8') as f:
                args.witness_secret = f.read().strip()
            print(f'🔑 已从文件读取见证点密钥: {args.witness_secret_file}')
        except Exception as e:
            print(f'⚠️  读取见证点密钥文件失败: {e}')
    elif not getattr(args, 'witness_secret', None) and args.enable_witness:
        # 仅抗合谋分支需要见证点密钥
        try:
            from .witness_secret import (DEFAULT_SECRET_PATH,
                                         load_or_create_secret,
                                         secret_fingerprint)
            _secret, _created = load_or_create_secret(DEFAULT_SECRET_PATH,
                                                      create=True)
            print(f'🔑 见证点密钥: {DEFAULT_SECRET_PATH} '
                  f'({"新建" if _created else "已存在"}, '
                  f'fingerprint={secret_fingerprint(_secret)})')
        except Exception as e:
            print(f'⚠️  见证点密钥初始化失败: {e}')

    # 密钥矩阵目录检查放在下方（数据集预设生效、抗合谋配置解析之后），
    # 因为抗合谋模式下的精确路径由配置表决定。

    DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
        'chestmnist': {
            'task_type': 'multilabel',
            'num_classes': 14,
            'in_channels': 1,
            'input_size': 28,
            'normalize_mean': [0.5],
            'normalize_std': [0.5],
            'default_batch_size': 128,
            'metrics': ['loss', 'acc_label', 'acc_sample', 'auc'],
        },
        'cifar10': {
            'task_type': 'multiclass',
            'num_classes': 10,
            'in_channels': 3,
            'input_size': 32,
            'normalize_mean': [0.4914, 0.4822, 0.4465],
            'normalize_std': [0.2470, 0.2435, 0.2616],
            'default_batch_size': 128,
            'metrics': ['loss', 'top1'],
        },
        'cifar100': {
            'task_type': 'multiclass',
            'num_classes': 100,
            'in_channels': 3,
            'input_size': 32,
            'normalize_mean': [0.5071, 0.4867, 0.4408],
            'normalize_std': [0.2675, 0.2565, 0.2761],
            'default_batch_size': 128,
            'metrics': ['loss', 'top1'],
        },
        'imagenet': {
            'task_type': 'multiclass',
            'num_classes': 1000,
            'in_channels': 3,
            'input_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'default_batch_size': 256,
            'metrics': ['loss', 'top1'],
        },
    }

    MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
        'resnet': {'default_variant': 'cifar'},
        'alexnet': {'default_variant': 'imagenet'},
    }

    EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
        'cifar10_resnet18_baseline': {'dataset': 'cifar10', 'model_name': 'resnet', 'optim': 'sgd', 'lr': 0.1, 'epochs': 200, 'batch_size': 128},
        'cifar100_resnet_baseline': {'dataset': 'cifar100', 'model_name': 'resnet', 'optim': 'sgd', 'lr': 0.1, 'epochs': 200, 'batch_size': 128},
        'imagenet_resnet18_baseline': {'dataset': 'imagenet', 'model_name': 'resnet', 'optim': 'sgd', 'lr': 0.1, 'epochs': 90, 'batch_size': 256},
    }

    def _apply_simple_overrides(namespace: argparse.Namespace, overrides: str):
        if not overrides:
            return
        for pair in [p.strip() for p in overrides.split(',') if p.strip()]:
            if '=' not in pair:
                continue
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not hasattr(namespace, key):
                continue
            current = getattr(namespace, key)
            if isinstance(current, bool):
                value = value.lower() in ('1', 'true', 'yes', 'y', 'on')
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(namespace, key, value)

    if args.preset and args.preset in EXPERIMENT_PRESETS:
        for key, value in EXPERIMENT_PRESETS[args.preset].items():
            setattr(args, key, value)

    ds_key = (args.dataset or '').lower()
    if ds_key in DATASET_PRESETS:
        ds_cfg = DATASET_PRESETS[ds_key]
        if args.num_classes is None:
            args.num_classes = ds_cfg['num_classes']
        if args.in_channels is None:
            args.in_channels = ds_cfg['in_channels']
        if getattr(args, 'task_type', None) is None:
            args.task_type = ds_cfg['task_type']
        if args.batch_size == 128 and 'default_batch_size' in ds_cfg:
            args.batch_size = ds_cfg['default_batch_size']
        args.input_size = ds_cfg['input_size']
        args.normalize_mean = ds_cfg['normalize_mean']
        args.normalize_std = ds_cfg['normalize_std']
        args.metrics = ds_cfg['metrics']

    if args.model_name in MODEL_PRESETS:
        args.model_variant = MODEL_PRESETS[args.model_name]['default_variant']

    if not hasattr(args, 'task_type'):
        args.task_type = 'multiclass'
    if not hasattr(args, 'input_size'):
        args.input_size = 224
    if not hasattr(args, 'normalize_mean'):
        args.normalize_mean = [0.5]
    if not hasattr(args, 'normalize_std'):
        args.normalize_std = [0.5]
    if not hasattr(args, 'metrics'):
        args.metrics = ['loss']

    _apply_simple_overrides(args, args.override)

    # ========================= 抗合谋配置表联动 =========================
    # 放在**数据集预设生效之后**：模型兼容性校验需要知道
    # in_channels / input_size（卷积层形状随 in_channels 变化，
    # chestmnist=1 而 cifar10/cifar100=3）。
    if args.key_matrix_profile == 'collusion' and args.enable_collusion_detection:
        _resolve_collusion_spec(args)

    # 密钥矩阵目录存在性检查：抗合谋模式已由配置表解析出精确路径，
    # 这里不要覆盖它；其余情况按 profile 推导。
    if not os.path.exists(args.key_matrix_path):
        print(f'⚠️  警告: 密钥矩阵目录不存在: {args.key_matrix_path}')
        print(f'   请先运行: python train_key_matrix.py --model_type {args.model_name} '
              f'--client_num {args.client_num} --profile {args.key_matrix_profile} '
              f'--dataset {args.dataset}')
        args.key_matrix_path = args.key_matrix_dir
    else:
        print(f'✅ 找到密钥矩阵目录: {args.key_matrix_path}')

    if args.baseline_mode:
        args.enable_watermark = False
        args.use_key_matrix = False
        args.enable_leakage_tracking = False

    args.leak_noise_configs = {
        1: args.leak_noise_weak,
        2: args.leak_noise_medium,
        3: args.leak_noise_strong,
    }

    return args
