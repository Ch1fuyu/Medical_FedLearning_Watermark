"""批量生成各种配置下的密钥矩阵与见证点布局。

用途
----
在一台**全新的机器**上，把实验需要的全部密钥矩阵一次性铺好，
之后直接运行 ``python main.py`` 即可开始实验，不必逐个敲
``train_key_matrix.py``。

它会：

1. 扫描抗合谋配置表（``config/collusion_presets*.json``），取出所有
   实际用到的 ``(数据集, 模型, 客户端数, design_K)`` 组合；
2. 逐个调用 :mod:`train_key_matrix` 的生成逻辑，产出
   ``key_matrix.pth`` / ``positions.json`` / ``witness_layout.json``；
3. 可选地一并生成**默认分支**（传统水印基线）所需的密钥矩阵；
4. 最后做一次完整性自检，列出"已生成 / 缺失"的清单。

用法
----
::

    # 1) 生成抗合谋分支所需的全部密钥矩阵（按配置表自动枚举）
    python generate_all_key_matrices.py

    # 2) 同时生成默认分支（传统水印基线）的密钥矩阵
    python generate_all_key_matrices.py --include-default

    # 3) 只生成指定数据集
    python generate_all_key_matrices.py --datasets chestmnist

    # 4) 只生成指定模型 / 客户端数
    python generate_all_key_matrices.py --models resnet alexnet \
        --client_nums 10

    # 5) 覆盖已存在的（默认跳过）
    python generate_all_key_matrices.py --force

    # 6) 只检查不生成
    python generate_all_key_matrices.py --check-only

    # 7) 使用指定的见证点密钥（跨机器复现同一套水印时用）
    python generate_all_key_matrices.py --witness_secret_file my.key

密钥
----
见证点权威值由密钥单向派生（HMAC-SHA256 PRF）。**同一密钥生成的所有
布局互相兼容**；换密钥则新旧布局的水印互不兼容。

默认使用 ``save/keys/witness_secret_collusion.key``（不存在时自动创建）。
若要在一台新机器上复现与另一台机器**完全一致**的水印，把那边的密钥
文件拷过来，用 ``--witness_secret_file`` 指定即可。

注意：运行时**不需要**密钥文件——见证点权威值已经固化在
``witness_layout.json`` 里（``point_values`` 字段）。
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ====================================================================
#  配置枚举
# ====================================================================

#: 抗合谋配置表里出现的模型类型
KNOWN_MODELS = ('resnet', 'alexnet')

#: 默认分支（传统水印基线）默认生成的客户端数
DEFAULT_CLIENT_NUMS = (5, 10, 20, 50)


def _config_dir() -> str:
    return os.path.join(ROOT, 'config')


def discover_specs(datasets: Optional[List[str]] = None,
                   models: Optional[List[str]] = None,
                   client_nums: Optional[List[int]] = None
                   ) -> List[Dict]:
    """扫描抗合谋配置表，枚举出所有需要生成的密钥矩阵规格。

    抗合谋配置表（``config/collusion_presets*.json``）的格式是::

        {"<K>": {"<model_name>": {"client_num": N, "dataset": "...", ...}}}

    同一个 ``(数据集, 模型, 客户端数)`` 可能在多个 K 下重复出现
    （因为 design_K 通常都一样），这里做去重。

    Args:
        datasets: 只保留这些数据集；None 表示全部。
        models: 只保留这些模型；None 表示全部。
        client_nums: 只保留这些客户端数；None 表示全部。

    Returns:
        规格列表，每项为 ``{'dataset', 'model', 'client_num', 'design_K'}``，
        按 (数据集, 模型, 客户端数) 排序。
    """
    from utils.collusion_presets import load_presets

    specs: Dict[Tuple[str, str, int], Dict] = {}

    config_files = sorted(
        f for f in os.listdir(_config_dir())
        if f.startswith('collusion_presets') and f.endswith('.json')
    )
    for fname in config_files:
        path = os.path.join(_config_dir(), fname)
        try:
            presets = load_presets(path)
        except Exception as e:  # noqa: BLE001
            print(f'⚠️  跳过不可用的配置表 {fname}: {e}')
            continue

        for (size, model), spec in presets.items():
            dataset = spec.dataset or 'chestmnist'
            key = (dataset, model, spec.client_num)
            if key in specs:
                # 同一组合出现多次：保留 design_K 最大（最宽松）的那次
                if spec.design_K and spec.design_K > specs[key]['design_K']:
                    specs[key]['design_K'] = spec.design_K
                continue
            specs[key] = {
                'dataset': dataset,
                'model': model,
                'client_num': spec.client_num,
                'design_K': spec.design_K if spec.design_K else size,
            }

    # 过滤
    out = []
    for (dataset, model, client_num), spec in specs.items():
        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue
        if client_nums and client_num not in client_nums:
            continue
        out.append(spec)

    out.sort(key=lambda s: (s['dataset'], s['model'], s['client_num']))
    return out


# ====================================================================
#  生成
# ====================================================================

def _build_model(model_type: str, dataset: str):
    """按 (模型, 数据集) 构造主任务模型。"""
    from models.alexnet import AlexNet
    from models.resnet import resnet18

    num_classes, in_channels, input_size = _dataset_spec(dataset)

    if model_type == 'resnet':
        return resnet18(num_classes=num_classes, in_channels=in_channels,
                        input_size=input_size)
    if model_type == 'alexnet':
        return AlexNet(in_channels, num_classes, input_size=input_size,
                       dropout_rate=0.5)
    raise ValueError(f"不支持的模型类型: {model_type}")


def _dataset_spec(dataset: str) -> Tuple[int, int, int]:
    """返回 (num_classes, in_channels, input_size)。"""
    dataset = (dataset or '').lower()
    presets = {
        'chestmnist': (14, 1, 28),
        'cifar10': (10, 3, 32),
        'cifar100': (100, 3, 32),
        'dermamnist': (7, 3, 28),
        'octmnist': (4, 1, 28),
        'pneumoniamnist': (2, 1, 28),
        'retinamnist': (5, 3, 28),
        'imagenet': (1000, 3, 224),
    }
    if dataset not in presets:
        raise ValueError(f"不支持的数据集: {dataset}")
    return presets[dataset]


def _target_dir(profile: str, dataset: str, model: str, client_num: int) -> str:
    """密钥矩阵的目标目录（与 utils.args 的推导规则一致）。"""
    from utils.args import default_key_matrix_dir
    base = default_key_matrix_dir(profile, dataset)
    return os.path.join(ROOT, base, model, f'client{client_num}')


def generate_one(profile: str,
                 dataset: str,
                 model: str,
                 client_num: int,
                 design_K: int,
                 secret: Optional[bytes] = None,
                 secret_file: Optional[str] = None,
                 amplitude: Optional[float] = None,
                 seed: int = 42,
                 strategy: str = 'equal',
                 force: bool = False,
                 quiet: bool = False) -> Tuple[bool, str]:
    """生成单个 (profile, 数据集, 模型, 客户端数) 的密钥矩阵。

    Args:
        profile: ``'collusion'``（抗合谋）或 ``'default'``（传统水印基线）。
        dataset: 数据集名。
        model: 模型类型。
        client_num: 客户端数。
        design_K: 见证点设计容量（仅 collusion 分支有意义）。
        secret: 见证点密钥字节；None 时按 secret_file / 默认路径解析。
        secret_file: 见证点密钥文件路径。
        amplitude: 见证点权威值幅度。
        seed: 随机种子。
        strategy: 水印分配策略。
        force: 目标目录已存在时是否覆盖。
        quiet: 是否抑制底层生成器的详细输出。

    Returns:
        (是否成功, 说明信息)。
    """
    import io
    import contextlib

    target = _target_dir(profile, dataset, model, client_num)

    if os.path.isdir(target) and not force:
        return True, f'已存在，跳过（--force 可覆盖）: {target}'

    from train_key_matrix import KeyMatrixGenerator

    # 保存目录：与 utils.args 的规则一致
    from utils.args import default_key_matrix_dir
    save_dir = os.path.join(ROOT, default_key_matrix_dir(profile, dataset))

    model_obj = _build_model(model, dataset)

    generator = KeyMatrixGenerator(
        model=model_obj,
        client_num=client_num,
        watermark_strategy=strategy,
        seed=seed,
        collusion_K=design_K,
        enable_witness=(profile == 'collusion'),
    )

    key_matrices, position_dict = generator.generate_key_matrices()

    witness_layout = generator.generate_witness_layout(
        position_dict, secret=secret, amplitude=amplitude)

    generator.save_key_matrices(
        key_matrices, position_dict, save_dir, model, client_num,
        witness_layout=witness_layout, dataset=dataset)

    if profile == 'collusion' and witness_layout is not None:
        from train_key_matrix import _cross_check_presets
        _cross_check_presets(client_num, model, witness_layout.design.K)

    return True, f'已生成: {target}'


# ====================================================================
#  自检
# ====================================================================

def check_specs(specs: List[Dict], profile: str = 'collusion') -> List[Dict]:
    """检查每个规格对应的目录是否已就绪。

    Args:
        specs: 规格列表。
        profile: ``'collusion'`` 或 ``'default'``。

    Returns:
        每项为 ``{'spec': ..., 'ok': bool, 'detail': str}``。
    """
    results = []
    for spec in specs:
        target = _target_dir(profile, spec['dataset'], spec['model'],
                             spec['client_num'])
        km = os.path.join(target, 'client_0', 'key_matrix.pth')
        pos = os.path.join(target, 'client_0', 'positions.json')
        wl = os.path.join(target, 'witness_layout.json')

        missing = [p for p in (km, pos) if not os.path.exists(p)]
        if profile == 'collusion' and not os.path.exists(wl):
            missing.append(wl)

        ok = not missing
        detail = '就绪' if ok else f'缺少 {len(missing)} 个文件'
        results.append({'spec': spec, 'ok': ok, 'detail': detail,
                        'path': target})
    return results


# ====================================================================
#  主流程
# ====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='批量生成各种配置下的密钥矩阵与见证点布局',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='只生成这些数据集（默认按配置表自动枚举）')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=list(KNOWN_MODELS),
                        help='只生成这些模型（默认按配置表自动枚举）')
    parser.add_argument('--client_nums', nargs='+', type=int, default=None,
                        help='只生成这些客户端数（默认按配置表自动枚举）')
    parser.add_argument('--include-default', action='store_true',
                        help='同时生成默认分支（传统水印基线）的密钥矩阵')
    parser.add_argument('--default-datasets', nargs='+', default=None,
                        help='默认分支的数据集（默认只用 chestmnist）')
    parser.add_argument('--default-client-nums', nargs='+', type=int,
                        default=list(DEFAULT_CLIENT_NUMS),
                        help=f'默认分支的客户端数（默认 {list(DEFAULT_CLIENT_NUMS)}）')
    parser.add_argument('--force', action='store_true',
                        help='覆盖已存在的密钥矩阵目录')
    parser.add_argument('--check-only', action='store_true',
                        help='只检查不生成')
    parser.add_argument('--witness_secret_file', type=str, default=None,
                        help='见证点密钥文件路径（默认 save/keys/witness_secret_collusion.key）')
    parser.add_argument('--witness_amplitude', type=float, default=None,
                        help='见证点权威值幅度（默认 0.05）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--strategy', type=str, default='equal',
                        choices=['equal', 'proportional'], help='水印分配策略')
    args = parser.parse_args()

    print('=' * 74)
    print('批量生成密钥矩阵')
    print('=' * 74)

    # ---- 枚举抗合谋规格 ----
    specs = discover_specs(args.datasets, args.models, args.client_nums)
    if not specs:
        print('❌ 没有找到任何规格。'
              '请检查 config/collusion_presets*.json 是否存在且非空。')
        return 1

    print(f'\n抗合谋分支（profile=collusion）共 {len(specs)} 个规格:')
    for s in specs:
        print(f'  · {s["dataset"]:11s} {s["model"]:8s} '
              f'client_num={s["client_num"]:<4d} design_K={s["design_K"]}')

    # ---- 枚举默认分支规格 ----
    default_specs = []
    if args.include_default:
        ds = args.default_datasets or ['chestmnist']
        ms = args.models or list(KNOWN_MODELS)
        for dataset in ds:
            for model in ms:
                for cn in args.default_client_nums:
                    default_specs.append({
                        'dataset': dataset, 'model': model,
                        'client_num': cn, 'design_K': 3,
                    })
        print(f'\n默认分支（profile=default）共 {len(default_specs)} 个规格:')
        for s in default_specs:
            print(f'  · {s["dataset"]:11s} {s["model"]:8s} '
                  f'client_num={s["client_num"]:<4d}')

    if args.check_only:
        print('\n=== 检查模式（不生成） ===')
        all_results = check_specs(specs, 'collusion')
        if default_specs:
            all_results += check_specs(default_specs, 'default')
        _print_check_report(all_results)
        return 0 if all(r['ok'] for r in all_results) else 2

    # ---- 解析见证点密钥 ----
    secret = None
    try:
        from utils.witness_secret import (
            DEFAULT_SECRET_PATH, resolve_secret, secret_fingerprint,
        )
        secret, source = resolve_secret(
            explicit=None,
            secret_file=args.witness_secret_file,
            default_path=DEFAULT_SECRET_PATH,
            create=True,
        )
        if secret is not None:
            print(f'\n见证点密钥来源: {source} '
                  f'(fingerprint={secret_fingerprint(secret)})')
            if args.witness_secret_file:
                print('  提示: 使用了外部密钥文件，新旧布局将共用该密钥。')
    except Exception as e:  # noqa: BLE001
        print(f'⚠️  见证点密钥解析失败: {e}')
        if not args.include_default:
            return 1

    # ---- 逐个生成 ----
    print('\n' + '=' * 74)
    print('开始生成')
    print('=' * 74)

    ok_count = 0
    fail_count = 0
    started = time.time()

    for profile, group in (('collusion', specs), ('default', default_specs)):
        if not group:
            continue
        print(f'\n--- {profile} 分支 ---')
        for s in group:
            tag = (f'[{profile}] {s["dataset"]}/{s["model"]}/'
                   f'client{s["client_num"]}')
            print(f'\n>>> {tag}')
            try:
                ok, msg = generate_one(
                    profile=profile,
                    dataset=s['dataset'],
                    model=s['model'],
                    client_num=s['client_num'],
                    design_K=s['design_K'],
                    secret=secret if profile == 'collusion' else None,
                    secret_file=args.witness_secret_file,
                    amplitude=args.witness_amplitude,
                    seed=args.seed,
                    strategy=args.strategy,
                    force=args.force,
                )
                if ok:
                    ok_count += 1
                    print(f'    ✓ {msg}')
                else:
                    fail_count += 1
                    print(f'    ✗ {msg}')
            except Exception as e:  # noqa: BLE001
                fail_count += 1
                print(f'    ✗ 失败: {e}')
                traceback.print_exc()

    elapsed = time.time() - started
    print('\n' + '=' * 74)
    print(f'生成完毕: 成功 {ok_count} 个, 失败 {fail_count} 个, '
          f'耗时 {elapsed:.1f}s')

    # ---- 自检 ----
    print('\n=== 完整性自检 ===')
    all_results = check_specs(specs, 'collusion')
    if default_specs:
        all_results += check_specs(default_specs, 'default')
    _print_check_report(all_results)

    if fail_count:
        print('\n❌ 有规格生成失败，请检查上面的错误信息。')
        return 1

    print('\n✅ 全部就绪。现在可以直接运行: python main.py')
    return 0


def _print_check_report(results: List[Dict]) -> None:
    """打印自检报告。"""
    ok = sum(1 for r in results if r['ok'])
    for r in results:
        s = r['spec']
        mark = '✓' if r['ok'] else '✗'
        print(f'  {mark} {s["dataset"]:11s} {s["model"]:8s} '
              f'client_num={s["client_num"]:<4d} {r["detail"]}')
    print(f'\n  共 {len(results)} 项, 就绪 {ok} 项, '
          f'缺失 {len(results) - ok} 项')
    if ok != len(results):
        print('\n  缺失项可用以下命令补齐:')
        print('    python generate_all_key_matrices.py --force')


if __name__ == '__main__':
    sys.exit(main())
