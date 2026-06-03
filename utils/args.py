import argparse
from typing import Dict, Any


def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= 基础配置参数 ========================
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')

    # ========================= 数据集和模型参数 ========================
    parser.add_argument('--dataset', type=str, default='chestmnist', choices=['chestmnist', 'cifar10', 'cifar100'], help='name of dataset')
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

    # ========================= 水印与密钥矩阵配置 ========================
    parser.add_argument('--enable_watermark', action='store_true', default=True, help='enable watermark embedding')
    parser.add_argument('--watermark_mode', type=str, default='enhanced', choices=['enhanced', 'normal'], help='watermark embedding mode')
    parser.add_argument('--use_key_matrix', action='store_true', default=True, help='use saved key matrices to embed watermark after local training')
    parser.add_argument('--key_matrix_dir', type=str, default='save/key_matrix', help='directory containing generated key matrices')

    # ========================= 模型泄漏追踪配置 ========================
    parser.add_argument('--enable_leakage_tracking', action='store_true', default=True, help='enable model leakage tracking simulation and detection')
    parser.add_argument('--leak_interval', type=int, default=30, help='leak simulation interval (every N rounds), set to 0 to disable')
    parser.add_argument('--leak_attack_mode', type=str, default='gaussian_noise', choices=['none', 'gaussian_noise'], help='attack mode after model leakage')
    parser.add_argument('--leak_noise_weak', type=float, default=0.01, help='weak noise scale for level 1 (ratio of mean watermark value)')
    parser.add_argument('--leak_noise_medium', type=float, default=0.05, help='medium noise scale for level 2 (ratio of mean watermark value)')
    parser.add_argument('--leak_noise_strong', type=float, default=0.1, help='strong noise scale for level 3 (ratio of mean watermark value)')

    # ========================= 差分隐私参数 ========================
    parser.add_argument('--dp', action='store_true', default=False, help='enable differential privacy')
    parser.add_argument('--sigma', type=float, default=0.1, help='Gaussian noise standard deviation')

    # ========================= 保存路径参数 ========================
    parser.add_argument('--save_excel_dir', type=str, default='save/excel', help='directory to save Excel metrics')
    parser.add_argument('--save_model_dir', type=str, default='save', help='directory to save model files')
    parser.add_argument('--log_file', type=str, default='./logs/console.logs', help='log file path')
    parser.add_argument('--data_root', type=str, default='./data', help='dataset root directory for downloads and cache')

    args = parser.parse_args()

    from .key_matrix_utils import get_key_matrix_path
    import os

    args.key_matrix_path = get_key_matrix_path(args.key_matrix_dir, args.model_name, args.client_num)

    if not os.path.exists(args.key_matrix_path):
        print(f'⚠️  警告: 密钥矩阵目录不存在: {args.key_matrix_path}')
        print(f'   请先运行: python train_key_matrix.py --model_type {args.model_name} --client_num {args.client_num}')
        args.key_matrix_path = args.key_matrix_dir
    else:
        print(f'✅ 找到密钥矩阵目录: {args.key_matrix_path}')

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
