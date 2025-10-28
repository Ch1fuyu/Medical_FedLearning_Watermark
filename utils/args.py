import argparse
from typing import Dict, Any

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= 基础配置参数 ========================
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
    
    # ========================= 数据集和模型参数 ========================
    parser.add_argument('--dataset', type=str, default='chestmnist', choices=['chestmnist', 'cifar10', 'cifar100'], help="name of dataset")
    parser.add_argument('--model_name', type=str, default='resnet', choices=['alexnet', 'resnet'],
                        help='model architecture name')
    # 便捷别名：--model 等价于 --model_name
    parser.add_argument('--model', dest='model_name', type=str, choices=['alexnet', 'resnet'], help='alias of --model_name')
    parser.add_argument('--num_classes', default=14, type=int, help='number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    # 高层预设开关
    parser.add_argument('--preset', type=str, default=None, help='experiment preset name for one-click config')
    parser.add_argument('--override', type=str, default=None, help='comma-separated key=value overrides')
    
    # ========================= 联邦学习核心参数 ========================
    parser.add_argument('--epochs', type=int, default=150, help='total communication rounds')
    parser.add_argument('--local_ep', type=int, default=2, help="local epochs per client: E")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size: B")
    parser.add_argument('--client_num', type=int, default=10, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1, help="fraction of participating clients: C")
    parser.add_argument('--iid', action='store_true', default=True, help='IID data distribution')
    
    # ========================= 优化器参数 ========================
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for local updates (0.1 is too high, changed to 0.001)')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    
    # ========================= 训练控制参数 ========================
    parser.add_argument('--log_interval', default=1, type=int, help='evaluation interval')
    parser.add_argument('--baseline_mode', action='store_true', default=False,
                        help='run baseline training without watermark (equivalent to --enable_watermark=False)')
    
    # ========================= 损失函数参数 ========================
    parser.add_argument('--class_weights', action='store_true', default=False,
                        help='use class weights for imbalanced dataset')
    
    # ========================= 多重损失函数参数 (MultiLoss) ========================
    parser.add_argument('--use_multiloss', action='store_true', default=True,
                        help='enable multi-term loss function for watermark robustness')
    parser.add_argument('--multiloss_init_a', type=float, default=0.6523,
                        help='initial alpha parameter for MultiLoss')
    parser.add_argument('--multiloss_init_b', type=float, default=0.0000800375825259,
                        help='initial beta parameter for MultiLoss')
    parser.add_argument('--multiloss_alpha_early', type=float, default=0.00005,
                        help='alpha value for early training phase (first 30% of epochs), increased for better watermark robustness')
    parser.add_argument('--multiloss_alpha_late', type=float, default=0.0001,
                        help='alpha value for late training phase (last 70% of epochs), increased for better watermark robustness')
    
    # ========================= Focal Loss 参数 ========================
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='enable FocalLoss for imbalanced multi-label classification')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focal loss gamma parameter (focusing parameter), range: [0, 5]')
    parser.add_argument('--focal_reduction', type=str, default='mean', choices=['mean', 'sum', 'none'],
                        help='focal loss reduction method')
    
    # ========================= 水印与密钥矩阵配置 ========================
    parser.add_argument('--enable_watermark', action='store_true', default=True,
                        help='enable watermark embedding (set to False for baseline training)')
    parser.add_argument('--watermark_mode', type=str, default='enhanced', choices=['enhanced', 'normal'],
                        help='watermark embedding mode: enhanced (每5轮融合) or normal (训练后嵌入)')
    parser.add_argument('--use_key_matrix', action='store_true', default=True,
                        help='use saved key matrices to embed watermark after local training')
    parser.add_argument('--key_matrix_dir', type=str, default='save/key_matrix',
                        help='directory containing generated key matrices')
    parser.add_argument('--encoder_path', type=str, default='save/autoencoder/encoder.pth',
                        help='path to the trained autoencoder encoder weights')
    
    # ========================= 水印缩放参数 ========================
    parser.add_argument('--enable_watermark_scaling', action='store_true', default=True,
                        help='enable watermark parameter scaling for better embedding')
    parser.add_argument('--scaling_factor', type=float, default=1.0,
                        help='fixed scaling factor for watermark parameters')
    
    # ========================= 差分隐私参数 ========================
    parser.add_argument('--dp', action='store_true', default=False, help='enable differential privacy')
    parser.add_argument('--sigma', type=float, default=0.1, help='Gaussian noise standard deviation')
    
    # ========================= 保存路径参数 ========================
    parser.add_argument('--save_excel_dir', type=str, default='save/excel', 
                        help='directory to save Excel metrics')
    parser.add_argument('--save_model_dir', type=str, default='save', 
                        help='directory to save model files')
    parser.add_argument('--log_file', type=str, default='./logs/console.logs', 
                        help='log file path')
    # 数据根目录（用于数据下载/缓存目录）
    parser.add_argument('--data_root', type=str, default='./data',
                        help='dataset root directory for downloads and cache')
    
    args = parser.parse_args()

    # ========================= 自动密钥矩阵路径生成 ========================
    # 根据模型类型和客户端数量自动生成密钥矩阵路径
    from .key_matrix_utils import get_key_matrix_path
    import os
    
    # 生成密钥矩阵路径
    args.key_matrix_path = get_key_matrix_path(args.key_matrix_dir, args.model_name, args.client_num)
    
    # 检查密钥矩阵是否存在
    if not os.path.exists(args.key_matrix_path):
        print(f"⚠️  警告: 密钥矩阵目录不存在: {args.key_matrix_path}")
        print(f"   请先运行: python train_key_matrix.py --model_type {args.model_name} --client_num {args.client_num}")
        # 使用基础目录作为回退
        args.key_matrix_path = args.key_matrix_dir
    else:
        print(f"✅ 找到密钥矩阵目录: {args.key_matrix_path}")

    # ========================= 预设注册表与自动推导 ========================
    DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
        # ChestMNIST：多标签，14 类，输入通道根据数据处理转换为 RGB
        'chestmnist': {
            'task_type': 'multilabel',
            'num_classes': 14,
            'in_channels': 3,
            'input_size': 28,
            'normalize_mean': [0.5, 0.5, 0.5],
            'normalize_std': [0.5, 0.5, 0.5],
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
        'resnet': {
            'default_variant': 'cifar',  # 3x3 stem, stride=1（当前实现已近似该结构）
        },
        'alexnet': {
            'default_variant': 'imagenet',
        },
    }

    EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
        # 一键配置
        'cifar10_resnet18_baseline': {
            'dataset': 'cifar10',
            'model_name': 'resnet',
            'optim': 'sgd',
            'lr': 0.1,
            'epochs': 200,
            'batch_size': 128,
        },
        'cifar100_resnet_baseline': {
            'dataset': 'cifar100',
            'model_name': 'resnet',
            'optim': 'sgd',
            'lr': 0.1,
            'epochs': 200,
            'batch_size': 128,
        },
        'imagenet_resnet18_baseline': {
            'dataset': 'imagenet',
            'model_name': 'resnet',
            'optim': 'sgd',
            'lr': 0.1,
            'epochs': 90,
            'batch_size': 256,
        },
    }

    def _apply_simple_overrides(namespace: argparse.Namespace, overrides: str):
        if not overrides:
            return
        pairs = [p.strip() for p in overrides.split(',') if p.strip()]
        for pair in pairs:
            if '=' not in pair:
                continue
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not hasattr(namespace, key):
                # 忽略未知键，保持向后兼容
                continue
            orig = getattr(namespace, key)
            # 尝试按原类型转换
            casted = value
            try:
                if isinstance(orig, bool):
                    casted = value.lower() in ('1', 'true', 'yes', 'y')
                elif isinstance(orig, int):
                    casted = int(value)
                elif isinstance(orig, float):
                    casted = float(value)
                else:
                    casted = value
            except Exception:
                casted = value
            setattr(namespace, key, casted)

    # 1) 应用 experiment 级预设（若提供）
    # 保持 “上方默认” 更高优先级
    if args.preset and args.preset in EXPERIMENT_PRESETS:
        _skip_when_preset = {'lr', 'epochs', 'batch_size'}
        preset_items = EXPERIMENT_PRESETS[args.preset].items()
        vars(args).update({k: v for k, v in preset_items if k not in _skip_when_preset and k in vars(args)})

    # 2) 根据 dataset 推导通用参数（除非用户显式提供）
    ds_key = (args.dataset or '').lower()
    if ds_key in DATASET_PRESETS:
        ds_cfg = DATASET_PRESETS[ds_key]
        # num_classes
        if 'num_classes' in ds_cfg and (args.num_classes is None or args.num_classes == 14 and ds_key != 'chestmnist'):
            args.num_classes = ds_cfg['num_classes']
        # in_channels
        if 'in_channels' in ds_cfg and args.in_channels is None:
            args.in_channels = ds_cfg['in_channels']
        # 对 batch_size 做温和建议（用户未修改时才采用）
        if getattr(args, 'batch_size', None) in (None, 128):
            if 'default_batch_size' in ds_cfg:
                args.batch_size = ds_cfg['default_batch_size']
        # 附加：记录 task_type，供 trainer/metrics 使用（不改变现有逻辑）
        setattr(args, 'task_type', ds_cfg.get('task_type', 'multilabel'))
        setattr(args, 'input_size', ds_cfg.get('input_size', 28))

    # 3) 根据 model 选择默认变体（仅记录，不改变现有构造）
    mn_key = (args.model_name or '').lower()
    if mn_key in MODEL_PRESETS:
        setattr(args, 'model_variant', MODEL_PRESETS[mn_key].get('default_variant', 'imagenet'))

    # 4) 应用用户 overrides（最后一步，优先级最高）
    _apply_simple_overrides(args, args.override)
    
    # 5) 处理基准模式和水印开关的互斥逻辑
    if args.baseline_mode:
        args.enable_watermark = False
        print("🔧 基准模式已启用，自动关闭水印嵌入")
    
    # 6) 根据水印开关调整相关参数
    if not args.enable_watermark:
        args.use_key_matrix = False
        args.enable_watermark_scaling = False
        print("🔧 水印已关闭，相关功能已禁用")

    return args