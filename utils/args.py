import argparse
import os
from typing import Dict, Any


def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= 基础配置参数 ========================
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
    
    # ========================= 数据集和模型参数 ========================
    parser.add_argument('--dataset', type=str, default='pathmnist',
                       choices=['chestmnist', 'cifar10', 'cifar100',
                                'pneumoniamnist', 'pathmnist'],
                       help="name of dataset")
    parser.add_argument('--model_name', type=str, default='resnet',
                       choices=['alexnet', 'resnet'],
                       help='model architecture name')
    parser.add_argument('--num_classes', default=None, type=int, help='number of classes')
    parser.add_argument('--in_channels', type=int, default=None, help='input channels')
    
    # ========================= 训练参数 ========================
    parser.add_argument('--epochs', type=int, default=150, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")

    # ========================= 优化器参数 ========================
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay (L2 regularization)')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False, 
                       help='use cosine annealing learning rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.5, 
                       help='dropout rate for AlexNet classifier (default: 0.5)')
    
    # ========================= 训练控制参数 ========================
    parser.add_argument('--patience', type=int, default=0,
                        help='early stopping patience (number of epochs without improvement)')
    parser.add_argument('--baseline_mode', action='store_true', default=False,
                        help='run baseline training without watermark')
    
    # ========================= 损失函数参数 ========================
    parser.add_argument('--class_weights', action='store_true', default=False,
                        help='use class weights for imbalanced dataset')
    
    # ========================= 多重损失函数参数 (MultiLoss) ========================
    parser.add_argument('--use_multiloss', action='store_true', default=True,
                        help='enable multi-term loss function for watermark robustness')
    parser.add_argument('--multiloss_init_a', type=float, default=0.65,
                        help='initial alpha parameter for MultiLoss')
    parser.add_argument('--multiloss_init_b', type=float, default=0.00008,
                        help='initial beta parameter for MultiLoss')
    parser.add_argument('--multiloss_alpha_early', type=float, default=0.005,
                        help='alpha value for early training phase (first 30%% of epochs)')
    parser.add_argument('--multiloss_alpha_late', type=float, default=0.001,
                        help='alpha value for late training phase (last 70%% of epochs)')
    
    # ========================= 水印与密钥矩阵配置 ========================
    parser.add_argument('--enable_watermark', action='store_true', default=True,
                        help='enable watermark embedding (set to False for baseline training)')
    parser.add_argument('--use_key_matrix', action='store_true', default=True,
                        help='use key matrix for watermark embedding (alias for enable_watermark)')
    parser.add_argument('--watermark_mode', type=str, default='enhanced', 
                       choices=['enhanced', 'normal'],
                       help='watermark embedding mode: enhanced (每5轮融合) or normal (训练后嵌入)')
    parser.add_argument('--key_matrix_dir', type=str, default='save/key_matrix',
                        help='directory containing generated key matrices')
    parser.add_argument('--encoder_path', type=str, default='save/autoencoder/encoder.pth',
                        help='path to the trained autoencoder encoder weights')
    
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
    parser.add_argument('--data_root', type=str, default='./data',
                        help='dataset root directory for downloads and cache')
    
    args = parser.parse_args()

    # ========================= 密钥矩阵路径 ========================
    # 密钥矩阵路径：save/key_matrix/{model_name}/
    args.key_matrix_path = os.path.join(args.key_matrix_dir, args.model_name).replace('\\', '/')
    if not os.path.exists(args.key_matrix_path):
        print(f"[WARN] 密钥矩阵目录不存在: {args.key_matrix_path}")
        print(f"   请先运行: python train_key_matrix.py --model_type {args.model_name} --in_channels {args.in_channels} --input_size 28")
    else:
        print(f"[OK] 找到密钥矩阵目录: {args.key_matrix_path}")

    # ========================= 预设注册表与自动推导 ========================
    DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
        # ========== MedMNIST 数据集 ==========
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
        'pneumoniamnist': {
            'task_type': 'binary',
            'num_classes': 2,
            'in_channels': 1,
            'input_size': 28,
            'normalize_mean': [0.5],
            'normalize_std': [0.5],
            'default_batch_size': 128,
            'metrics': ['loss', 'auc', 'top1'],
        },
        'pathmnist': {
            'task_type': 'multiclass',
            'num_classes': 9,
            'in_channels': 3,
            'input_size': 28,
            'normalize_mean': [0.5, 0.5, 0.5],
            'normalize_std': [0.5, 0.5, 0.5],
            'default_batch_size': 128,
            'metrics': ['loss', 'top1', 'auc'],
        },
        # ========== 自然图像数据集 ==========
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
    }

    # 根据 dataset 推导通用参数
    ds_key = (args.dataset or '').lower()
    if ds_key in DATASET_PRESETS:
        ds_cfg = DATASET_PRESETS[ds_key]
        if 'num_classes' in ds_cfg and args.num_classes is None:
            args.num_classes = ds_cfg['num_classes']
        if 'in_channels' in ds_cfg and args.in_channels is None:
            args.in_channels = ds_cfg['in_channels']
        if getattr(args, 'batch_size', None) in (None, 128):
            if 'default_batch_size' in ds_cfg:
                args.batch_size = ds_cfg['default_batch_size']
        setattr(args, 'task_type', ds_cfg.get('task_type', 'multilabel'))
        setattr(args, 'input_size', ds_cfg.get('input_size', 28))
    
    # 处理基准模式
    if args.baseline_mode:
        args.enable_watermark = False
        args.use_key_matrix = False
        print("基准模式已启用，自动关闭水印嵌入")
    
    # 根据水印开关调整相关参数
    if not args.enable_watermark:
        args.use_key_matrix = False
        print("水印已关闭")

    return args
