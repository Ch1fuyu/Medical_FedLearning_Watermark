import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= 基础配置参数 ========================
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
    parser.add_argument('--exp-id', type=int, default=1, help='experiment id')
    parser.add_argument('--eval', action='store_true', default=False, help='for evaluation only')
    
    # ========================= 数据集和模型参数 ========================
    parser.add_argument('--dataset', type=str, default='chestmnist', help="name of dataset (ChestMNIST only)")
    parser.add_argument('--model_name', type=str, default='resnet', choices=['alexnet', 'resnet'],
                        help='model architecture name')
    parser.add_argument('--num_classes', default=14, type=int, help='number of classes (14 for ChestMNIST multi-label)')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    
    # ========================= 联邦学习核心参数 ========================
    parser.add_argument('--client_num', type=int, default=10, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1, help="fraction of participating clients: C")
    parser.add_argument('--local_ep', type=int, default=2, help="local epochs per client: E")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size: B")
    parser.add_argument('--epochs', type=int, default=100, help='total communication rounds')
    parser.add_argument('--sampling_type', type=str, default='uniform', help='client sampling strategy')
    parser.add_argument('--iid', action='store_true', default=True, help='IID data distribution')
    
    # ========================= 优化器参数 ========================
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for local updates')
    parser.add_argument('--lr_outer', type=float, default=1, help='outer learning rate (unused)')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay (disabled to align with official strategy)')
    parser.add_argument('--gamma', type=float, default=0.99, help='exponential weight decay (unused)')
    
    # ========================= 训练控制参数 ========================
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--log_interval', default=1, type=int, help='evaluation interval')
    parser.add_argument('--save-interval', type=int, default=0, help='model save interval')
    
    # ========================= 损失函数参数 ========================
    parser.add_argument('--class_weights', action='store_true', default=False,
                        help='use class weights for imbalanced dataset (aligned with official strategy)')
    parser.add_argument('--pos_weight_factor', type=float, default=1.0, 
                        help='factor to adjust positive class weights')
    
    # ========================= 水印和IPR参数 (暂时关闭) ========================
    parser.add_argument('--wm_length', type=int, default=1000, help='watermark length (temporarily disabled)')
    parser.add_argument('--num_sign', type=int, default=1, help="number of signature users (temporarily disabled)")
    parser.add_argument('--weight_type', default='gamma', choices=['gamma', 'kernel'], help='weight type (temporarily disabled)')
    parser.add_argument('--num_bit', type=int, default=20, help="number of signature bits (temporarily disabled)")
    parser.add_argument('--loss_type', default='sign', choices=['sign', 'CE'], help='signature loss type (temporarily disabled)')
    parser.add_argument('--loss_alpha', type=float, default=0.2, help='signature loss scale factor (temporarily disabled)')
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'], help='normalization type')
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='shuffle', 
                        help='passport key type')
    parser.add_argument('--passport-config', default='passport_configs/alexnet_passport.json',
                        help='passport configuration file')
    
    # ========================= 后门攻击参数 ========================
    parser.add_argument('--backdoor_indis', action='store_false', default=True, help='backdoor in distribution')
    parser.add_argument('--num_back', type=int, default=1, help="number of backdoor users")
    parser.add_argument('--num_trigger', type=int, default=40, help="number of trigger samples")
    
    # ========================= 差分隐私参数 ========================
    parser.add_argument('--dp', action='store_true', default=False, help='enable differential privacy (temporarily disabled)')
    parser.add_argument('--sigma', type=float, default=0.1, help='Gaussian noise standard deviation (temporarily disabled)')
    
    # ========================= 其他参数 ========================
    parser.add_argument('--bp_interval', default=30, type=int, help='backpropagation interval')
    parser.add_argument('--pruning', action='store_true', help='enable model pruning')
    parser.add_argument('--percent', default=5, type=float, help='pruning percentage')
    
    args = parser.parse_args()
    return args