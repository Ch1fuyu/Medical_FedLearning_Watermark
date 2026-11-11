import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from config.globals import set_seed
from models.alexnet import AlexNet
from models.light_autoencoder import LightAutoencoder
from models.resnet import resnet18

# 设置随机种子
set_seed()

# 数据集预设配置
DATASET_PRESETS = {
    'cifar10': {
        'num_classes': 10,
        'in_channels': 3,
        'input_size': 32,
    },
    'cifar100': {
        'num_classes': 100,
        'in_channels': 3,
        'input_size': 32,
    },
    'imagenet': {
        'num_classes': 1000,
        'in_channels': 3,
        'input_size': 224,
    },
}

class KeyMatrixGenerator:
    """密钥矩阵生成器"""
    
    def __init__(self, model, client_num, watermark_strategy='equal', seed=42,
                 collusion_K=3, enable_witness=True):
        """
        初始化密钥矩阵生成器

        Args:
            model: 主任务模型
            client_num: 客户端数量
            watermark_strategy: 水印分配策略 ('equal', 'proportional', 'custom')
            seed: 随机种子
            collusion_K: 抗合谋设计抵抗的最大合谋规模
            enable_witness: 是否生成见证点布局（抗合谋水印）
        """
        self.model = model
        self.client_num = client_num
        self.watermark_strategy = watermark_strategy
        self.seed = seed
        self.collusion_K = collusion_K
        self.enable_witness = enable_witness
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 获取编码器参数数量作为总水印大小
        self.encoder = LightAutoencoder().encoder
        self.encoder_params = sum(param.numel() for param in self.encoder.parameters())
        
        # 获取主任务模型参数信息
        self.model_param_info = self._get_model_param_info()
        self.total_model_params = sum(param_info['numel'] for param_info in self.model_param_info)
        
        # 根据水印比例计算实际水印大小（使用编码器参数作为水印总量）
        self.total_watermark_size = self.encoder_params
        
        print(f"主任务模型总参数数量: {self.total_model_params:,}")
        print(f"编码器参数数量: {self.encoder_params:,}")
        print(f"水印大小: {self.total_watermark_size:,}")
    
    def _get_model_param_info(self):
        """获取模型参数信息（仅包含卷积层参数）"""
        param_info = []
        start_idx = 0
        skipped_count = 0
        for name, param in self.model.named_parameters():
            # 检查是否为卷积层参数：1. 包含 'conv' 和 'weight'
            # 2. 或包含 'downsample.0.weight' (ResNet 的 1x1 卷积) 3. 参数维度为 4D (卷积层权重)
            is_conv_weight = (
                'conv' in name.lower() and 'weight' in name.lower()
            ) or (
                'downsample.0.weight' in name.lower()  # ResNet downsample conv
            )
            
            # 还必须是4D张量（卷积层权重的形状）
            if is_conv_weight and len(param.shape) == 4:
                param_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'start_idx': start_idx
                })
                start_idx += param.numel()
            else:
                skipped_count += 1
                
        if skipped_count > 0:
            print(f"已跳过 {skipped_count} 个非卷积层参数（BN、FC层等）")
        print(f"共提取 {len(param_info)} 个卷积层")
        return param_info
    
    def _calculate_watermark_sizes(self):
        """计算每个客户端的水印大小"""
        if self.watermark_strategy == 'equal':
            # 平均分配
            base_size = self.total_watermark_size // self.client_num
            remainder = self.total_watermark_size % self.client_num
            
            sizes = [base_size] * self.client_num
            # 将余数分配给前几个客户端
            for i in range(remainder):
                sizes[i] += 1
                
        elif self.watermark_strategy == 'proportional':
            # 按比例分配（这里简化为平均分配，可根据需要扩展）
            sizes = [self.total_watermark_size // self.client_num] * self.client_num
            remainder = self.total_watermark_size % self.client_num
            for i in range(remainder):
                sizes[i] += 1
                
        else:
            raise ValueError(f"不支持的水印分配策略: {self.watermark_strategy}")
        
        return sizes
    
    def generate_key_matrices(self):
        """生成所有客户端的密钥矩阵"""
        watermark_sizes = self._calculate_watermark_sizes()
        
        # 获取所有可用的参数位置（使用局部索引）
        all_positions = []
        for param_info in self.model_param_info:
            for i in range(param_info['numel']):
                # 使用局部索引：i (在参数内部的索引)
                all_positions.append((param_info['name'], i))
        
        # 随机打乱位置
        np.random.shuffle(all_positions)
        
        # 为每个客户端分配位置
        key_matrices = {}
        position_dict = {}
        start_idx = 0
        
        for client_id in range(self.client_num):
            watermark_size = watermark_sizes[client_id]
            end_idx = start_idx + watermark_size
            
            # 获取该客户端的水印位置
            client_positions = all_positions[start_idx:end_idx]
            position_dict[client_id] = client_positions
            
            # 生成密钥矩阵（与模型形状相同，但只有指定位置为1）
            key_matrix = self._create_key_matrix(client_positions)
            key_matrices[client_id] = key_matrix
            
            print(f"客户端 {client_id}: 水印大小 {watermark_size}, 位置范围 {start_idx}-{end_idx-1}")
            start_idx = end_idx
        
        return key_matrices, position_dict

    def generate_witness_layout(self, position_dict, all_positions=None,
                                secret=None, amplitude=None):
        """生成见证点布局（抗合谋水印）。

        见证点位置必须是**跨客户端共享**的，否则合谋者各自的水印区域
        互不相交，无法通过参数平均反推合谋集合。

        位置来源：所有 conv 位置中**未被任何客户端分配**的那些。
        这样见证点与现有"每客户端一段独占切片"的水印机制
        **完全不重叠**，不需要改动任何现有 offset 计算逻辑。

        见证点**取值**由服务器密钥单向派生（不再取自 encoder 参数切片），
        因此合谋者即使拼齐自己分到的 encoder 切片也无法反推权威值，
        更无法伪造指向无辜客户端的见证点签名。

        Args:
            position_dict: {client_id: [(param_name, local_idx), ...]}
            all_positions: 全部可用 conv 位置 [(param_name, local_idx), ...]；
                None 时由本类内部重建。
            secret: 见证点密钥（bytes）；None 时由 build_witness_layout 随机生成。
            amplitude: 见证点权威值幅度；None 时用默认值。

        Returns:
            WitnessLayout 或 None（未启用见证点 / 位置不足）
        """
        if not self.enable_witness:
            return None

        from utils.collusion_resistant import build_witness_layout

        # 重建全部可用位置
        if all_positions is None:
            all_positions = []
            for param_info in self.model_param_info:
                for i in range(param_info['numel']):
                    all_positions.append((param_info['name'], i))

        # 已被客户端占用的位置
        used = set()
        for cid in range(self.client_num):
            for pos in position_dict[cid]:
                used.add((pos[0], pos[1]))

        # 未被占用的位置 -> 见证点共享池
        free_pool = [p for p in all_positions if p not in used]

        # 现有切片的单客户端长度
        slice_size = len(position_dict[0]) if self.client_num > 0 else 0

        kwargs = {}
        if secret is not None:
            kwargs['secret'] = secret
        if amplitude is not None:
            kwargs['amplitude'] = amplitude

        try:
            layout = build_witness_layout(
                n_clients=self.client_num,
                K=self.collusion_K,
                encoder_params=self.encoder_params,
                positions=free_pool,
                slice_size=slice_size,
                verbose=False,
                **kwargs,
            )
        except ValueError as e:
            print(f"⚠️ 见证点布局生成失败: {e}")
            return None

        v_raw = layout.design.singer.v
        print(f"见证点布局: v={layout.v} k={layout.design.k} G={layout.G} "
              f"见证点位置={len(layout.positions)} "
              f"(取自 {len(free_pool)} 个未分配位置)")
        if v_raw != layout.v:
            g_before = self.encoder_params // (layout.n_clients * v_raw)
            print(f"  地面集压缩: v_raw={v_raw} -> v_used={layout.v} "
                  f"(剔除 {v_raw - layout.v} 个无人标记的点，"
                  f"G 从 {g_before} 提升到 {layout.G})")
        print(f"  见证点取值 = 密钥派生 (secret_id={layout.secret_id}), "
              f"与 encoder 切片完全解耦")
        print(f"  独占切片区 = [{layout.slice_offset}, {layout.slice_end})  "
              f"({layout.slice_size}/客户端，offset 不变)")
        return layout

    def _create_key_matrix(self, positions):
        """为指定位置创建密钥矩阵"""
        key_matrix = {}
        
        # 初始化所有参数为0
        for param_info in self.model_param_info:
            key_matrix[param_info['name']] = torch.zeros(param_info['shape'])
        
        # 在指定位置设置为1（positions 中存储的是局部索引）
        for param_name, local_idx in positions:
            if param_name not in key_matrix:
                continue
            
            # 验证局部索引范围
            param_info = None
            for info in self.model_param_info:
                if info['name'] == param_name:
                    param_info = info
                    break
            
            if param_info is None:
                continue
            
            if local_idx < 0 or local_idx >= param_info['numel']:
                continue
            
            # 将一维局部索引转换为多维索引
            param_shape = key_matrix[param_name].shape
            multi_idx = np.unravel_index(local_idx, param_shape)
            key_matrix[param_name][multi_idx] = 1.0
        
        return key_matrix
    
    def save_key_matrices(self, key_matrices, position_dict, save_dir, model_type, client_num,
                          witness_layout=None, dataset=None):
        """保存密钥矩阵到文件"""
        # 创建新的目录结构: save_dir/model_type/client{client_num}/
        model_dir = os.path.join(save_dir, model_type)
        client_dir = os.path.join(model_dir, f'client{client_num}')
        os.makedirs(client_dir, exist_ok=True)

        # ---- 覆盖检测 ----
        # 密钥矩阵目录不含数据集维度（<base>/<model>/client<N>），而卷积层
        # 形状随 in_channels 变化，因此 chestmnist 与 cifar10/cifar100
        # 会**共用同一目录**。若用另一个数据集覆盖，会导致原来那套
        # 实验的密钥矩阵失效。此处提前告警。
        info_path = os.path.join(client_dir, 'key_matrix_info.json')
        if dataset and os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    old_info = json.load(f)
                old_ds = old_info.get('dataset')
                if old_ds and old_ds != dataset:
                    print(f'⚠️  警告：密钥矩阵目录已被另一个数据集占用:')
                    print(f'     目录: {client_dir}')
                    print(f'     原有数据集: {old_ds} -> 本次: {dataset}')
                    print(f'     卷积层形状随 in_channels 变化，两者不兼容；'
                          f'本次生成会覆盖原有密钥矩阵。')
                    print(f'     若需同时保留两种数据集，请为其中一种指定不同的 '
                          f'--save_dir。')
            except Exception:  # noqa: BLE001
                pass

        # 保存每个客户端的密钥矩阵
        for client_id, key_matrix in key_matrices.items():
            client_subdir = os.path.join(client_dir, f'client_{client_id}')
            os.makedirs(client_subdir, exist_ok=True)

            # 保存密钥矩阵
            key_matrix_path = os.path.join(client_subdir, 'key_matrix.pth')
            torch.save(key_matrix, key_matrix_path)

            # 保存位置信息
            position_path = os.path.join(client_subdir, 'positions.json')
            with open(position_path, 'w') as f:
                json.dump(position_dict[client_id], f, indent=2)

        # 保存见证点布局（抗合谋水印）
        witness_path = None
        if witness_layout is not None:
            from utils.collusion_resistant import save_witness_layout
            witness_path = os.path.join(client_dir, 'witness_layout.json')
            save_witness_layout(witness_layout, witness_path)
            print(f"见证点布局已保存到: {witness_path}")

        # 保存总体信息
        info = {
            'client_num': self.client_num,
            'watermark_strategy': self.watermark_strategy,
            'total_watermark_size': self.total_watermark_size,
            'total_model_params': self.total_model_params,
            'encoder_params': self.encoder_params,
            'model_type': self.model.__class__.__name__,
            'created_time': datetime.now().isoformat(),
            'seed': self.seed,
            'enable_witness': witness_layout is not None,
            # 生成时使用的数据集（用于跨数据集覆盖检测）
            'dataset': dataset,
            # 模型结构指纹：用于检测"密钥矩阵与实际模型不匹配"。
            # 卷积层形状随 in_channels 变化（chestmnist=1, cifar10=3），
            # 因此换数据集时必须重新生成密钥矩阵。
            'model_signature': self._model_signature(),
        }
        if witness_layout is not None:
            info['collusion_K'] = witness_layout.design.K
            info['witness_v'] = witness_layout.v
            info['witness_k'] = witness_layout.design.k
            info['witness_G'] = witness_layout.G
            info['witness_positions'] = len(witness_layout.positions)

        info_path = os.path.join(client_dir, 'key_matrix_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"密钥矩阵已保存到: {client_dir}")
        return info_path

    def _model_signature(self) -> dict:
        """提取模型结构指纹（卷积层名称 -> 形状）。

        卷积层形状随 ``in_channels`` 变化，因此指纹可用于判断
        "当前密钥矩阵是不是为这个模型结构生成的"。
        """
        sig = {}
        for name, param in self.model.named_parameters():
            if 'conv' in name.lower() and 'weight' in name.lower() \
                    and len(param.shape) == 4:
                sig[name] = list(param.shape)
        return sig

def load_key_matrix(save_dir, client_id):
    """加载指定客户端的密钥矩阵"""
    client_dir = os.path.join(save_dir, f'client_{client_id}')
    key_matrix_path = os.path.join(client_dir, 'key_matrix.pth')
    
    if not os.path.exists(key_matrix_path):
        raise FileNotFoundError(f"客户端 {client_id} 的密钥矩阵不存在: {key_matrix_path}")
    
    return torch.load(key_matrix_path, map_location='cpu')

def load_key_matrix_info(save_dir):
    """加载密钥矩阵信息"""
    info_path = os.path.join(save_dir, 'key_matrix_info.json')
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"密钥矩阵信息文件不存在: {info_path}")
    
    with open(info_path, 'r') as f:
        return json.load(f)

def _cross_check_presets(client_num: int, model_type: str, design_K: int):
    """与抗合谋配置表交叉校验 design_K 是否一致。

    配置表（``config/collusion_presets.json``）里若声明了该
    ``(client_num, model_name)`` 的 design_K，且与本次生成的布局不一致，
    则提示同步更新配置表——否则 main.py 会在启动时报"配置已过期"。

    该检查是**提示性**的，不阻断生成流程。
    """
    try:
        from utils.collusion_presets import load_presets
        presets = load_presets()
    except Exception as e:  # noqa: BLE001
        print(f'ℹ️  跳过抗合谋配置表交叉校验（配置表不可用: {e}）')
        return

    conflicts = []
    matched = []
    for (size, mname), spec in sorted(presets.items()):
        if spec.client_num != client_num or mname != model_type:
            continue
        matched.append(size)
        if spec.design_K is not None and spec.design_K != design_K:
            conflicts.append((size, spec.design_K))

    if conflicts:
        ks = ', '.join(f"K={k}(配置表 design_K={v})" for k, v in conflicts)
        print(f'⚠️  抗合谋配置表与本次生成的布局 design_K 不一致: {ks}，'
              f'实际生成的是 design_K={design_K}。')
        print(f'   请同步更新 config/collusion_presets.json 中对应条目的 '
              f'design_K 为 {design_K}（或删掉该字段以跳过一致性校验）。')
    elif matched:
        print(f'ℹ️  抗合谋配置表已同步 '
              f'(K={matched}, client_num={client_num}, '
              f'model_name={model_type}, design_K={design_K})')


def main():
    parser = argparse.ArgumentParser(description='生成联邦学习密钥矩阵')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=['resnet', 'alexnet'], help='主任务模型类型')
    parser.add_argument('--client_num', type=int, default=5, help='客户端数量')
    parser.add_argument('--watermark_strategy', type=str, default='equal',
                       choices=['equal', 'proportional'], help='水印分配策略')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录（默认按 profile 自动决定）')
    parser.add_argument('--profile', type=str, default='default',
                       choices=['default', 'collusion'],
                       help='配置档案：default=save/key_matrix（传统水印基线），'
                            'collusion=save/key_matrix_collusion（抗合谋分支）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist',
                               'cifar10', 'cifar100', 'imagenet'],
                       help='数据集类型')
    parser.add_argument('--collusion_K', type=int, default=3,
                       help='抗合谋设计抵抗的最大合谋规模（见证点设计用）。'
                            '注意：见证点设计的实质参数（v/k/G/关联矩阵）只取决于 '
                            '--client_num，与本值无关；本值只是"设计容量"标注，'
                            '会写入 key_matrix_info.json 与 witness_layout.json。'
                            '评测用的 K -> 规格映射见 config/collusion_presets.json')
    parser.add_argument('--disable_witness', action='store_true',
                       help='禁用见证点布局（只生成传统的独占切片水印）')
    parser.add_argument('--witness_secret', type=str, default=None,
                       help='见证点密钥（十六进制字符串或任意文本）。'
                            '见证点权威值由该密钥单向派生；'
                            '生产环境必须显式指定并妥善保管，'
                            '否则每次生成的见证点水印互不兼容。')
    parser.add_argument('--witness_secret_file', type=str, default=None,
                       help='见证点密钥文件路径（推荐，避免密钥进入命令行历史）。'
                            '默认使用 save/keys/witness_secret_collusion.key，'
                            '不存在时自动创建（该目录已被 gitignore）。')
    parser.add_argument('--witness_amplitude', type=float, default=None,
                       help='见证点权威值幅度（默认 0.05）')
    
    args = parser.parse_args()

    # 保存目录：优先 --save_dir，否则按 profile 决定
    # profile 让不同实验分支（传统水印基线 / 抗合谋）使用互不干扰的目录，
    # 避免来回切分支时 key_matrix 与 witness_layout 不匹配。
    if args.save_dir is None:
        args.save_dir = ('./save/key_matrix_collusion'
                         if args.profile == 'collusion'
                         else './save/key_matrix')
    print(f'配置档案: {args.profile} -> 保存目录: {args.save_dir}')

    # 使用预设配置获取数据集参数
    dataset_key = args.dataset.lower()
    if dataset_key in DATASET_PRESETS:
        dataset_config = DATASET_PRESETS[dataset_key]
        num_classes = dataset_config['num_classes']
        in_channels = dataset_config['in_channels']
        input_size = dataset_config['input_size']
    else:
        # 回退到手动配置（用于MedMNIST数据集）
        if args.dataset == 'chestmnist':
            num_classes = 14
            in_channels = 1
            input_size = 28
        elif args.dataset == 'dermamnist':
            num_classes = 7
            in_channels = 3
            input_size = 28
        elif args.dataset == 'octmnist':
            num_classes = 4
            in_channels = 1
            input_size = 28
        elif args.dataset == 'pneumoniamnist':
            num_classes = 2
            in_channels = 1
            input_size = 28
        elif args.dataset == 'retinamnist':
            num_classes = 5
            in_channels = 3
            input_size = 28
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建模型
    if args.model_type == 'resnet':
        model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    elif args.model_type == 'alexnet':
        # 使用默认 dropout_rate=0.5，密钥矩阵生成不需要调整 dropout
        model = AlexNet(in_channels, num_classes, input_size=input_size, dropout_rate=0.5)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    print(f"使用模型: {args.model_type}")
    print(f"数据集: {args.dataset}")
    print(f"类别数: {num_classes}, 输入通道数: {in_channels}, 输入尺寸: {input_size}")
    print(f"客户端数量: {args.client_num}")
    print(f"水印分配策略: {args.watermark_strategy}")
    
    # 生成密钥矩阵
    generator = KeyMatrixGenerator(
        model=model,
        client_num=args.client_num,
        watermark_strategy=args.watermark_strategy,
        seed=args.seed,
        collusion_K=args.collusion_K,
        enable_witness=not args.disable_witness,
    )

    key_matrices, position_dict = generator.generate_key_matrices()

    # 解析见证点密钥
    # 优先级：--witness_secret > --witness_secret_file > 默认密钥文件
    # 默认密钥文件路径固定（save/keys/witness_secret_collusion.key），
    # 保证"密钥长期不变"：同一密钥生成的所有布局互相兼容。
    secret = None
    if args.witness_secret or args.witness_secret_file or not args.disable_witness:
        from utils.witness_secret import (
            DEFAULT_SECRET_PATH, resolve_secret, secret_fingerprint,
        )
        default_path = None if args.disable_witness else DEFAULT_SECRET_PATH
        secret, source = resolve_secret(
            explicit=args.witness_secret,
            secret_file=args.witness_secret_file,
            default_path=default_path,
            create=True,
        )
        if secret is not None:
            print(f'见证点密钥来源: {source} '
                  f'(长度 {len(secret)} 字节, fingerprint={secret_fingerprint(secret)})')
    else:
        print('见证点已禁用，不解析密钥')

    # 生成见证点布局（抗合谋水印）
    witness_layout = generator.generate_witness_layout(
        position_dict, secret=secret, amplitude=args.witness_amplitude)

    # 保存密钥矩阵
    info_path = generator.save_key_matrices(key_matrices, position_dict,
                                            args.save_dir, args.model_type,
                                            args.client_num,
                                            witness_layout=witness_layout,
                                            dataset=args.dataset)

    # 与抗合谋配置表交叉校验：配置表里若声明了该 (client_num, model_name)
    # 的 design_K，且与本次生成的布局不一致，提示同步更新配置表。
    if args.profile == 'collusion' and witness_layout is not None:
        _cross_check_presets(args.client_num, args.model_type,
                             witness_layout.design.K)

    # 显示统计信息
    print("\n=== 密钥矩阵生成完成 ===")
    print(f"保存目录: {args.save_dir}")
    print(f"信息文件: {info_path}")
    
    # 验证密钥矩阵
    print("\n=== 验证密钥矩阵 ===")
    for client_id in range(args.client_num):
        key_matrix = key_matrices[client_id]
        total_ones = sum(tensor.sum().item() for tensor in key_matrix.values())
        print(f"客户端 {client_id}: 密钥矩阵中1的数量 = {int(total_ones)}")
    
    print("\n密钥矩阵生成完成！")

if __name__ == '__main__':
    main()
