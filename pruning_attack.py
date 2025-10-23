import copy
import os
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.light_autoencoder import LightAutoencoder
from models.resnet import resnet18
from utils.watermark_reconstruction import WatermarkReconstructor
from utils.delta_pcc_utils import evaluate_delta_pcc, calculate_fixed_tau, format_delta_pcc_result, print_delta_pcc_summary


def extract_model_info_from_path(model_path):
    """
    从模型路径中提取数据集、模型名、模型类型和客户端数量信息
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        dict: 包含dataset、model_name、model_type和client_num的字典
    """
    try:
        # 标准化路径分隔符
        normalized_path = model_path.replace('\\', '/')
        
        # 分割路径
        path_parts = normalized_path.split('/')
        
        # 查找数据集、模型名、模型类型和客户端数量
        dataset = 'unknown'
        model_name = 'unknown'
        model_type = 'resnet'  # 默认值
        client_num = 10  # 默认值
        
        # 从路径中提取信息
        for i, part in enumerate(path_parts):
            if part in ['cifar10', 'cifar100', 'chestmnist']:
                dataset = part
            elif part in ['resnet', 'cnn', 'vgg', 'densenet']:
                model_name = part
                model_type = part
            elif part == 'resnet18':
                model_name = 'resnet'
                model_type = 'resnet'
            elif part == 'cnn_simple':
                model_name = 'cnn'
                model_type = 'cnn'
            elif part.startswith('client') and part[6:].isdigit():
                # 提取客户端数量，如 client10 -> 10
                client_num = int(part[6:])
        
        # 如果从路径中无法确定模型类型，使用模型名
        if model_type == 'resnet' and model_name != 'unknown':
            model_type = model_name
        
        return {
            'dataset': dataset,
            'model_name': model_name,
            'model_type': model_type,
            'client_num': client_num
        }
    except Exception as e:
        print(f"警告：无法从路径 {model_path} 中提取模型信息: {e}")
        return {
            'dataset': 'unknown',
            'model_name': 'unknown',
            'model_type': 'resnet',
            'client_num': 10
        }


def load_test_data(dataset_name: str, batch_size: int = 128, data_dir: str = './data'):
    """
    根据数据集名称加载相应的测试数据
    
    Args:
        dataset_name: 数据集名称 ('cifar10', 'cifar100', 'chestmnist', 'mnist')
        batch_size: 批次大小
        data_dir: 数据目录
        
    Returns:
        测试数据加载器
    """
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'chestmnist':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        from utils.dataset import LocalChestMNISTDataset
        dataset_path = os.path.join(data_dir, 'chestmnist.npz')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"ChestMNIST数据集文件不存在: {dataset_path}")
        
        test_dataset = LocalChestMNISTDataset(dataset_path, split='test', transform=transform)
        
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"✓ 已加载{dataset_name.upper()}测试集: {len(test_dataset)} 个样本")
    return test_loader

def load_mnist_test_data(batch_size: int = 128, data_dir: str = './data'):
    """
    加载MNIST测试数据，使用与训练时相同的数据预处理策略
    
    Args:
        batch_size: 批次大小
        data_dir: 数据目录
        
    Returns:
        MNIST测试数据加载器
    """
    return load_test_data('mnist', batch_size, data_dir)

def build_autoencoder_from_watermark(watermark_params, decoder_path: str, device: str = 'cuda'):
    """
    从水印参数构建自编码器
    
    Args:
        watermark_params: 从主模型中提取的水印参数
        decoder_path: 解码器权重文件路径
        device: 设备类型
        
    Returns:
        构建的自编码器模型
    """
    try:
        # 创建自编码器实例
        autoencoder = LightAutoencoder().to(device)
        
        # 从水印参数构建编码器
        # 这里需要根据实际的水印参数结构来映射到编码器层
        # 假设watermark_params是一个字典，包含编码器各层的参数
        if isinstance(watermark_params, dict):
            encoder_state_dict = {}
            for name, param in watermark_params.items():
                if 'encoder' in name:
                    encoder_state_dict[name] = param
            autoencoder.encoder.load_state_dict(encoder_state_dict)
        else:
            # 如果watermark_params是张量，需要重新整形并分配到编码器层
            print("⚠️  水印参数格式需要进一步处理")
            return None
        
        # 加载预训练的解码器
        if os.path.exists(decoder_path):
            decoder_state_dict = torch.load(decoder_path, map_location=device, weights_only=False)
            autoencoder.decoder.load_state_dict(decoder_state_dict)
            print(f"✓ 已加载解码器: {decoder_path}")
        else:
            print(f"❌ 解码器文件不存在: {decoder_path}")
            return None
        
        autoencoder.eval()
        return autoencoder
        
    except Exception as e:
        print(f"❌ 构建自编码器失败: {e}")
        return None



def load_main_task_model(model_path: str, device: str = 'cuda'):
    """
    加载主任务模型，自动从checkpoint推断数据集参数
    
    Args:
        model_path: 模型文件路径
        device: 设备类型
        
    Returns:
        tuple: (加载的模型, 模型信息字典)
    """
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    # 加载checkpoint获取参数信息
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint中获取数据集参数
    net_info = checkpoint.get('net_info', {})
    arguments = checkpoint.get('arguments', {})
    
    # 优先从arguments获取，否则使用默认值
    num_classes = arguments.get('num_classes', 14)
    in_channels = arguments.get('in_channels', 3)
    dataset = arguments.get('dataset', 'chestmnist')
    
    # 根据数据集设置input_size
    if dataset.lower() == 'cifar10' or dataset.lower() == 'cifar100':
        input_size = 32
    elif dataset.lower() == 'imagenet':
        input_size = 224
    else:  # chestmnist等
        input_size = 28
    
    # 从模型路径提取模型名称
    model_name = os.path.basename(model_path)
    if '.' in model_name:
        model_name = model_name.split('.')[0]  # 去掉扩展名
    
    # 构建模型信息字典
    model_info = {
        'dataset': dataset,
        'num_classes': num_classes,
        'in_channels': in_channels,
        'input_size': input_size,
        'model_name': model_name,
        'model_path': model_path,
        'model_filename': os.path.basename(model_path)
    }
    
    print(f"✓ 检测到数据集: {dataset}, 类别数: {num_classes}, 输入通道: {in_channels}, 输入尺寸: {input_size}")
    print(f"✓ 模型名称: {model_name}")
    
    # 创建模型实例
    model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['net_info']['best_model'][0])
    print(f"✓ 已加载主任务模型: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, model_info

def evaluate_model_accuracy(model, test_loader, device='cuda', dataset_type='chestmnist'):
    """
    评估模型在测试集上的准确率
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 设备类型
        dataset_type: 数据集类型，用于确定评估方式
        
    Returns:
        float: 模型准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            if dataset_type.lower() == 'chestmnist':
                # 多标签分类：使用sigmoid + 阈值0.5
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                # 计算样本级准确率（所有标签都正确才算正确）
                correct += (predicted == target).all(dim=1).sum().item()
            else:
                # 单标签分类：使用argmax
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
            
            total += target.size(0)
    
    accuracy = correct / total
    return accuracy

def evaluate_model_auc(model, test_loader, device='cuda', dataset_type='chestmnist'):
    """
    评估模型在测试集上的AUC
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 设备类型
        dataset_type: 数据集类型，用于确定评估方式
        
    Returns:
        float: 模型AUC（多标签任务取平均AUC）
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            if dataset_type.lower() == 'chestmnist':
                # 多标签分类：使用sigmoid输出概率
                predictions = torch.sigmoid(outputs).cpu().numpy()
                targets = target.cpu().numpy()
            else:
                # 单标签分类：使用softmax输出概率
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                targets = target.cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # 合并所有批次的结果
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets) if dataset_type.lower() == 'chestmnist' else np.hstack(all_targets)
    
    if dataset_type.lower() == 'chestmnist':
        # 多标签分类：计算每个标签的AUC然后取平均
        auc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
                auc_scores.append(auc)
            except ValueError:
                # 如果某个标签只有一种类别，跳过
                continue
        
        if auc_scores:
            avg_auc = np.mean(auc_scores)
        else:
            avg_auc = 0.0
    else:
        # 单标签分类：计算多类AUC
        try:
            avg_auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr', average='macro')
        except ValueError:
            avg_auc = 0.0
    
    return avg_auc

def threshold_pruning(model, pruning_ratio: float):
    """
    对模型进行阈值剪枝
    
    Args:
        model: 待剪枝的模型
        pruning_ratio: 剪枝比例 (0.0-1.0)
        
    Returns:
        剪枝后的模型
    """
    if pruning_ratio <= 0:
        return model
    
    # 创建模型副本
    pruned_model = copy.deepcopy(model)
    
    # 收集所有权重参数
    all_weights = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:  # 只对权重进行剪枝，不包括偏置
            all_weights.append(param.data.view(-1))
    
    if not all_weights:
        print("警告: 没有找到可剪枝的权重参数")
        return pruned_model
    
    # 合并所有权重
    all_weights = torch.cat(all_weights)
    
    # 计算剪枝阈值
    threshold = torch.quantile(torch.abs(all_weights), pruning_ratio)
    
    # 应用剪枝
    pruned_count = 0
    total_count = 0
    
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            # 创建掩码
            mask = torch.abs(param.data) > threshold
            pruned_count += (~mask).sum().item()
            total_count += param.data.numel()
            
            # 应用掩码
            param.data *= mask.float()
    
    print(f"剪枝比例: {pruning_ratio:.1%}, 阈值: {threshold:.6f}, 剪枝参数: {pruned_count}/{total_count}")
    
    return pruned_model

def evaluate_watermark_after_pruning(model, reconstructor):
    """
    评估剪枝后模型的水印完整性
    
    Args:
        model: 剪枝后的模型
        reconstructor: 水印重建器实例
        
    Returns:
        水印重建结果
    """
    try:
        
        # 获取模型状态字典
        model_state_dict = model.state_dict()
        
        # 从所有客户端重建自编码器
        reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(model_state_dict)
        
        if reconstructed_autoencoder is None:
            print("❌ 水印重建失败")
            return None
        
        # 计算水印参数统计
        key_manager = reconstructor.key_manager
        all_client_ids = key_manager.list_clients()
        
        total_watermark_params = 0
        damaged_watermark_params = 0
        
        for client_id in all_client_ids:
            try:
                # 提取水印参数并检查剪枝影响
                watermark_values = key_manager.extract_watermark(model_state_dict, client_id, check_pruning=True)
                total_watermark_params += len(watermark_values)
                
                # 检查被剪枝的水印参数（完全等于0的参数）
                damaged_count = (watermark_values == 0.0).sum().item()
                damaged_watermark_params += damaged_count
                
            except Exception as e:
                pass  # 静默处理错误
        
        # 计算水印完整性指标
        if total_watermark_params > 0:
            watermark_integrity = 1.0 - (damaged_watermark_params / total_watermark_params)
        else:
            watermark_integrity = 0.0
        
        return {
            'reconstructed_autoencoder': reconstructed_autoencoder,
            'watermark_integrity': watermark_integrity,
            'total_watermark_params': total_watermark_params,
            'damaged_watermark_params': damaged_watermark_params
        }
        
    except Exception as e:
        print(f"❌ 水印评估失败: {e}")
        return None

def save_pruning_results(results, model_info, save_dir='./save/pruning_results'):
    """
    保存剪枝攻击实验结果
    
    Args:
        results: 实验结果列表
        model_info: 模型信息字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 从模型信息中提取数据集和模型名称
    dataset = model_info.get('dataset', 'unknown')
    model_name = model_info.get('model_name', 'unknown')
    
    # 生成文件名
    filename_prefix = f'pruning_attack_{dataset}_{timestamp}'
    csv_file = os.path.join(save_dir, f'{filename_prefix}.csv')
    
    df_data = []
    for result in results:
        df_data.append({
            'pruning_ratio': result['pruning_ratio'],
            'auc': result.get('auc_after', 0.0),
            'accuracy': result.get('accuracy_after', 0.0),
            'watermark_integrity': result['watermark_integrity'],
            'total_watermark_params': result['total_watermark_params'],
            'damaged_watermark_params': result['damaged_watermark_params'],
            'delta_pcc': result.get('delta_pcc', float('inf')),
            'is_infringement': result.get('is_infringement', False),
            'result_text': result.get('result_text', '评估失败')
        })
    
    df = pd.DataFrame(df_data)
    
    # 格式化数值列，保留6位小数
    numeric_columns = ['pruning_ratio', 'auc', 'accuracy', 'watermark_integrity', 
                      'total_watermark_params', 'damaged_watermark_params', 'delta_pcc']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
    
    # 在CSV文件开头添加模型文件名信息
    model_filename = model_info.get('model_filename', 'unknown')
    
    # 先写入注释行，然后写入数据
    with open(csv_file, 'w', encoding='utf-8-sig') as f:
        f.write(f"# 模型文件: {model_filename}\n")
        f.write(f"# 数据集: {model_info.get('dataset', 'unknown')}\n")
        f.write(f"# 生成时间: {timestamp}\n")
        f.write("#\n")
    
    # 追加数据到文件
    df.to_csv(csv_file, mode='a', index=False, encoding='utf-8-sig')
    
    print(f"✓ 结果已保存到: {save_dir}")
    print(f"  - CSV文件: {csv_file}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='剪枝攻击实验')
    parser.add_argument('--model_path', type=str, 
                       default='./save/resnet/cifar10/202510231542_Dp_0.1_iid_True_lt_sign_ep_150_le_2_cn_5_fra_1.0000_acc_0.9319_enhanced.pkl',
                       help='模型文件路径')
    parser.add_argument('--key_matrix_dir', type=str, default='./save/key_matrix',
                       help='密钥矩阵基础目录')
    parser.add_argument('--autoencoder_dir', type=str, default='./save/autoencoder',
                       help='自编码器目录')
    args = parser.parse_args()
    
    # 从模型路径中提取模型类型和客户端数量
    model_info = extract_model_info_from_path(args.model_path)
    model_type = model_info.get('model_type', 'resnet')
    client_num = model_info.get('client_num', 10)
    
    # 生成密钥矩阵路径
    from utils.key_matrix_utils import get_key_matrix_path
    args.key_matrix_path = get_key_matrix_path(args.key_matrix_dir, model_type, client_num)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型路径
    model_path = args.model_path
    
    # 密钥矩阵目录
    key_matrix_dir = args.key_matrix_path
    autoencoder_dir = args.autoencoder_dir
    
    # 加载主任务模型
    model, model_info = load_main_task_model(model_path, device)
    
    # 根据模型信息加载相应的测试数据
    dataset_name = model_info.get('dataset', 'chestmnist')
    test_loader = load_test_data(dataset_name, batch_size=128)
    
    if model is not None and model_info is not None:
        print(f"开始剪枝攻击实验 (设备: {device})")
        
        # 初始化水印重建器（使用args中的统一设置）
        from utils.args import parser_args
        args = parser_args()
        reconstructor = WatermarkReconstructor(
            key_matrix_dir, 
            autoencoder_dir, 
            enable_scaling=args.enable_watermark_scaling, 
            scaling_factor=args.scaling_factor
        )
        
        # ==================== 水印检测容忍度设置 ====================
        PERF_FAIL_RATIO = 0.3
        # =========================================================
        print(f"水印检测容忍度设置: {PERF_FAIL_RATIO}")
        # 计算固定阈值τ（基于原始未剪枝模型）
        print("计算固定阈值τ...")
        fixed_tau = calculate_fixed_tau(model.state_dict(), reconstructor, test_loader, device, perf_fail_ratio=PERF_FAIL_RATIO)
        if fixed_tau is None:
            print("❌ 无法计算固定阈值，将使用动态阈值")
        
        # 定义剪枝比例：从0%到100%，步长10%
        pruning_ratios = [i/10.0 for i in range(0, 11)]  # [0.0, 0.1, 0.2, ..., 1.0]
        
        # 存储实验结果
        results = []
        
        # 首先评估原始模型的性能
        original_auc = evaluate_model_auc(model, test_loader, device, dataset_type=dataset_name)
        original_accuracy = evaluate_model_accuracy(model, test_loader, device, dataset_type=dataset_name)
        print(f"原始模型AUC: {original_auc:.4f}, 准确率: {original_accuracy:.4f}")
        
        for ratio in pruning_ratios:
            print(f"\n--- 剪枝比例: {ratio:.0%} ---")
            
            # 对模型进行剪枝
            pruned_model = threshold_pruning(model, ratio)
            
            # 评估剪枝后模型的AUC和准确率
            pruned_auc = evaluate_model_auc(pruned_model, test_loader, device, dataset_type=dataset_name)
            pruned_accuracy = evaluate_model_accuracy(pruned_model, test_loader, device, dataset_type=dataset_name)
            
            print(f"剪枝后模型AUC: {pruned_auc:.4f}, 准确率: {pruned_accuracy:.4f}")
            
            # 评估水印完整性
            watermark_result = evaluate_watermark_after_pruning(pruned_model, reconstructor)
            
            # 评估ΔPCC
            # 使用原始模型作为基准，比较剪枝前后的性能
            delta_pcc_result = evaluate_delta_pcc(
                model.state_dict(), pruned_model.state_dict(), reconstructor, test_loader, device, 
                perf_fail_ratio=PERF_FAIL_RATIO, fixed_tau=fixed_tau
            )
            
            # 记录结果
            result = {
                'pruning_ratio': ratio,
                'auc_before': original_auc,
                'auc_after': pruned_auc,
                'accuracy_before': original_accuracy,
                'accuracy_after': pruned_accuracy,
                'accuracy_drop': original_accuracy - pruned_accuracy
            }
            
            if watermark_result is not None:
                result.update({
                    'watermark_integrity': watermark_result['watermark_integrity'],
                    'total_watermark_params': watermark_result['total_watermark_params'],
                    'damaged_watermark_params': watermark_result['damaged_watermark_params']
                })
            else:
                result.update({
                    'watermark_integrity': 0.0,
                    'total_watermark_params': 0,
                    'damaged_watermark_params': 0
                })
            
            # 添加ΔPCC结果
            delta_pcc_default = {
                'perf_before': float('inf'),
                'perf_after': float('inf'),
                'perf_fail': float('inf'),
                'tau': float('inf'),
                'delta_perf': float('inf'),
                'delta_pcc': float('inf'),
                'is_infringement': False,
                'result_text': '评估失败'
            }
            result.update(format_delta_pcc_result(delta_pcc_result, delta_pcc_default))
            
            results.append(result)
            
            # 根据数据集类型调整输出格式
            if dataset_name == 'chestmnist':
                print(f"剪枝{ratio:.0%}: AUC{pruned_auc:.4f} [主要] | 准确率{pruned_accuracy:.4f} [参考] | 水印完整性{result['watermark_integrity']:.2%} | ΔPCC{result['delta_pcc']:.6f} | {result['result_text']}")
            else:
                print(f"剪枝{ratio:.0%}: AUC{pruned_auc:.4f} [参考] | 准确率{pruned_accuracy:.4f} [主要] | 水印完整性{result['watermark_integrity']:.2%} | ΔPCC{result['delta_pcc']:.6f} | {result['result_text']}")
        
        # 根据数据集类型调整总结表格格式
        print(f"\n{'='*80}")
        print("实验结果总结")
        print(f"{'='*80}")
        
        if dataset_name == 'chestmnist':
            print(f"{'剪枝%':<6} {'AUC[主要]':<10} {'准确率[参考]':<10} {'水印完整性%':<13} {'ΔPCC':<9} {'侵权判断':<8}")
            print("-" * 70)
            
            for result in results:
                print(f"{result['pruning_ratio']:>4.0%} "
                      f"{result['auc_after']:>8.4f} "
                      f"{result['accuracy_after']:>8.4f} "
                      f"{result['watermark_integrity']:>11.2%} {result['delta_pcc']:>9.6f} "
                      f"{result['result_text']:>6}")
        else:
            print(f"{'剪枝%':<6} {'AUC[参考]':<10} {'准确率[主要]':<10} {'水印完整性%':<13} {'ΔPCC':<9} {'侵权判断':<8}")
            print("-" * 70)
            
            for result in results:
                print(f"{result['pruning_ratio']:>4.0%} "
                      f"{result['auc_after']:>8.4f} "
                      f"{result['accuracy_after']:>8.4f} "
                      f"{result['watermark_integrity']:>11.2%} {result['delta_pcc']:>9.6f} "
                      f"{result['result_text']:>6}")
        
        # 保存实验结果
        print("\n保存实验结果...")
        save_pruning_results(results, model_info)
            
    else:
        print("主任务模型加载失败")

if __name__ == '__main__':
    main()
