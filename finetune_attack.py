#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调攻击实验代码
针对模型微调攻击的实验，继续对主任务进行训练，观察水印完整性、ΔPCC值变化以及判断是否侵权
"""

import copy
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Windows多进程兼容性处理
if sys.platform.startswith('win'):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from models.resnet import resnet18
from utils.dataset import LocalChestMNISTDataset
from utils.watermark_reconstruction import WatermarkReconstructor
from utils.delta_pcc_utils import evaluate_delta_pcc, calculate_fixed_tau, format_delta_pcc_result, print_delta_pcc_summary


def extract_model_info_from_path(model_path):
    """
    从模型路径中提取数据集和模型名信息
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        dict: 包含dataset和model_name的字典
    """
    try:
        # 标准化路径分隔符
        normalized_path = model_path.replace('\\', '/')
        
        # 分割路径
        path_parts = normalized_path.split('/')
        
        # 查找数据集和模型名
        dataset = 'unknown'
        model_name = 'unknown'
        
        # 从路径中提取信息
        for i, part in enumerate(path_parts):
            if part in ['cifar10', 'cifar100', 'chestmnist']:
                dataset = part
            elif part in ['resnet', 'alexnet', 'cnn', 'vgg', 'densenet']:
                model_name = part
            elif part == 'resnet18':
                model_name = 'resnet'
            elif part == 'cnn_simple':
                model_name = 'cnn'
        
        return {
            'dataset': dataset,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"警告: 无法从路径中提取模型信息: {e}")
        return {
            'dataset': 'unknown',
            'model_name': 'unknown'
        }


def create_safe_dataloader(dataset, batch_size, shuffle=False, num_workers=None):
    """
    创建安全的数据加载器，自动处理Windows多进程问题
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数，None时自动选择（Windows上默认0避免序列化问题）
        
    Returns:
        数据加载器
    """
    import sys
    
    # Windows上使用单进程模式避免序列化问题，Linux/Mac上可以使用多进程
    if num_workers is None:
        if sys.platform.startswith('win'):
            num_workers = 0  # Windows上禁用多进程，避免EOFError和序列化问题
        else:
            num_workers = 4  # Linux/Mac上使用4个进程
    
    # 如果指定了num_workers，直接使用
    if num_workers == 0:
        # 单进程模式，不需要pin_memory和persistent_workers
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
    else:
        # 多进程模式（仅用于Linux/Mac）
        try:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers,
                pin_memory=True,  # 启用内存固定，提高GPU传输效率
                persistent_workers=True,  # 保持工作进程，减少重启开销
                drop_last=False  # 保留最后一个不完整的batch
            )
        except Exception as e:
            print(f"⚠️  多进程数据加载失败，回退到单进程模式: {e}")
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )


def load_mnist_test_data(batch_size: int = 128, data_dir: str = './data'):
    """
    加载MNIST测试数据，用于自编码器性能评估

    Args:
        batch_size: 批次大小
        data_dir: 数据目录

    Returns:
        MNIST测试数据加载器
    """
    # 使用与train_autoencoder.py相同的数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST测试集
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = create_safe_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ 已加载MNIST测试集: {len(test_dataset)} 个样本")
    return test_loader


def load_chestmnist_data(data_root: str = './data'):
    """
    加载ChestMNIST数据集用于微调训练

    Args:
        data_root: 数据根目录

    Returns:
        训练和测试数据加载器
    """
    # ChestMNIST数据预处理
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 加载ChestMNIST数据集
    dataset_path = os.path.join(data_root, 'chestmnist.npz')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ChestMNIST数据集文件不存在: {dataset_path}")

    train_set = LocalChestMNISTDataset(dataset_path, split='train', transform=transform_train)
    test_set = LocalChestMNISTDataset(dataset_path, split='test', transform=transform_test)

    print(f"✓ 已加载ChestMNIST数据集 - 训练集: {len(train_set)}, 测试集: {len(test_set)}")

    return train_set, test_set


def load_cifar10_data(batch_size: int = 128, data_root: str = './data'):
    """
    加载CIFAR-10数据集用于微调训练

    Args:
        batch_size: 批次大小
        data_root: 数据根目录

    Returns:
        训练和测试数据加载器
    """
    # CIFAR-10数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = create_safe_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_safe_dataloader(test_dataset, batch_size=batch_size*2, shuffle=False)

    print(f"✓ 已加载CIFAR-10数据集: 训练集 {len(train_dataset)} 个样本, 测试集 {len(test_dataset)} 个样本")
    return train_loader, test_loader


def load_cifar100_data(batch_size: int = 128, data_root: str = './data'):
    """
    加载CIFAR-100数据集用于微调训练

    Args:
        batch_size: 批次大小
        data_root: 数据根目录

    Returns:
        训练和测试数据加载器
    """
    # CIFAR-100数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = create_safe_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_safe_dataloader(test_dataset, batch_size=batch_size*2, shuffle=False)

    print(f"✓ 已加载CIFAR-100数据集: 训练集 {len(train_dataset)} 个样本, 测试集 {len(test_dataset)} 个样本")
    return train_loader, test_loader


def load_main_task_model(model_path: str, device: str = 'cuda'):
    """
    加载主任务模型，自动从checkpoint推断数据集参数

    Args:
        model_path: 模型文件路径
        device: 设备类型

    Returns:
        加载的模型
    """
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载checkpoint获取参数信息
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint中获取数据集参数
    net_info = checkpoint.get('net_info', {})
    arguments = checkpoint.get('arguments', {})
    
    # 优先从arguments获取，否则使用默认值
    dataset = arguments.get('dataset', 'chestmnist')
    model_name = arguments.get('model_name', 'unknown')
    
    # 如果无法从arguments获取model_name，尝试从路径提取
    if model_name == 'unknown':
        model_info = extract_model_info_from_path(model_path)
        model_name = model_info.get('model_name', 'resnet')  # 默认使用resnet
    
    # 数据集预设配置（与 utils/args.py 保持一致）
    DATASET_PRESETS = {
        'chestmnist': {
            'num_classes': 14,
            'in_channels': 1,
            'input_size': 28,
        },
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
    
    # 根据数据集获取预设值
    ds_key = dataset.lower()
    preset = DATASET_PRESETS.get(ds_key, DATASET_PRESETS['chestmnist'])  # 默认使用chestmnist预设
    
    # 优先从arguments获取，如果为None则使用预设值
    num_classes = arguments.get('num_classes')
    if num_classes is None:
        num_classes = preset['num_classes']
    
    in_channels = arguments.get('in_channels')
    if in_channels is None:
        in_channels = preset['in_channels']
    
    input_size = arguments.get('input_size')
    if input_size is None:
        input_size = preset['input_size']
    
    print(f"✓ 检测到数据集: {dataset}, 类别数: {num_classes}, 输入通道: {in_channels}, 输入尺寸: {input_size}")
    print(f"✓ 模型类型: {model_name}")

    # 根据模型名称创建对应的模型实例
    if model_name in ['alexnet']:
        from models.alexnet import alexnet
        model = alexnet(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    elif model_name in ['resnet', 'resnet18']:
        model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    else:
        print(f"⚠️  未知模型类型: {model_name}，使用默认resnet18")
        model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=input_size)

    # 加载模型权重
    model.load_state_dict(checkpoint['net_info']['best_model'][0])
    print(f"✓ 已加载主任务模型: {model_path}")

    model = model.to(device)
    model.train()  # 设置为训练模式，因为要进行微调

    return model


def finetune_model(model, train_loader, test_loader, epochs: int, lr: float = 0.001,
                   device: str = 'cuda', eval_interval: int = 10, pcc_interval: int = 10,
                   reconstructor=None, original_model_state=None, mnist_test_loader=None, fixed_tau=None,
                   optimizer_type: str = 'adam', dataset_type: str = 'chestmnist'):
    """
    对模型进行微调训练（精简输出）
    
    Args:
        model: 要微调的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        eval_interval: 基本评估间隔（每轮显示训练/测试指标）
        pcc_interval: PCC计算间隔（每几轮计算一次ΔPCC和侵权检测）
        reconstructor: 水印重建器
        original_model_state: 原始模型状态
        mnist_test_loader: MNIST测试数据加载器
        fixed_tau: 固定阈值τ
    """
    from tqdm import tqdm
    # 根据optimizer_type参数选择优化器
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:  # 默认使用Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 根据数据集类型选择合适的损失函数
    if dataset_type == 'chestmnist':
        criterion = nn.BCEWithLogitsLoss()
    else:  # cifar10 等多分类任务
        criterion = nn.CrossEntropyLoss()
    
    model_states, performance_metrics = [], []
    print(f"开始微调训练，共 {epochs} 轮，每 {eval_interval} 轮评估一次，每 {pcc_interval} 轮计算PCC")
    
    # 初始化delta_pcc_result变量
    delta_pcc_result = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        try:
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                if dataset_type == 'chestmnist':
                    loss = criterion(model(data), target.float())
                else:  # 多分类任务
                    loss = criterion(model(data), target.long())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        except Exception as e:
            print(f"❌ 训练过程中发生错误: {e}")
            print("尝试重新创建数据加载器...")
            # 重新创建数据加载器
            train_loader = create_safe_dataloader(train_loader.dataset, train_loader.batch_size, shuffle=True, num_workers=0)
            continue

        avg_loss = total_loss / len(train_loader)

        # 每轮都进行基本评估（损失、AUC、准确率）
        model.eval()
        test_loss, all_predictions, all_targets = 0.0, [], []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"E{epoch+1}评估", leave=False):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if dataset_type == 'chestmnist':
                    test_loss += criterion(output, target.float()).item()
                    all_predictions.append(torch.sigmoid(output).cpu().numpy())
                else:  # 多分类任务
                    test_loss += criterion(output, target.long()).item()
                    all_predictions.append(torch.softmax(output, dim=1).cpu().numpy())
                
                all_targets.append(target.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 计算AUC和准确率（根据数据集类型）
        if dataset_type == 'chestmnist':
            # 多标签二分类任务
            try:
                from sklearn.metrics import roc_auc_score
                auc_scores = [
                    roc_auc_score(all_targets[:, i], all_predictions[:, i])
                    for i in range(all_targets.shape[1])
                    if len(np.unique(all_targets[:, i])) > 1
                ]
                mean_auc = np.mean(auc_scores) if auc_scores else 0.0
            except ImportError:
                mean_auc = 0.0
            
            # 计算准确率（多标签）
            pred_binary = (all_predictions > 0.5).astype(int)
            accuracy = np.mean((pred_binary == all_targets).astype(float))
        else:
            # 多分类任务
            try:
                from sklearn.metrics import roc_auc_score
                # 使用one-vs-rest策略计算多分类AUC
                mean_auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr', average='macro')
            except ImportError:
                mean_auc = 0.0
            
            # 计算准确率（多分类）
            pred_classes = np.argmax(all_predictions, axis=1)
            accuracy = np.mean((pred_classes == all_targets).astype(float))

        # 打印基本指标（每轮都显示）
        print(f"\n=== 第 {epoch+1} 轮评估 ===")
        if dataset_type == 'chestmnist':
            print(f"训练损失: {avg_loss:.4f} | 测试损失: {avg_test_loss:.4f} | "
                  f"AUC: {mean_auc:.4f} [主要] | 准确率: {accuracy:.2%} [参考]")
        else:
            print(f"训练损失: {avg_loss:.4f} | 测试损失: {avg_test_loss:.4f} | "
                  f"AUC: {mean_auc:.4f} [参考] | 准确率: {accuracy:.2%} [主要]")

        # 根据pcc_interval参数计算ΔPCC和侵权检测（计算量大的操作）
        delta_pcc_result = None
        if ((epoch + 1) == 1) or ((epoch + 1) % pcc_interval == 0):
            print("🔍 进行ΔPCC和侵权检测评估...")
            # 保存状态
            model_states.append(copy.deepcopy(model.state_dict()))
            
            if reconstructor and original_model_state and mnist_test_loader:
                # 使用torch.no_grad()减少内存使用
                with torch.no_grad():
                    delta_pcc_result = evaluate_delta_pcc(
                        original_model_state, model_states[-1], reconstructor,
                        mnist_test_loader, device, perf_fail_ratio=0.1, fixed_tau=fixed_tau, model=model
                    )
            
            # 打印ΔPCC结果
            print_delta_pcc_summary(delta_pcc_result)
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 保存性能指标（每轮都保存）
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'test_loss': avg_test_loss,
            'test_auc': mean_auc,
            'test_accuracy': accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # 添加ΔPCC和侵权判断信息（每10轮更新）
        if delta_pcc_result:
            metrics.update({
                'perf_before': delta_pcc_result['perf_before'],
                'perf_fail': delta_pcc_result['perf_fail'],
                'tau': delta_pcc_result['tau'],
                'delta_perf': delta_pcc_result['delta_perf'],
                'delta_pcc': delta_pcc_result['delta_pcc'],
                'is_infringement': delta_pcc_result['is_infringement'],
                'result_text': delta_pcc_result['result_text']
            })
        else:
            # 如果没有ΔPCC结果，填充默认值
            metrics.update({
                'perf_before': None,
                'perf_fail': None,
                'tau': None,
                'delta_perf': None,
                'delta_pcc': None,
                'is_infringement': None,
                'result_text': 'N/A'
            })
        
        performance_metrics.append(metrics)
        
        # 更新metrics中的ΔPCC信息
        current_metrics = performance_metrics[-1]
        current_metrics.update(format_delta_pcc_result(delta_pcc_result))
        
        # 清理delta_pcc_result内存
        if delta_pcc_result:
            del delta_pcc_result
            delta_pcc_result = None

        print("-" * 50)
        
        # 清理基本评估的临时变量
        del all_predictions, all_targets

    return model_states, performance_metrics


def evaluate_watermark_integrity(model_state_dict, reconstructor, model=None):
    """
    评估水印完整性

    Args:
        model_state_dict: 模型状态字典
        reconstructor: 水印重建器
        model: 模型对象（可选），用于确保参数顺序一致性

    Returns:
        水印完整性评估结果
    """
    try:
        # 从模型状态字典重建自编码器
        reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(model_state_dict)

        if reconstructed_autoencoder is None:
            return {
                'watermark_integrity': 0.0,
                'reconstruction_success': False,
                'total_watermark_params': 0,
                'damaged_watermark_params': 0
            }

        # 计算水印参数统计
        key_manager = reconstructor.key_manager
        all_client_ids = key_manager.list_clients()

        total_watermark_params = 0
        damaged_watermark_params = 0

        for cid in all_client_ids:
            try:
                # 提取水印参数（传入模型对象以确保参数顺序一致）
                watermark_values = key_manager.extract_watermark(model_state_dict, cid, check_pruning=True, model=model)
                total_watermark_params += len(watermark_values)

                # 检查被破坏的水印参数（完全等于0的参数）
                damaged_count = (watermark_values == 0.0).sum().item()
                damaged_watermark_params += damaged_count

            except Exception as e:
                pass  # 静默处理错误

        # 计算水印完整性
        if total_watermark_params > 0:
            watermark_integrity = 1.0 - (damaged_watermark_params / total_watermark_params)
        else:
            watermark_integrity = 0.0

        return {
            'watermark_integrity': watermark_integrity,
            'reconstruction_success': True,
            'total_watermark_params': total_watermark_params,
            'damaged_watermark_params': damaged_watermark_params
        }

    except Exception as e:
        print(f"❌ 水印完整性评估失败: {e}")
        return {
            'watermark_integrity': 0.0,
            'reconstruction_success': False,
            'total_watermark_params': 0,
            'damaged_watermark_params': 0
        }



def save_results(results, model_path: str, save_dir: str = './save/finetune_attack'):
    """
    保存实验结果

    Args:
        results: 实验结果
        model_path: 微调对象模型路径
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 从模型路径提取原始模型的日期前缀
    # 支持多种文件名格式：
    # 1. YYYYMMDDHHMMSS_...  （14位数字开头）
    # 2. model_YYYYMMDDHHMMSS_... （中间包含日期）
    # 3. 其他格式（返回 unknown）
    model_filename = os.path.basename(model_path)
    import re
    
    # 尝试多种匹配模式
    model_date_prefix = 'unknown'
    
    # 模式1: 文件名以14位数字开头
    match = re.match(r'^(\d{14})_', model_filename)
    if match:
        model_date_prefix = match.group(1)
    else:
        # 模式2: 尝试在路径中找到日期（如 .../YYYYMMDDHHMMSS_modelname/...）
        match = re.search(r'/(\d{14})_', model_path)
        if match:
            model_date_prefix = match.group(1)
        else:
            # 模式3: 文件名中任何位置的14位数字
            match = re.search(r'(\d{14})', model_filename)
            if match:
                model_date_prefix = match.group(1)
    
    # 文件名前缀：finetune_attack_实验时间戳_原始模型日期
    filename_prefix = f'finetune_attack_{timestamp}_{model_date_prefix}'
    
    # 保存详细结果
    results_file = os.path.join(save_dir, f'{filename_prefix}.pkl')
    torch.save(results, results_file)

    # 保存CSV格式的简化结果
    csv_file = os.path.join(save_dir, f'{filename_prefix}.csv')

    import pandas as pd
    df_data = []
    for result in results:
        df_data.append({
            'epoch': result['epoch'],
            'train_loss': result['train_loss'],
            'test_loss': result['test_loss'],
            'test_auc': result['test_auc'],
            'test_accuracy': result['test_accuracy'],
            'learning_rate': result['learning_rate'],
            'perf_fail': result.get('perf_fail', None),
            'tau': result.get('tau', None),
            'delta_perf': result.get('delta_perf', None),
            'delta_pcc': result.get('delta_pcc', None),
            'is_infringement': result.get('is_infringement', None),
            'result_text': result.get('result_text', 'N/A')
        })

    df = pd.DataFrame(df_data)
    
    # 格式化数值列：默认保留6位小数，ΔPCC保留8位小数（便于精细对比）
    numeric_columns = ['train_loss', 'test_loss', 'test_auc', 'test_accuracy', 'learning_rate', 
                      'perf_fail', 'tau', 'delta_perf', 'delta_pcc']
    precision_map = {
        'delta_pcc': 8,
    }
    for col in numeric_columns:
        if col in df.columns:
            prec = precision_map.get(col, 6)
            df[col] = df[col].apply(
                lambda x, p=prec: f"{float(x):.{p}f}"
                if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x
            )
    
    # 在CSV文件最后一行添加PKL文件名信息（保存原始模型路径）
    pkl_filename = os.path.basename(model_path)
    df.loc[len(df)] = ['PKL_FILE'] + [''] * (len(df.columns) - 2) + [pkl_filename]
    
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"✓ 结果已保存到: {save_dir}")
    print(f"  - 详细结果: {results_file}")
    print(f"  - 汇总结果: {csv_file}")

    # 保存CSV时额外输出最高ΔPCC（8位小数），便于快速核对
    try:
        delta_pcc_values = [
            float(r.get('delta_pcc'))
            for r in results
            if r.get('delta_pcc') is not None
        ]
        if delta_pcc_values:
            max_delta_pcc = max(delta_pcc_values)
            print(f"  - 最高ΔPCC: {max_delta_pcc:.8f}")
    except Exception:
        pass


def main():
    """主函数"""
    import argparse
    
    # 首先加载args.py中的参数配置
    from utils.args import parser_args
    base_args = parser_args()
    
    # 解析微调攻击特定的命令行参数
    parser = argparse.ArgumentParser(description='微调攻击实验')
    parser.add_argument('--model_path', type=str, 
                       default='./save/alexnet/chestmnist/202604071813_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_auc_0.7286_enhanced.pkl',
                       help='模型文件路径')
    parser.add_argument('--model_type', type=str, default='alexnet',
                       choices=['resnet', 'alexnet'],
                       help='模型类型')
    parser.add_argument('--client_num', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--dataset', type=str, default='chestmnist',
                       choices=['cifar10', 'cifar100', 'chestmnist'],
                       help='数据集类型')
    parser.add_argument('--key_matrix_dir', type=str, default='./save/key_matrix',
                       help='密钥矩阵基础目录')
    parser.add_argument('--autoencoder_dir', type=str, default='./save/autoencoder',
                       help='自编码器目录')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                       help='优化器类型（默认使用args.py中的optim）')
    parser.add_argument('--finetune_epochs', type=int, default=50,
                       help='微调轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率（默认使用args.py中的lr）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（默认使用args.py中的batch_size）')

    # 解析命令行参数
    cmd_args = parser.parse_args()
    
    # 合并参数：命令行参数优先，否则使用args.py中的参数
    args = argparse.Namespace()
    args.model_path = cmd_args.model_path
    args.model_type = cmd_args.model_type
    args.client_num = cmd_args.client_num
    args.key_matrix_dir = cmd_args.key_matrix_dir
    args.autoencoder_dir = cmd_args.autoencoder_dir
    args.finetune_epochs = cmd_args.finetune_epochs
    args.learning_rate = cmd_args.learning_rate if cmd_args.learning_rate is not None else base_args.lr
    args.batch_size = cmd_args.batch_size if cmd_args.batch_size is not None else base_args.batch_size
    args.optimizer = cmd_args.optimizer if cmd_args.optimizer is not None else base_args.optim
    args.dataset = cmd_args.dataset
    
    # 使用key_matrix_utils生成正确的密钥矩阵路径
    from utils.key_matrix_utils import get_key_matrix_path
    
    # 从模型路径自动推断正确的模型类型
    model_info = extract_model_info_from_path(args.model_path)
    inferred_model_type = model_info.get('model_name', cmd_args.model_type)
    
    print(f"🔍 从模型路径推断的模型类型: {inferred_model_type}")
    print(f"   原指定的模型类型: {cmd_args.model_type}")
    
    # 使用推断的模型类型
    args.key_matrix_path = get_key_matrix_path(cmd_args.key_matrix_dir, inferred_model_type, cmd_args.client_num)
    
    # 从args.py获取其他必要参数
    args.data_root = base_args.data_root
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 配置参数
    model_path = args.model_path
    key_matrix_dir = args.key_matrix_path  # 使用生成的完整路径
    autoencoder_dir = args.autoencoder_dir
    
    # 微调参数
    finetune_epochs = args.finetune_epochs
    eval_interval = 1  # 每轮都进行基本评估
    pcc_interval = 5  # PCC计算间隔，可以调整（建议5-20之间，值越大计算越少但监控越粗糙）
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    optimizer_type = args.optimizer

    print(f"微调攻击实验参数:")
    print(f"  - 微调轮数: {finetune_epochs}")
    print(f"  - 基本评估: 每轮 | ΔPCC评估: 每{pcc_interval}轮")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 优化器: {optimizer_type}")
    print(f"  - 模型类型: {args.model_type}")
    print(f"  - 客户端数量: {args.client_num}")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 密钥矩阵路径: {key_matrix_dir}")
    print(f"  - 自编码器路径: {autoencoder_dir}")
    print("-" * 60)

    # 加载数据
    print("加载数据...")
    # 使用命令行参数指定的数据集
    dataset = args.dataset
    dataset_type = dataset  # 用于损失函数选择
    
    # 根据数据集类型加载相应的数据
    if dataset == 'cifar10':
        train_loader, test_loader = load_cifar10_data(batch_size=batch_size, data_root=args.data_root)
        mnist_test_loader = load_mnist_test_data(batch_size=128, data_dir=args.data_root)
    elif dataset == 'cifar100':
        print("使用CIFAR-100数据集进行微调攻击实验")
        train_loader, test_loader = load_cifar100_data(batch_size=batch_size, data_root=args.data_root)
        dataset_type = 'cifar100'
        mnist_test_loader = load_mnist_test_data(batch_size=128, data_dir=args.data_root)
    elif dataset == 'chestmnist':
        train_set, test_set = load_chestmnist_data(data_root=args.data_root)
        train_loader = create_safe_dataloader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = create_safe_dataloader(test_set, batch_size=batch_size, shuffle=False)
        mnist_test_loader = load_mnist_test_data(batch_size=128, data_dir=args.data_root)
    else:
        print(f"❌ 不支持的数据集: {dataset}")
        return

    # 加载主任务模型
    print("加载主任务模型...")
    model = load_main_task_model(model_path, device)
    if model is None:
        print("❌ 主任务模型加载失败")
        return

    # 保存原始模型状态
    original_model_state = copy.deepcopy(model.state_dict())

    # 初始化水印重建器（使用args中的统一设置）
    reconstructor = WatermarkReconstructor(
        key_matrix_dir, 
        autoencoder_dir
    )
    
    # 预计算固定阈值τ，避免重复计算
    print("预计算固定阈值τ...")
    fixed_tau = None
    if reconstructor and original_model_state and mnist_test_loader:
        fixed_tau = calculate_fixed_tau(original_model_state, reconstructor, mnist_test_loader, device, perf_fail_ratio=0.05, model=model)
        if fixed_tau is None:
            print("❌ 无法计算固定阈值，将使用动态阈值")
        else:
            print(f"✓ 固定阈值τ={fixed_tau:.6f}")

    print("开始微调攻击实验...")
    print("=" * 80)
    
    
    # 第0轮：测试微调前的水印检测
    print("=== 第0轮评估（微调前）===")
    
    # 先进行AUC评估
    model.eval()
    test_loss, all_predictions, all_targets = 0.0, [], []
    
    # 根据数据集类型选择合适的损失函数
    if dataset_type == 'chestmnist':
        criterion = nn.BCEWithLogitsLoss()
        activation_fn = torch.sigmoid
    else:  # cifar10, cifar100, mnist等多分类任务
        criterion = nn.CrossEntropyLoss()
        activation_fn = torch.softmax
    
    # 添加进度提示
    from tqdm import tqdm
    print("正在评估模型性能...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="评估中", leave=False)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if dataset_type == 'chestmnist':
                test_loss += criterion(output, target.float()).item()
                all_predictions.append(activation_fn(output).cpu().numpy())
            else:  # 多分类任务
                test_loss += criterion(output, target.long()).item()
                all_predictions.append(activation_fn(output, dim=1).cpu().numpy())
            
            all_targets.append(target.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算AUC和准确率（根据数据集类型）
    if dataset_type == 'chestmnist':
        # 多标签二分类任务
        try:
            from sklearn.metrics import roc_auc_score
            auc_scores = [
                roc_auc_score(all_targets[:, i], all_predictions[:, i])
                for i in range(all_targets.shape[1])
                if len(np.unique(all_targets[:, i])) > 1
            ]
            mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        except ImportError:
            mean_auc = 0.0
        
        # 计算准确率（多标签）
        pred_binary = (all_predictions > 0.5).astype(int)
        accuracy = np.mean((pred_binary == all_targets).astype(float))
    else:
        # 多分类任务
        try:
            from sklearn.metrics import roc_auc_score
            # 使用one-vs-rest策略计算多分类AUC
            mean_auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr', average='macro')
        except ImportError:
            mean_auc = 0.0
        
        # 计算准确率（多分类）
        pred_classes = np.argmax(all_predictions, axis=1)
        accuracy = np.mean((pred_classes == all_targets).astype(float))
    
    print(f"测试损失: {avg_test_loss:.4f} | AUC: {mean_auc:.4f} | 准确率: {accuracy:.2%}")
    
    # 根据数据集类型显示指标重要性
    if dataset_type == 'chestmnist':
        print(f"📊 ChestMNIST多标签任务 - AUC为主要指标，准确率为参考指标")
    else:
        print(f"📊 {dataset_type.upper()}多分类任务 - 准确率为主要指标，AUC为参考指标")
    
    # ==================== 水印检测容忍度设置 ====================
    PERF_FAIL_RATIO = 0.3
    # =========================================================
    print(f"水印检测容忍度设置: {PERF_FAIL_RATIO}")
    
    # 进行ΔPCC评估
    delta_pcc_result_0 = evaluate_delta_pcc(
        original_model_state, original_model_state, reconstructor,
        mnist_test_loader, device, perf_fail_ratio=PERF_FAIL_RATIO, fixed_tau=fixed_tau
    )
    
    # 创建第0轮的结果记录
    initial_result = {
        'epoch': 0,
        'train_loss': 0.0,  # 第0轮没有训练
        'test_loss': avg_test_loss,   # 第0轮测试损失
        'test_auc': mean_auc,    # 第0轮测试AUC
        'test_accuracy': accuracy,  # 第0轮测试准确率
        'learning_rate': 0.0,  # 第0轮没有学习率
    }
    
    # 添加第0轮的ΔPCC和侵权判断信息
    initial_result.update(format_delta_pcc_result(delta_pcc_result_0))

    # 进行微调训练
    model_states, performance_metrics = finetune_model(
        model, train_loader, test_loader,
        epochs=finetune_epochs, lr=learning_rate,
        device=device, eval_interval=eval_interval, pcc_interval=pcc_interval,
        reconstructor=reconstructor, original_model_state=original_model_state,
        mnist_test_loader=mnist_test_loader, fixed_tau=fixed_tau,
        optimizer_type=optimizer_type, dataset_type=dataset_type
    )

    # 微调训练已完成，ΔPCC和侵权判断已在训练过程中评估
    print("\n" + "=" * 80)
    print("微调攻击实验完成")
    print("=" * 80)

    # 将第0轮结果和训练结果合并
    results = [initial_result] + performance_metrics

    # 保存结果
    print("\n" + "=" * 80)
    print("保存实验结果...")
    save_results(results, args.model_path, save_dir='./save/finetune_attack')

    # 输出总结
    print("\n" + "=" * 80)
    print("微调攻击实验总结")
    print("=" * 80)
    
    # 根据数据集类型调整显示格式
    if dataset_type == 'chestmnist':
        print(f"{'轮次':<4} {'训练损失':<10} {'测试损失':<10} {'测试AUC':<10} {'测试准确率%':<8} {'ΔPCC':<8} {'侵权判断':<8}")
        print("-" * 80)
        
        for result in results:
            delta_pcc_str = f"{result['delta_pcc']:.8f}" if result['delta_pcc'] is not None else "N/A"
            infringement_str = "是" if result['is_infringement'] else "否" if result['is_infringement'] is not None else "N/A"
            
            print(f"{result['epoch']:>3}  "
                  f"{result['train_loss']:>8.4f}  "
                  f"{result['test_loss']:>8.4f}  "
                  f"{result['test_auc']:>8.4f}  "  # AUC更宽显示
                  f"{result['test_accuracy']:>6.2%}  "  # 准确率稍窄
                  f"{delta_pcc_str:>10}  "
                  f"{infringement_str:>6}")
    else:
        # CIFAR10等多分类任务
        print(f"{'轮次':<4} {'训练损失':<10} {'测试损失':<10} {'测试AUC':<8} {'测试准确率%':<10} {'ΔPCC':<8} {'侵权判断':<8}")
        print("-" * 80)
        
        for result in results:
            delta_pcc_str = f"{result['delta_pcc']:.8f}" if result['delta_pcc'] is not None else "N/A"
            infringement_str = "是" if result['is_infringement'] else "否" if result['is_infringement'] is not None else "N/A"
            
            print(f"{result['epoch']:>3}  "
                  f"{result['train_loss']:>8.4f}  "
                  f"{result['test_loss']:>8.4f}  "
                  f"{result['test_auc']:>6.4f}  "
                  f"{result['test_accuracy']:>8.2%}  "
                  f"{delta_pcc_str:>10}  "
                  f"{infringement_str:>6}")

    # 分析趋势
    print("\n趋势分析:")
    if len(results) > 1:
        initial_auc = results[0]['test_auc']
        final_auc = results[-1]['test_auc']
        auc_change = final_auc - initial_auc

        initial_acc = results[0]['test_accuracy']
        final_acc = results[-1]['test_accuracy']
        acc_change = final_acc - initial_acc

        if dataset_type == 'chestmnist':
            print(f"测试AUC变化: {initial_auc:.4f} → {final_auc:.4f} (变化: {auc_change:+.4f}) [主要指标]")
            print(f"测试准确率变化: {initial_acc:.2%} → {final_acc:.2%} (变化: {acc_change:+.2%}) [参考指标]")
        else:
            print(f"测试AUC变化: {initial_auc:.4f} → {final_auc:.4f} (变化: {auc_change:+.4f}) [参考指标]")
            print(f"测试准确率变化: {initial_acc:.2%} → {final_acc:.2%} (变化: {acc_change:+.2%}) [主要指标]")
        
        # 分析ΔPCC趋势
        delta_pcc_values = [r['delta_pcc'] for r in results if r['delta_pcc'] is not None]
        if len(delta_pcc_values) > 1:
            initial_delta_pcc = delta_pcc_values[0]
            final_delta_pcc = delta_pcc_values[-1]
            delta_pcc_change = final_delta_pcc - initial_delta_pcc
            print(f"ΔPCC变化: {initial_delta_pcc:.8f} → {final_delta_pcc:.8f} (变化: {delta_pcc_change:+.8f})")
        
        # 分析侵权判断
        infringement_count = sum(1 for r in results if r['is_infringement'] is True)
        total_evaluations = sum(1 for r in results if r['is_infringement'] is not None)
        if total_evaluations > 0:
            infringement_rate = infringement_count / total_evaluations
            print(f"侵权判断: {infringement_count}/{total_evaluations} 轮被判定为侵权 ({infringement_rate:.1%})")

    print("\n微调攻击实验完成！")


if __name__ == '__main__':
    main()
