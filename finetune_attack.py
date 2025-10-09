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
from tqdm import tqdm

# Windows多进程兼容性处理
if sys.platform.startswith('win'):
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from models.resnet import resnet18
from utils.dataset import LocalChestMNISTDataset
from utils.watermark_reconstruction import WatermarkReconstructor


def create_safe_dataloader(dataset, batch_size, shuffle=False, num_workers=None):
    """
    创建安全的数据加载器，自动处理Windows多进程问题
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数，None时自动选择
        
    Returns:
        数据加载器
    """
    import sys
    
    # Windows系统自动设置num_workers=0避免多进程问题
    if num_workers is None:
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 2
    
    try:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    except Exception as e:
        print(f"⚠️  多进程数据加载失败，回退到单进程模式: {e}")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


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


def test_autoencoder_mse(autoencoder, test_loader, device: str = 'cuda'):
    """
    测试自编码器在测试集上的MSE loss（统一使用剪枝攻击的计算方式）
    
    Args:
        autoencoder: 自编码器模型
        test_loader: 测试数据加载器
        device: 设备类型
        
    Returns:
        MSE loss值
    """
    if autoencoder is None:
        return float('inf')
    
    autoencoder.eval()
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 前向传播
            reconstructed = autoencoder(data)
            
            # 计算MSE loss (使用sum reduction，然后手动平均)
            mse_loss = torch.nn.functional.mse_loss(reconstructed, data, reduction='sum')
            total_mse += mse_loss.item()
            total_samples += data.size(0)  # 累加样本数
    
    avg_mse = total_mse / total_samples
    return avg_mse


def evaluate_encoder_decoder_from_all_clients(reconstructor, model_state_dict, test_loader, device: str = 'cuda'):
    """
    从所有客户端的水印参数重建自编码器并评估性能（统一使用剪枝攻击的MSE计算方式）
    
    Args:
        reconstructor: 水印重建器
        model_state_dict: 模型状态字典
        test_loader: 测试数据加载器
        device: 设备类型
        
    Returns:
        评估结果字典
    """
    try:
        # 使用所有客户端的密钥矩阵重建自编码器
        reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(model_state_dict)
        
        if reconstructed_autoencoder is None:
            return {
                'mse': float('inf'),
                'ssim': 0.0,
                'psnr': 0.0,
                'reconstruction_success': False,
                'watermark_damaged': False
            }

        # 使用统一的MSE计算方式
        mse = test_autoencoder_mse(reconstructed_autoencoder, test_loader, device)
        
        # 计算其他指标（如果需要的话）
        try:
            metrics = reconstructor.evaluate_autoencoder_performance(reconstructed_autoencoder, test_loader)
            ssim = metrics.get('ssim', 0.0)
            psnr = metrics.get('psnr', 0.0)
        except:
            ssim = 0.0
            psnr = 0.0
        
        return {
            'mse': mse,
            'ssim': ssim,
            'psnr': psnr,
            'reconstruction_success': True,
            'watermark_damaged': False
        }
        
    except Exception as e:
        return {
            'mse': float('inf'),
            'ssim': 0.0,
            'psnr': 0.0,
            'reconstruction_success': False,
            'watermark_damaged': False
        }


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


def load_main_task_model(model_path: str, device: str = 'cuda'):
    """
    加载主任务模型（ResNet18 for ChestMNIST）

    Args:
        model_path: 模型文件路径
        device: 设备类型

    Returns:
        加载的模型
    """
    # 创建模型实例
    model = resnet18(num_classes=14, in_channels=3, input_size=28)

    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['net_info']['best_model'][0])
        print(f"✓ 已加载主任务模型: {model_path}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    model = model.to(device)
    model.train()  # 设置为训练模式，因为要进行微调

    return model


def finetune_model(model, train_loader, test_loader, epochs: int, lr: float = 0.001,
                   device: str = 'cuda', eval_interval: int = 10, reconstructor=None,
                   original_model_state=None, mnist_test_loader=None):
    """
    对模型进行微调训练（精简输出）
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    # 确保step_size至少为1，避免除零错误
    step_size = max(1, epochs//3) if epochs > 0 else 1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    model_states, performance_metrics = [], []
    print(f"开始微调训练，共 {epochs} 轮，每 {eval_interval} 轮评估一次")
    
    # 初始化delta_pcc_result变量
    delta_pcc_result = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        try:
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target.float())
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
        scheduler.step()

        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            # 保存状态
            model_states.append(copy.deepcopy(model.state_dict()))

            # 评估
            model.eval()
            test_loss, all_predictions, all_targets = 0.0, [], []
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target.float()).item()
                    all_predictions.append(torch.sigmoid(output).cpu().numpy())
                    all_targets.append(target.cpu().numpy())

            avg_test_loss = test_loss / len(test_loader)
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

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

            pred_binary = (all_predictions > 0.5).astype(int)
            accuracy = np.mean((pred_binary == all_targets).astype(float))

            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'test_auc': mean_auc,
                'test_accuracy': accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # 添加ΔPCC和侵权判断信息
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

            # ΔPCC
            delta_pcc_result = None
            if reconstructor and original_model_state and mnist_test_loader:
                delta_pcc_result = evaluate_delta_pcc(
                    original_model_state, model_states[-1], reconstructor,
                    mnist_test_loader, device, perf_fail_ratio=0.5
                )

            # 保存第1轮和第2轮的ΔPCC结果到performance_metrics中
            if delta_pcc_result:
                # 更新当前轮次的metrics
                current_metrics = performance_metrics[-1]  # 获取刚添加的metrics
                current_metrics.update({
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
                current_metrics = performance_metrics[-1]  # 获取刚添加的metrics
                current_metrics.update({
                    'perf_before': None,
                    'perf_fail': None,
                    'tau': None,
                    'delta_perf': None,
                    'delta_pcc': None,
                    'is_infringement': None,
                    'result_text': 'N/A'
                })

            # 打印核心指标
            print(f"\n=== 第 {epoch+1} 轮评估 ===")
            print(f"训练损失: {avg_loss:.4f} | 测试损失: {avg_test_loss:.4f} | "
                  f"AUC: {mean_auc:.4f} | 准确率: {accuracy:.2%}")
            if delta_pcc_result:
                print(f"性能基准: {delta_pcc_result['perf_before']:.6f} | 性能变化: {delta_pcc_result['delta_perf']:.6f}")
                print(f"ΔPCC: {delta_pcc_result['delta_pcc']:.6f} | 侵权判断: {delta_pcc_result['result_text']}")
            else:
                print("ΔPCC: N/A | 侵权判断: N/A")
            print("-" * 50)

    return model_states, performance_metrics


def evaluate_watermark_integrity(model_state_dict, reconstructor):
    """
    评估水印完整性

    Args:
        model_state_dict: 模型状态字典
        reconstructor: 水印重建器

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
                # 提取水印参数
                watermark_values = key_manager.extract_watermark(model_state_dict, cid, check_pruning=True)
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


def evaluate_delta_pcc(original_model_state, current_model_state, reconstructor, 
                      mnist_test_loader, device: str = 'cuda', perf_fail_ratio: float = 0.1):
    """
    评估ΔPCC（精简输出）
    """
    try:
        # 评估原始模型
        original_result = evaluate_encoder_decoder_from_all_clients(
            reconstructor, original_model_state, mnist_test_loader, device
        )
        if not original_result['reconstruction_success']:
            return None
        perf_before = original_result['mse']
        
        perf_fail = perf_before * (1 + perf_fail_ratio)
        tau = perf_fail - perf_before

        # 评估当前模型
        current_result = evaluate_encoder_decoder_from_all_clients(
            reconstructor, current_model_state, mnist_test_loader, device
        )
        if not current_result['reconstruction_success']:
            return None
        perf_after = current_result['mse']

        delta_perf = abs(perf_after - perf_before)
        delta_pcc = delta_perf / tau if tau > 0 else float('inf')

        is_infringement = delta_pcc < 1.0
        result_text = "侵权" if is_infringement else "不侵权"

        return {
            'perf_before': perf_before,
            'perf_after': perf_after,
            'perf_fail': perf_fail,
            'tau': tau,
            'delta_perf': delta_perf,
            'delta_pcc': delta_pcc,
            'is_infringement': is_infringement,
            'result_text': result_text
        }
    except Exception as e:
        return None

def test_autoencoder_mse(autoencoder, test_loader, device: str = 'cuda'):
    """
    测试自编码器在测试集上的MSE loss

    Args:
        autoencoder: 自编码器模型
        test_loader: 测试数据加载器
        device: 设备类型

    Returns:
        MSE loss值
    """
    if autoencoder is None:
        return float('inf')

    autoencoder.eval()
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # 前向传播
            reconstructed = autoencoder(data)

            # 计算MSE loss
            mse_loss = torch.nn.functional.mse_loss(reconstructed, data, reduction='sum')
            total_mse += mse_loss.item()
            total_samples += data.size(0)

    avg_mse = total_mse / total_samples
    return avg_mse


def save_results(results, save_dir: str = './save/finetune_attack'):
    """
    保存实验结果

    Args:
        results: 实验结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(save_dir, f'finetune_attack_results_{timestamp}.pkl')
    torch.save(results, results_file)

    # 保存CSV格式的简化结果
    csv_file = os.path.join(save_dir, f'finetune_attack_summary_{timestamp}.csv')

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
            'perf_before': result.get('perf_before', None),
            'perf_fail': result.get('perf_fail', None),
            'tau': result.get('tau', None),
            'delta_perf': result.get('delta_perf', None),
            'delta_pcc': result.get('delta_pcc', None),
            'is_infringement': result.get('is_infringement', None),
            'result_text': result.get('result_text', 'N/A')
        })

    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"✓ 结果已保存到: {save_dir}")
    print(f"  - 详细结果: {results_file}")
    print(f"  - 汇总结果: {csv_file}")


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 配置参数
    model_path = './save/resnet/chestmnist/202510091431_Dp_0.1_iid_True_ns_1_wt_gamma_lt_sign_ep_10_le_2_cn_10_fra_1.0000_auc_0.5964_enhanced.pkl'
    key_matrix_dir = './save/key_matrix'
    autoencoder_dir = './save/autoencoder'

    # 微调参数
    finetune_epochs = 10
    eval_interval = 1  # 每轮都评估
    learning_rate = 0.001
    batch_size = 128

    print(f"微调攻击实验参数:")
    print(f"  - 微调轮数: {finetune_epochs}")
    print(f"  - 评估间隔: {eval_interval}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 批次大小: {batch_size}")
    print("-" * 60)

    # 加载数据
    print("加载数据...")
    mnist_test_loader = load_mnist_test_data(batch_size=128)
    chestmnist_train_set, chestmnist_test_set = load_chestmnist_data()

    # 创建数据加载器 (使用安全的数据加载器创建函数)
    train_loader = create_safe_dataloader(chestmnist_train_set, batch_size=batch_size, shuffle=True)
    test_loader = create_safe_dataloader(chestmnist_test_set, batch_size=batch_size*2, shuffle=False)

    # 加载主任务模型
    print("加载主任务模型...")
    model = load_main_task_model(model_path, device)
    if model is None:
        print("❌ 主任务模型加载失败")
        return

    # 保存原始模型状态
    original_model_state = copy.deepcopy(model.state_dict())

    # 初始化水印重建器
    reconstructor = WatermarkReconstructor(key_matrix_dir, autoencoder_dir)

    print("开始微调攻击实验...")
    print("=" * 80)
    
    # 第0轮：测试微调前的水印检测
    print("=== 第0轮评估（微调前）===")
    
    # 先进行AUC评估
    model.eval()
    test_loss, all_predictions, all_targets = 0.0, [], []
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.float()).item()
            all_predictions.append(torch.sigmoid(output).cpu().numpy())
            all_targets.append(target.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算AUC（使用与训练时相同的逻辑）
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

    # 计算准确率
    pred_binary = (all_predictions > 0.5).astype(int)
    accuracy = np.mean((pred_binary == all_targets).astype(float))
    
    print(f"测试损失: {avg_test_loss:.4f} | AUC: {mean_auc:.4f} | 准确率: {accuracy:.2%}")
    
    # 进行ΔPCC评估
    delta_pcc_result_0 = evaluate_delta_pcc(
        original_model_state, original_model_state, reconstructor,
        mnist_test_loader, device, perf_fail_ratio=0.7
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
    if delta_pcc_result_0:
        initial_result.update({
            'perf_before': delta_pcc_result_0['perf_before'],
            'perf_fail': delta_pcc_result_0['perf_fail'],
            'tau': delta_pcc_result_0['tau'],
            'delta_perf': delta_pcc_result_0['delta_perf'],
            'delta_pcc': delta_pcc_result_0['delta_pcc'],
            'is_infringement': delta_pcc_result_0['is_infringement'],
            'result_text': delta_pcc_result_0['result_text']
        })
    else:
        initial_result.update({
            'perf_before': None,
            'perf_fail': None,
            'tau': None,
            'delta_perf': None,
            'delta_pcc': None,
            'is_infringement': None,
            'result_text': 'N/A'
        })

    # 进行微调训练
    model_states, performance_metrics = finetune_model(
        model, train_loader, test_loader,
        epochs=finetune_epochs, lr=learning_rate,
        device=device, eval_interval=eval_interval,
        reconstructor=reconstructor, original_model_state=original_model_state,
        mnist_test_loader=mnist_test_loader
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
    save_results(results)

    # 输出总结
    print("\n" + "=" * 80)
    print("微调攻击实验总结")
    print("=" * 80)
    print(f"{'轮次':<4} {'训练损失':<10} {'测试损失':<10} {'测试AUC':<8} {'测试准确率%':<10} {'ΔPCC':<8} {'侵权判断':<8}")
    print("-" * 80)

    for result in results:
        delta_pcc_str = f"{result['delta_pcc']:.4f}" if result['delta_pcc'] is not None else "N/A"
        infringement_str = "是" if result['is_infringement'] else "否" if result['is_infringement'] is not None else "N/A"
        
        print(f"{result['epoch']:>3}  "
              f"{result['train_loss']:>8.4f}  "
              f"{result['test_loss']:>8.4f}  "
              f"{result['test_auc']:>6.4f}  "
              f"{result['test_accuracy']:>8.2%}  "
              f"{delta_pcc_str:>6}  "
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

        print(f"测试AUC变化: {initial_auc:.4f} → {final_auc:.4f} (变化: {auc_change:+.4f})")
        print(f"测试准确率变化: {initial_acc:.2%} → {final_acc:.2%} (变化: {acc_change:+.2%})")
        
        # 分析ΔPCC趋势
        delta_pcc_values = [r['delta_pcc'] for r in results if r['delta_pcc'] is not None]
        if len(delta_pcc_values) > 1:
            initial_delta_pcc = delta_pcc_values[0]
            final_delta_pcc = delta_pcc_values[-1]
            delta_pcc_change = final_delta_pcc - initial_delta_pcc
            print(f"ΔPCC变化: {initial_delta_pcc:.4f} → {final_delta_pcc:.4f} (变化: {delta_pcc_change:+.4f})")
        
        # 分析侵权判断
        infringement_count = sum(1 for r in results if r['is_infringement'] is True)
        total_evaluations = sum(1 for r in results if r['is_infringement'] is not None)
        if total_evaluations > 0:
            infringement_rate = infringement_count / total_evaluations
            print(f"侵权判断: {infringement_count}/{total_evaluations} 轮被判定为侵权 ({infringement_rate:.1%})")

    print("\n微调攻击实验完成！")


if __name__ == '__main__':
    main()
