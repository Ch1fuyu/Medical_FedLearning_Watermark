#!/usr/bin/env python3
"""
阈值剪枝攻击实验
对保存的模型进行不同阈值的剪枝攻击，测试模型鲁棒性
"""

import os
import sys
import argparse
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.resnet import resnet18
from models.alexnet import AlexNet
from utils.dataset import get_data
from utils.trainer_private import TesterPrivate
from utils.watermark_reconstruction import WatermarkReconstructor, create_test_loader_for_autoencoder
from config.globals import set_seed

def load_model_from_pkl(pkl_path, model_type='resnet', num_classes=14, in_channels=3):
    """从pkl文件加载模型"""
    # 创建模型
    if model_type == 'resnet':
        model = resnet18(num_classes=num_classes, in_channels=in_channels, input_size=28)
    elif model_type == 'alexnet':
        model = AlexNet(in_channels, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载pkl文件
    logs = torch.load(pkl_path, map_location='cpu', weights_only=False)
    
    # 获取最佳模型权重
    if 'best_model' in logs['net_info'] and len(logs['net_info']['best_model']) > 0:
        model_state = logs['net_info']['best_model'][0]
        model.load_state_dict(model_state)
        print("✓ 成功加载最佳模型权重")
    else:
        print("❌ 未找到最佳模型权重")
        return None, None
    
    return model, logs

def threshold_pruning(model, threshold_percent):
    """对模型进行阈值剪枝"""
    pruned_model = copy.deepcopy(model)
    
    # 计算所有参数的绝对值
    all_params = torch.cat([param.data.abs().flatten() for param in pruned_model.parameters()])
    total_count = all_params.numel()
    
    # 计算阈值并应用剪枝
    threshold_value = torch.quantile(all_params, threshold_percent / 100.0)
    pruned_count = 0
    
    for param in pruned_model.parameters():
        mask = param.data.abs() > threshold_value
        pruned_count += (~mask).sum().item()
        param.data = param.data * mask.float()
    
    return pruned_model, pruned_count, total_count

def evaluate_model(model, test_loader, device, model_name="Model"):
    """评估模型性能"""
    model = model.to(device)
    tester = TesterPrivate(model, device)
    loss, acc, auc, sample_acc = tester.test(test_loader)
    return {'loss': loss, 'acc': acc, 'auc': auc, 'sample_acc': sample_acc}

def evaluate_watermark_reconstruction(model, key_matrix_dir, autoencoder_weights_dir, 
                                    autoencoder_test_loader, client_id=0, use_deltapcc=True, perf_fail_mse=0.5):
    """
    评估水印重建和侵权判断
    
    Args:
        model: 待评估的模型
        key_matrix_dir: 密钥矩阵目录
        autoencoder_weights_dir: 自编码器权重目录
        autoencoder_test_loader: 自编码器测试数据加载器
        client_id: 客户端ID（仅用于ΔPCC方法，传统方法会使用所有客户端）
        use_deltapcc: 是否使用ΔPCC方法
        perf_fail_mse: 失效性能的MSE值（仅用于ΔPCC方法）
    
    Returns:
        dict: 水印重建和侵权判断结果
        
    Note:
        - 当use_deltapcc=True时，使用指定client_id的水印参数
        - 当use_deltapcc=False时，自动从所有客户端提取水印参数进行侵权判断
    """
    try:
        # 创建水印重建器
        reconstructor = WatermarkReconstructor(key_matrix_dir, autoencoder_weights_dir)
        
        if use_deltapcc:
            # 使用ΔPCC方法
            # 设置ΔPCC评估参数
            reconstructor.setup_deltapcc_evaluation(autoencoder_test_loader, perf_fail_mse)
            
            # 计算ΔPCC
            deltapcc_result = reconstructor.calculate_deltapcc(model.state_dict(), client_id, check_pruning=True)
            
            if not deltapcc_result['reconstruction_success']:
                return {
                    'watermark_reconstruction_success': False,
                    'infringement_detected': False,
                    'error': 'Failed to reconstruct autoencoder from watermark',
                    'delta_pcc': float('inf'),
                    'perf_before': float('inf'),
                    'perf_after': float('inf'),
                    'delta_perf': float('inf'),
                    'tau': float('inf')
                }
            
            return {
                'watermark_reconstruction_success': True,
                'infringement_detected': deltapcc_result['infringement_detected'],
                'delta_pcc': deltapcc_result['delta_pcc'],
                'perf_before': deltapcc_result['perf_before'],
                'perf_after': deltapcc_result['perf_after'],
                'perf_fail': deltapcc_result['perf_fail'],
                'delta_perf': deltapcc_result['delta_perf'],
                'tau': deltapcc_result['tau'],
                'psnr': deltapcc_result['psnr'],
                'ssim': deltapcc_result['ssim'],
                'method': 'deltapcc'
            }
        else:
            # 使用传统方法 - 从所有客户端提取水印参数进行侵权判断
            # 重建自编码器
            reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(
                model.state_dict()
            )
            
            if reconstructed_autoencoder is None:
                return {
                    'watermark_reconstruction_success': False,
                    'infringement_detected': False,
                    'error': 'Failed to reconstruct autoencoder from watermark',
                    'method': 'traditional'
                }
            
            # 比较性能
            comparison_results = reconstructor.compare_with_original_autoencoder(
                reconstructed_autoencoder, autoencoder_test_loader
            )
            
            # 侵权判断
            infringement_results = reconstructor.assess_infringement(comparison_results)
            
            return {
                'watermark_reconstruction_success': True,
                'infringement_detected': infringement_results['overall_infringement'],
                'reconstructed_psnr': comparison_results['reconstructed']['psnr'],
                'original_psnr': comparison_results['original']['psnr'],
                'psnr_retention': comparison_results['retention']['psnr_retention'],
                'reconstructed_ssim': comparison_results['reconstructed']['ssim'],
                'original_ssim': comparison_results['original']['ssim'],
                'ssim_retention': comparison_results['retention']['ssim_retention'],
                'reconstruction_loss': comparison_results['reconstructed']['reconstruction_loss'],
                'original_reconstruction_loss': comparison_results['original']['reconstruction_loss'],
                'infringement_criteria': infringement_results,
                'method': 'traditional'
            }
        
    except Exception as e:
        return {
            'watermark_reconstruction_success': False,
            'infringement_detected': False,
            'error': str(e),
            'method': 'deltapcc' if use_deltapcc else 'traditional'
        }

def main():
    parser = argparse.ArgumentParser(description='阈值剪枝攻击实验')
    parser.add_argument('--pkl_path', type=str,
                        default='save/resnet/chestmnist/202509172334_Dp_False_sig_0.1_iid_True_ns_1_wt_gamma_lt_sign_bit_20_alp_0.2_nb_1_type_True_tri_40_ep_100_le_2_cn_10_fra_1.0000_acc_0.6919.pkl',
                       help='待测试的pkl文件路径')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=['resnet', 'alexnet'], help='模型类型')
    parser.add_argument('--dataset', type=str, default='chestmnist', 
                       help='数据集名称')
    parser.add_argument('--num_classes', type=int, default=14, 
                       help='类别数')
    parser.add_argument('--in_channels', type=int, default=3, 
                       help='输入通道数')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='批大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--key_matrix_dir', type=str, default='save/key_matrix',
                       help='密钥矩阵目录')
    parser.add_argument('--autoencoder_weights_dir', type=str, default='save/autoencoder',
                       help='自编码器权重目录')
    parser.add_argument('--enable_watermark_reconstruction',default=True,
                        action='store_true',
                       help='启用水印重建和侵权判断功能')
    parser.add_argument('--client_id', type=int, default=0,
                       help='用于水印重建的客户端ID')
    parser.add_argument('--use_deltapcc', action='store_true', default=True,
                       help='使用ΔPCC方法进行侵权判断')
    parser.add_argument('--perf_fail_mse', type=float, default=0.05,
                       help='失效性能的MSE值（容忍下限）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed = args.seed
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print("=" * 80)
    print("阈值剪枝攻击实验")
    print("=" * 80)
    print(f"随机种子: {seed}")
    print(f"模型文件: {args.pkl_path}")
    print(f"模型文件名: {os.path.basename(args.pkl_path)}")
    print(f"模型类型: {args.model_type}")
    print(f"数据集: {args.dataset}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"启用水印重建: {args.enable_watermark_reconstruction}")
    if args.enable_watermark_reconstruction:
        print(f"密钥矩阵目录: {args.key_matrix_dir}")
        print(f"自编码器权重目录: {args.autoencoder_weights_dir}")
        print(f"客户端ID: {args.client_id}")
        print(f"使用ΔPCC方法: {args.use_deltapcc}")
        if args.use_deltapcc:
            print(f"失效性能MSE: {args.perf_fail_mse}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(args.pkl_path):
        print(f"❌ 文件不存在: {args.pkl_path}")
        return
    
    # 加载模型
    model, logs = load_model_from_pkl(
        args.pkl_path, 
        args.model_type, 
        args.num_classes, 
        args.in_channels
    )
    
    if model is None:
        print("❌ 模型加载失败")
        return
    
    # 加载测试数据 - 确保与训练时完全一致
    print("加载测试数据...")
    _, test_set, _ = get_data(
        dataset_name=args.dataset,
        data_root='./data',
        iid=True,
        client_num=1
    )
    
    # 设置DataLoader的随机种子
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,  # 设为0避免多进程随机性问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 如果启用水印重建功能，创建自编码器测试数据加载器
    autoencoder_test_loader = None
    if args.enable_watermark_reconstruction:
        print("创建自编码器测试数据加载器...")
        try:
            autoencoder_test_loader = create_test_loader_for_autoencoder(
                batch_size=args.batch_size, 
                num_samples=1000
            )
            print("✓ 自编码器测试数据加载器创建成功")
        except Exception as e:
            print(f"❌ 创建自编码器测试数据加载器失败: {e}")
            print("   将禁用水印重建功能")
            args.enable_watermark_reconstruction = False
    
    # 显示训练时记录的性能
    print("\n训练时记录的性能:")
    if 'best_model_acc' in logs['net_info']:
        print(f"  训练时最佳准确率: {logs['net_info']['best_model_acc']:.4f}%")
    if 'best_model_auc' in logs['net_info']:
        print(f"  训练时最佳AUC: {logs['net_info']['best_model_auc']:.4f}")
    
    # 评估原始模型
    print("\n当前评估的原始模型性能:")
    original_metrics = evaluate_model(model, test_loader, args.device, "原始模型")
    
    print(f"  Loss: {original_metrics['loss']:.4f}")
    print(f"  Acc:  {original_metrics['acc']:.4f}%")
    print(f"  AUC:  {original_metrics['auc']:.4f}")
    print(f"  Sample Acc: {original_metrics['sample_acc']:.4f}%")
    
    # 计算差异
    if 'best_model_acc' in logs['net_info'] and 'best_model_auc' in logs['net_info']:
        acc_diff = logs['net_info']['best_model_acc'] - original_metrics['acc']
        auc_diff = logs['net_info']['best_model_auc'] - original_metrics['auc']
        print(f"\n性能差异:")
        print(f"  准确率差异: {acc_diff:.4f}%")
        print(f"  AUC差异: {auc_diff:.4f}")
        
        if abs(acc_diff) > 1.0 or abs(auc_diff) > 0.05:
            print("  ⚠️  检测到显著性能差异，可能原因:")
            print("    1. 随机种子设置不一致")
            print("    2. 数据加载顺序不同")
            print("    3. 模型包含水印参数")
            print("    4. 评估环境不同")
    
    # 进行不同阈值的剪枝攻击
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []
    
    # 如果启用水印重建功能，预先创建水印重建器（避免重复创建）
    reconstructor = None
    if args.enable_watermark_reconstruction and autoencoder_test_loader is not None:
        print("创建水印重建器...")
        reconstructor = WatermarkReconstructor(args.key_matrix_dir, args.autoencoder_weights_dir)
        
        if args.use_deltapcc:
            # 设置ΔPCC评估参数（只设置一次）
            print("设置ΔPCC评估参数...")
            reconstructor.setup_deltapcc_evaluation(autoencoder_test_loader, args.perf_fail_mse)
    
    print(f"\n开始剪枝攻击实验...")
    if args.enable_watermark_reconstruction:
        print("-" * 120)
        if args.use_deltapcc:
            print(f"{'阈值(%)':<8} {'剪枝率(%)':<10} {'Loss':<8} {'Acc(%)':<8} {'AUC':<8} {'Sample Acc(%)':<12} {'水印重建':<8} {'侵权检测':<8} {'ΔPCC':<10}")
        else:
            print(f"{'阈值(%)':<8} {'剪枝率(%)':<10} {'Loss':<8} {'Acc(%)':<8} {'AUC':<8} {'Sample Acc(%)':<12} {'水印重建':<8} {'侵权检测':<8} {'PSNR保持率':<10}")
        print("-" * 120)
    else:
        print("-" * 80)
        print(f"{'阈值(%)':<8} {'剪枝率(%)':<10} {'Loss':<8} {'Acc(%)':<8} {'AUC':<8} {'Sample Acc(%)':<12}")
        print("-" * 80)
    
    for threshold in thresholds:
        # 创建模型副本进行剪枝
        pruned_model = copy.deepcopy(model)
        
        # 进行剪枝
        pruned_model, pruned_count, total_count = threshold_pruning(pruned_model, threshold)
        pruning_rate = (pruned_count / total_count) * 100
        
        # 评估剪枝后的模型
        metrics = evaluate_model(pruned_model, test_loader, args.device, f"剪枝{threshold}%")
        
        # 水印重建和侵权判断
        watermark_results = {}
        if args.enable_watermark_reconstruction and autoencoder_test_loader is not None and reconstructor is not None:
            if args.use_deltapcc:
                # 使用ΔPCC方法
                deltapcc_result = reconstructor.calculate_deltapcc(pruned_model.state_dict(), args.client_id, check_pruning=True)
                
                if not deltapcc_result['reconstruction_success']:
                    watermark_results = {
                        'watermark_reconstruction_success': False,
                        'infringement_detected': False,
                        'error': 'Failed to reconstruct autoencoder from watermark',
                        'delta_pcc': float('inf'),
                        'perf_before': float('inf'),
                        'perf_after': float('inf'),
                        'delta_perf': float('inf'),
                        'tau': float('inf')
                    }
                else:
                    watermark_results = {
                        'watermark_reconstruction_success': True,
                        'infringement_detected': deltapcc_result['infringement_detected'],
                        'delta_pcc': deltapcc_result['delta_pcc'],
                        'perf_before': deltapcc_result['perf_before'],
                        'perf_after': deltapcc_result['perf_after'],
                        'perf_fail': deltapcc_result['perf_fail'],
                        'delta_perf': deltapcc_result['delta_perf'],
                        'tau': deltapcc_result['tau'],
                        'psnr': deltapcc_result['psnr'],
                        'ssim': deltapcc_result['ssim'],
                        'method': 'deltapcc'
                    }
            else:
                # 使用传统方法
                watermark_results = evaluate_watermark_reconstruction(
                    pruned_model, 
                    args.key_matrix_dir, 
                    args.autoencoder_weights_dir,
                    autoencoder_test_loader,
                    args.client_id,
                    args.use_deltapcc,
                    args.perf_fail_mse
                )
        
        # 记录结果
        result = {
            'threshold': threshold,
            'pruning_rate': pruning_rate,
            'loss': metrics['loss'],
            'acc': metrics['acc'],
            'auc': metrics['auc'],
            'sample_acc': metrics['sample_acc'],
            'acc_drop': original_metrics['acc'] - metrics['acc'],
            'auc_drop': original_metrics['auc'] - metrics['auc']
        }
        
        # 添加水印重建结果
        if watermark_results:
            if watermark_results.get('method') == 'deltapcc':
                # ΔPCC方法结果
                result.update({
                    'watermark_reconstruction_success': watermark_results.get('watermark_reconstruction_success', False),
                    'infringement_detected': watermark_results.get('infringement_detected', False),
                    'delta_pcc': watermark_results.get('delta_pcc', float('inf')),
                    'perf_before': watermark_results.get('perf_before', float('inf')),
                    'perf_after': watermark_results.get('perf_after', float('inf')),
                    'perf_fail': watermark_results.get('perf_fail', float('inf')),
                    'delta_perf': watermark_results.get('delta_perf', float('inf')),
                    'tau': watermark_results.get('tau', float('inf')),
                    'psnr': watermark_results.get('psnr', 0.0),
                    'ssim': watermark_results.get('ssim', 0.0),
                    'method': 'deltapcc'
                })
            else:
                # 传统方法结果
                result.update({
                    'watermark_reconstruction_success': watermark_results.get('watermark_reconstruction_success', False),
                    'infringement_detected': watermark_results.get('infringement_detected', False),
                    'psnr_retention': watermark_results.get('psnr_retention', 0.0),
                    'ssim_retention': watermark_results.get('ssim_retention', 0.0),
                    'reconstruction_loss': watermark_results.get('reconstruction_loss', float('inf')),
                    'original_reconstruction_loss': watermark_results.get('original_reconstruction_loss', float('inf')),
                    'method': 'traditional'
                })
        
        results.append(result)
        
        # 输出结果
        if args.enable_watermark_reconstruction and watermark_results:
            watermark_success = "✓" if watermark_results.get('watermark_reconstruction_success', False) else "❌"
            infringement = "✓" if watermark_results.get('infringement_detected', False) else "❌"
            
            if watermark_results.get('method') == 'deltapcc':
                # ΔPCC方法输出
                delta_pcc = watermark_results.get('delta_pcc', float('inf'))
                delta_pcc_str = f"{delta_pcc:.4f}" if delta_pcc != float('inf') else "∞"
                print(f"{threshold:<8} {pruning_rate:<10.2f} {metrics['loss']:<8.4f} {metrics['acc']:<8.2f} {metrics['auc']:<8.4f} {metrics['sample_acc']:<12.2f} {watermark_success:<8} {infringement:<8} {delta_pcc_str:<10}")
            else:
                # 传统方法输出
                psnr_retention = f"{watermark_results.get('psnr_retention', 0.0):.2%}"
                print(f"{threshold:<8} {pruning_rate:<10.2f} {metrics['loss']:<8.4f} {metrics['acc']:<8.2f} {metrics['auc']:<8.4f} {metrics['sample_acc']:<12.2f} {watermark_success:<8} {infringement:<8} {psnr_retention:<10}")
        else:
            print(f"{threshold:<8} {pruning_rate:<10.2f} {metrics['loss']:<8.4f} {metrics['acc']:<8.2f} {metrics['auc']:<8.4f} {metrics['sample_acc']:<12.2f}")
    
    if args.enable_watermark_reconstruction:
        print("-" * 120)
    else:
        print("-" * 80)
    
    # 分析结果
    print(f"\n结果分析:")
    print("=" * 50)
    
    # 找到性能下降最严重的剪枝阈值
    max_acc_drop_idx = np.argmax([r['acc_drop'] for r in results])
    max_auc_drop_idx = np.argmax([r['auc_drop'] for r in results])
    
    print(f"最大准确率下降: {results[max_acc_drop_idx]['acc_drop']:.2f}% (阈值: {results[max_acc_drop_idx]['threshold']}%)")
    print(f"最大AUC下降: {results[max_auc_drop_idx]['auc_drop']:.4f} (阈值: {results[max_auc_drop_idx]['threshold']}%)")
    
    # 找到性能保持较好的剪枝阈值
    good_pruning_thresholds = []
    for r in results:
        if r['acc_drop'] < 5.0 and r['auc_drop'] < 0.05:  # 准确率下降<5%, AUC下降<0.05
            good_pruning_thresholds.append(r['threshold'])
    
    if good_pruning_thresholds:
        print(f"性能保持较好的剪枝阈值: {good_pruning_thresholds}%")
    else:
        print("所有剪枝阈值都导致显著性能下降")
    
    # 水印重建和侵权判断分析
    if args.enable_watermark_reconstruction:
        print(f"\n水印重建和侵权判断分析:")
        print("-" * 50)
        
        # 统计水印重建成功率
        successful_reconstructions = [r for r in results if r.get('watermark_reconstruction_success', False)]
        reconstruction_success_rate = len(successful_reconstructions) / len(results) * 100
        print(f"水印重建成功率: {reconstruction_success_rate:.1f}% ({len(successful_reconstructions)}/{len(results)})")
        
        # 统计侵权检测结果
        infringement_detected = [r for r in results if r.get('infringement_detected', False)]
        infringement_rate = len(infringement_detected) / len(results) * 100
        print(f"侵权检测率: {infringement_rate:.1f}% ({len(infringement_detected)}/{len(results)})")
        
        if args.use_deltapcc:
            # ΔPCC方法分析
            print(f"\nΔPCC方法分析:")
            print("-" * 30)
            
            # 分析ΔPCC值
            if successful_reconstructions:
                delta_pcc_values = [r.get('delta_pcc', float('inf')) for r in successful_reconstructions]
                valid_delta_pcc = [v for v in delta_pcc_values if v != float('inf')]
                
                if valid_delta_pcc:
                    avg_delta_pcc = np.mean(valid_delta_pcc)
                    min_delta_pcc = np.min(valid_delta_pcc)
                    max_delta_pcc = np.max(valid_delta_pcc)
                    median_delta_pcc = np.median(valid_delta_pcc)
                    
                    print(f"ΔPCC值 - 平均: {avg_delta_pcc:.4f}, 最小: {min_delta_pcc:.4f}, 最大: {max_delta_pcc:.4f}, 中位数: {median_delta_pcc:.4f}")
                    
                    # 找到ΔPCC最小的剪枝阈值（最可能侵权）
                    best_delta_pcc_idx = np.argmin(valid_delta_pcc)
                    best_delta_pcc_threshold = successful_reconstructions[best_delta_pcc_idx]['threshold']
                    print(f"最小ΔPCC: {min_delta_pcc:.4f} (阈值: {best_delta_pcc_threshold}%)")
                    
                    # 分析性能变化
                    perf_before_values = [r.get('perf_before', float('inf')) for r in successful_reconstructions]
                    perf_after_values = [r.get('perf_after', float('inf')) for r in successful_reconstructions]
                    valid_perf_before = [v for v in perf_before_values if v != float('inf')]
                    valid_perf_after = [v for v in perf_after_values if v != float('inf')]
                    
                    if valid_perf_before and valid_perf_after:
                        print(f"基准性能 (perf_before): {valid_perf_before[0]:.6f}")
                        print(f"失效性能 (perf_fail): {successful_reconstructions[0].get('perf_fail', 0.0):.6f}")
                        print(f"阈值 (τ): {successful_reconstructions[0].get('tau', 0.0):.6f}")
                        print(f"测试后性能 - 平均: {np.mean(valid_perf_after):.6f}, 最小: {np.min(valid_perf_after):.6f}, 最大: {np.max(valid_perf_after):.6f}")
                else:
                    print("所有ΔPCC值都无效")
            
            # 分析PSNR和SSIM
            if successful_reconstructions:
                psnr_values = [r.get('psnr', 0.0) for r in successful_reconstructions]
                ssim_values = [r.get('ssim', 0.0) for r in successful_reconstructions]
                
                print(f"PSNR - 平均: {np.mean(psnr_values):.2f}, 最小: {np.min(psnr_values):.2f}, 最大: {np.max(psnr_values):.2f}")
                print(f"SSIM - 平均: {np.mean(ssim_values):.4f}, 最小: {np.min(ssim_values):.4f}, 最大: {np.max(ssim_values):.4f}")
        else:
            # 传统方法分析
            print(f"\n传统方法分析:")
            print("-" * 30)
            
            # 分析PSNR保持率
            if successful_reconstructions:
                psnr_retentions = [r.get('psnr_retention', 0.0) for r in successful_reconstructions]
                avg_psnr_retention = np.mean(psnr_retentions)
                min_psnr_retention = np.min(psnr_retentions)
                max_psnr_retention = np.max(psnr_retentions)
                
                print(f"PSNR保持率 - 平均: {avg_psnr_retention:.2%}, 最小: {min_psnr_retention:.2%}, 最大: {max_psnr_retention:.2%}")
                
                # 找到PSNR保持率最高的剪枝阈值
                best_psnr_idx = np.argmax(psnr_retentions)
                best_psnr_threshold = successful_reconstructions[best_psnr_idx]['threshold']
                print(f"最佳PSNR保持率: {max_psnr_retention:.2%} (阈值: {best_psnr_threshold}%)")
            
            # 分析SSIM保持率
            if successful_reconstructions:
                ssim_retentions = [r.get('ssim_retention', 0.0) for r in successful_reconstructions]
                avg_ssim_retention = np.mean(ssim_retentions)
                min_ssim_retention = np.min(ssim_retentions)
                max_ssim_retention = np.max(ssim_retentions)
                
                print(f"SSIM保持率 - 平均: {avg_ssim_retention:.2%}, 最小: {min_ssim_retention:.2%}, 最大: {max_ssim_retention:.2%}")
        
        # 分析重建损失
        if successful_reconstructions:
            reconstruction_losses = [r.get('reconstruction_loss', float('inf')) for r in successful_reconstructions]
            original_losses = [r.get('original_reconstruction_loss', float('inf')) for r in successful_reconstructions]
            
            valid_losses = [(r, o) for r, o in zip(reconstruction_losses, original_losses) if r != float('inf') and o != float('inf')]
            if valid_losses:
                loss_ratios = [r/o for r, o in valid_losses]
                avg_loss_ratio = np.mean(loss_ratios)
                print(f"重建损失比率 - 平均: {avg_loss_ratio:.2f}x")
                
                # 找到损失比率最低的剪枝阈值
                min_loss_ratio_idx = np.argmin(loss_ratios)
                min_loss_ratio_threshold = successful_reconstructions[min_loss_ratio_idx]['threshold']
                print(f"最佳重建损失比率: {min(loss_ratios):.2f}x (阈值: {min_loss_ratio_threshold}%)")
        
        # 综合侵权风险评估
        print(f"\n综合侵权风险评估:")
        high_risk_thresholds = []
        medium_risk_thresholds = []
        low_risk_thresholds = []
        
        for r in results:
            if not r.get('watermark_reconstruction_success', False):
                continue
                
            psnr_retention = r.get('psnr_retention', 0.0)
            ssim_retention = r.get('ssim_retention', 0.0)
            infringement_detected = r.get('infringement_detected', False)
            
            # 风险评估标准
            if infringement_detected and psnr_retention >= 0.7 and ssim_retention >= 0.7:
                high_risk_thresholds.append(r['threshold'])
            elif psnr_retention >= 0.5 or ssim_retention >= 0.5:
                medium_risk_thresholds.append(r['threshold'])
            else:
                low_risk_thresholds.append(r['threshold'])
        
        print(f"高风险阈值 (确认侵权): {high_risk_thresholds}%")
        print(f"中风险阈值 (可能侵权): {medium_risk_thresholds}%")
        print(f"低风险阈值 (未侵权): {low_risk_thresholds}%")
    
    # 保存结果到CSV
    try:
        os.makedirs('save/pruning_results', exist_ok=True)
        
        # 创建DataFrame，只包含剪枝相关的数据
        df = pd.DataFrame(results)
        
        # 重新排列列的顺序
        if args.enable_watermark_reconstruction:
            if args.use_deltapcc:
                columns_order = ['threshold', 'pruning_rate', 'loss', 'acc', 'auc', 'sample_acc', 
                                'acc_drop', 'auc_drop', 'watermark_reconstruction_success', 
                                'infringement_detected', 'delta_pcc', 'perf_before', 'perf_after', 
                                'perf_fail', 'delta_perf', 'tau', 'psnr', 'ssim', 'method']
            else:
                columns_order = ['threshold', 'pruning_rate', 'loss', 'acc', 'auc', 'sample_acc', 
                                'acc_drop', 'auc_drop', 'watermark_reconstruction_success', 
                                'infringement_detected', 'psnr_retention', 'ssim_retention',
                                'reconstruction_loss', 'original_reconstruction_loss', 'method']
        else:
            columns_order = ['threshold', 'pruning_rate', 'loss', 'acc', 'auc', 'sample_acc', 
                            'acc_drop', 'auc_drop']
        df = df[columns_order]
        
        # 生成带日期的文件名
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = os.path.basename(args.pkl_path).replace('.pkl', '')
        csv_path = f'save/pruning_results/pruning_attack_{args.model_type}_{args.dataset}_{timestamp}.csv'
        
        # 准备备注信息
        watermark_info = ""
        if args.enable_watermark_reconstruction:
            watermark_info = f"""# 水印重建功能: 已启用
# 密钥矩阵目录: {args.key_matrix_dir}
# 自编码器权重目录: {args.autoencoder_weights_dir}
# 客户端ID: {args.client_id}
# 
# 水印重建相关列说明:
# watermark_reconstruction_success: 水印重建是否成功
# infringement_detected: 是否检测到侵权
# psnr_retention: PSNR保持率
# ssim_retention: SSIM保持率
# reconstruction_loss: 重建损失
# original_reconstruction_loss: 原始重建损失
#
"""
        else:
            watermark_info = "# 水印重建功能: 未启用\n"
        
        model_info = f"""# 剪枝攻击实验结果
# 模型文件: {args.pkl_path}
# 模型类型: {args.model_type}
# 数据集: {args.dataset}
# 设备: {args.device}
# 批大小: {args.batch_size}
# 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 原始模型性能: Loss={original_metrics['loss']:.4f}, Acc={original_metrics['acc']:.4f}%, AUC={original_metrics['auc']:.4f}
# 
{watermark_info}
# 数据说明:
# threshold: 剪枝阈值(%)
# pruning_rate: 实际剪枝率(%)
# loss: 损失值
# acc: 准确率(%)
# auc: AUC值
# sample_acc: 样本级准确率(%)
# acc_drop: 准确率下降(%)
# auc_drop: AUC下降
#
"""
        
        # 先写入备注信息，再写入数据
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write(model_info)
            df.to_csv(f, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存到: {csv_path}")
        print(f"CSV包含 {len(results)} 行剪枝数据，模型信息已作为备注保存")
    except Exception as e:
        print(f"保存结果失败: {e}")
    
    print("=" * 80)
    print("实验完成!")

if __name__ == '__main__':
    main()
