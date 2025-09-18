#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调攻击测试脚本
模拟攻击者使用相同数据集对模型进行单机微调，测试水印的鲁棒性
"""

import os
import sys
import copy
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.args import get_args
from utils.dataset import get_dataset
from models.resnet import ResNet18
from models.alexnet import AlexNet
from utils.key_matrix_utils import KeyMatrixManager
from utils.watermark_reconstruction import WatermarkReconstructor, create_test_loader_for_autoencoder


def load_model_from_pkl(pkl_path):
    """从pkl文件加载模型"""
    print(f"Loading model from: {pkl_path}")
    
    data = torch.load(pkl_path, weights_only=False)
    model_state = data['model_state_dict']
    args = data['args']
    best_acc = data.get('best_acc', 0.0)
    best_auc = data.get('best_auc', 0.0)
    
    print(f"Model: {args.model}, Dataset: {args.dataset}, Device: {args.device}")
    print(f"Best accuracy: {best_acc:.4f}, Best AUC: {best_auc:.4f}")
    
    return model_state, args, best_acc, best_auc

def create_model(args):
    """根据参数创建模型"""
    if args.model == 'resnet18':
        return ResNet18(num_classes=args.num_classes)
    elif args.model == 'alexnet':
        return AlexNet(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

def finetune_model(model, train_loader, val_loader, args, finetune_epochs=10, finetune_lr=0.001):
    """对模型进行微调"""
    print(f"\n开始微调攻击 - 轮数: {finetune_epochs}, 学习率: {finetune_lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    finetune_history = []
    
    for epoch in range(finetune_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            if args.device == 'cuda' and torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        val_loss, val_acc, val_auc, val_sample_acc = evaluate_model(model, val_loader, args)
        
        # 记录历史
        finetune_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_sample_acc': val_sample_acc
        })
        
        print(f"Epoch {epoch+1:2d}/{finetune_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
    
    return finetune_history

def evaluate_model(model, data_loader, args):
    """评估模型性能"""
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    sample_acc_meter = 0.0
    run_count = 0
    
    all_y_true = []
    all_y_score = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            if args.device == 'cuda' and torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = criterion(output, target)
            
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            acc = correct / target.size(0)
            
            # 计算样本级准确率（多标签）
            if args.dataset in ['chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist']:
                # 对于多标签分类，使用sigmoid + 阈值0.5
                probs = torch.sigmoid(output)
                pred_labels = (probs > 0.5).float()
                true_labels = target.float()
                sample_acc = (pred_labels == true_labels).all(dim=1).float().mean().item()
                sample_acc_meter += sample_acc * target.size(0)
                
                # 收集AUC计算数据
                all_y_true.extend(target.cpu().numpy())
                all_y_score.extend(probs.cpu().numpy())
            else:
                # 单标签分类
                sample_acc = acc
                sample_acc_meter += sample_acc * target.size(0)
                
                # 收集AUC计算数据
                all_y_true.extend(target.cpu().numpy())
                all_y_score.extend(torch.softmax(output, dim=1).cpu().numpy())
            
            loss_meter += loss.item() * target.size(0)
            acc_meter += correct
            run_count += target.size(0)
    
    # 计算最终指标
    final_loss = loss_meter / run_count
    final_acc = 100. * acc_meter / run_count
    final_sample_acc = 100. * sample_acc_meter / run_count
    
    # 计算AUC
    if len(all_y_true) > 0 and len(all_y_score) > 0:
        try:
            if args.dataset in ['chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist']:
                # 多标签AUC
                auc_val = roc_auc_score(all_y_true, all_y_score, average='macro', multi_class='ovr')
            else:
                # 单标签AUC
                if len(np.unique(all_y_true)) > 2:
                    auc_val = roc_auc_score(all_y_true, all_y_score, multi_class='ovr', average='macro')
                else:
                    auc_val = roc_auc_score(all_y_true, all_y_score)
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            auc_val = 0.0
    else:
        auc_val = 0.0
    
    return final_loss, final_acc, auc_val, final_sample_acc

def detect_watermark(model, key_manager, client_id):
    """检测模型中是否存在水印"""
    if key_manager is None:
        return False, 0.0
    
    try:
        # 提取水印
        watermark_values = key_manager.extract_watermark(model.state_dict(), client_id)
        
        # 计算水印强度（非零参数的比例）
        watermark_strength = np.mean(np.abs(watermark_values) > 1e-6)
        
        # 如果水印强度大于阈值，认为存在水印
        has_watermark = watermark_strength > 0.1  # 阈值可调整
        
        return has_watermark, watermark_strength
    except Exception as e:
        print(f"水印检测失败: {e}")
        return False, 0.0

def main():
    parser = argparse.ArgumentParser(description='微调攻击测试')
    parser.add_argument('--model_path', type=str, required=True, help='待测试模型pkl文件路径')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='微调轮数')
    parser.add_argument('--finetune_lr', type=float, default=0.001, help='微调学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--key_matrix_dir', type=str, default='save/key_matrix', help='密钥矩阵目录')
    parser.add_argument('--remove_watermark', action='store_true', help='是否在微调前移除水印')
    parser.add_argument('--enable_watermark_reconstruction', action='store_true', 
                       help='启用水印重建和侵权判断功能')
    parser.add_argument('--autoencoder_weights_dir', type=str, default='save/autoencoder',
                       help='自编码器权重目录')
    parser.add_argument('--client_id', type=int, default=0, help='用于水印重建的客户端ID')
    parser.add_argument('--use_deltapcc', action='store_true', default=True,
                       help='使用ΔPCC方法进行侵权判断')
    parser.add_argument('--perf_fail_mse', type=float, default=0.5,
                       help='失效性能的MSE值（容忍下限）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载模型
    model_state, model_args, original_best_acc, original_best_auc = load_model_from_pkl(args.model_path)
    
    # 创建模型
    model = create_model(model_args)
    model.load_state_dict(model_state)
    
    if model_args.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    
    # 加载数据集
    print(f"\n加载数据集: {model_args.dataset}")
    train_loader, val_loader, test_loader = get_dataset(model_args)
    
    # 如果启用水印重建功能，创建自编码器测试数据加载器
    autoencoder_test_loader = None
    if args.enable_watermark_reconstruction:
        print("创建自编码器测试数据加载器...")
        try:
            autoencoder_test_loader = create_test_loader_for_autoencoder(
                batch_size=model_args.batch_size, 
                num_samples=1000
            )
            print("✓ 自编码器测试数据加载器创建成功")
        except Exception as e:
            print(f"❌ 创建自编码器测试数据加载器失败: {e}")
            print("   将禁用水印重建功能")
            args.enable_watermark_reconstruction = False
    
    # 评估原始模型性能
    print(f"\n评估原始模型性能...")
    original_loss, original_acc, original_auc, original_sample_acc = evaluate_model(model, test_loader, model_args)
    print(f"原始模型 - Loss: {original_loss:.4f}, Acc: {original_acc:.2f}%, AUC: {original_auc:.4f}, Sample Acc: {original_sample_acc:.2f}%")
    
    # 加载密钥矩阵管理器（用于水印检测）
    key_manager = None
    if os.path.exists(args.key_matrix_dir):
        try:
            key_manager = KeyMatrixManager(args.key_matrix_dir)
            print(f"成功加载密钥矩阵管理器")
        except Exception as e:
            print(f"加载密钥矩阵管理器失败: {e}")
    
    # 检测原始模型中的水印
    if key_manager is not None:
        has_watermark, watermark_strength = detect_watermark(model, key_manager, 0)  # 假设客户端0
        print(f"原始模型水印检测 - 存在水印: {has_watermark}, 水印强度: {watermark_strength:.4f}")
    
    # 如果启用移除水印选项
    if args.remove_watermark and key_manager is not None:
        print(f"\n移除水印...")
        try:
            # 将水印位置设为0
            model_state_dict = model.state_dict()
            for param_name, param_tensor in model_state_dict.items():
                if param_name in key_manager.load_positions(0):
                    positions = key_manager.load_positions(0)[param_name]
                    flat_param = param_tensor.flatten()
                    for pos in positions:
                        if pos < len(flat_param):
                            flat_param[pos] = 0.0
                    model_state_dict[param_name] = flat_param.reshape(param_tensor.shape)
            model.load_state_dict(model_state_dict)
            print("水印已移除")
        except Exception as e:
            print(f"移除水印失败: {e}")
    
    # 进行微调攻击
    finetune_history = finetune_model(model, train_loader, val_loader, model_args, 
                                    args.finetune_epochs, args.finetune_lr)
    
    # 评估微调后的模型性能
    print(f"\n评估微调后模型性能...")
    final_loss, final_acc, final_auc, final_sample_acc = evaluate_model(model, test_loader, model_args)
    print(f"微调后模型 - Loss: {final_loss:.4f}, Acc: {final_acc:.2f}%, AUC: {final_auc:.4f}, Sample Acc: {final_sample_acc:.2f}%")
    
    # 检测微调后模型中的水印
    if key_manager is not None:
        has_watermark_after, watermark_strength_after = detect_watermark(model, key_manager, 0)
        print(f"微调后模型水印检测 - 存在水印: {has_watermark_after}, 水印强度: {watermark_strength_after:.4f}")
    
    # 水印重建和侵权判断
    watermark_results_after = {}
    if args.enable_watermark_reconstruction and autoencoder_test_loader is not None:
        print(f"\n评估微调后模型的水印重建...")
        try:
            reconstructor = WatermarkReconstructor(args.key_matrix_dir, args.autoencoder_weights_dir)
            
            if args.use_deltapcc:
                # 使用ΔPCC方法
                print("使用ΔPCC方法进行侵权判断...")
                reconstructor.setup_deltapcc_evaluation(autoencoder_test_loader, args.perf_fail_mse)
                
                deltapcc_result = reconstructor.calculate_deltapcc(model.state_dict(), args.client_id)
                
                if deltapcc_result['reconstruction_success']:
                    watermark_results_after = {
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
                    
                    print(f"水印重建成功 - 侵权检测: {'是' if deltapcc_result['infringement_detected'] else '否'}")
                    print(f"ΔPCC值: {deltapcc_result['delta_pcc']:.4f}")
                    print(f"基准性能: {deltapcc_result['perf_before']:.6f}")
                    print(f"测试后性能: {deltapcc_result['perf_after']:.6f}")
                    print(f"性能变化: {deltapcc_result['delta_perf']:.6f}")
                    print(f"阈值τ: {deltapcc_result['tau']:.6f}")
                    print(f"PSNR: {deltapcc_result['psnr']:.2f}")
                    print(f"SSIM: {deltapcc_result['ssim']:.4f}")
                else:
                    print("❌ 水印重建失败")
                    watermark_results_after = {
                        'watermark_reconstruction_success': False,
                        'infringement_detected': False,
                        'method': 'deltapcc'
                    }
            else:
                # 使用传统方法
                print("使用传统方法进行侵权判断...")
                reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_watermark(
                    model.state_dict(), args.client_id
                )
                
                if reconstructed_autoencoder is not None:
                    # 比较性能
                    comparison_results = reconstructor.compare_with_original_autoencoder(
                        reconstructed_autoencoder, autoencoder_test_loader
                    )
                    
                    # 侵权判断
                    infringement_results = reconstructor.assess_infringement(comparison_results)
                    
                    watermark_results_after = {
                        'watermark_reconstruction_success': True,
                        'infringement_detected': infringement_results['overall_infringement'],
                        'psnr_retention': comparison_results['retention']['psnr_retention'],
                        'ssim_retention': comparison_results['retention']['ssim_retention'],
                        'reconstruction_loss': comparison_results['reconstructed']['reconstruction_loss'],
                        'original_reconstruction_loss': comparison_results['original']['reconstruction_loss'],
                        'method': 'traditional'
                    }
                    
                    print(f"水印重建成功 - 侵权检测: {'是' if infringement_results['overall_infringement'] else '否'}")
                    print(f"PSNR保持率: {comparison_results['retention']['psnr_retention']:.2%}")
                    print(f"SSIM保持率: {comparison_results['retention']['ssim_retention']:.2%}")
                else:
                    print("❌ 水印重建失败")
                    watermark_results_after = {
                        'watermark_reconstruction_success': False,
                        'infringement_detected': False,
                        'method': 'traditional'
                    }
        except Exception as e:
            print(f"❌ 水印重建评估失败: {e}")
            watermark_results_after = {
                'watermark_reconstruction_success': False,
                'infringement_detected': False,
                'error': str(e),
                'method': 'deltapcc' if args.use_deltapcc else 'traditional'
            }
    
    # 计算性能变化
    acc_change = final_acc - original_acc
    auc_change = final_auc - original_auc
    sample_acc_change = final_sample_acc - original_sample_acc
    
    print(f"\n性能变化:")
    print(f"准确率变化: {acc_change:+.2f}%")
    print(f"AUC变化: {auc_change:+.4f}")
    print(f"样本准确率变化: {sample_acc_change:+.2f}%")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "save/finetune_attack"
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备结果数据
    results = []
    for epoch_data in finetune_history:
        results.append({
            'epoch': epoch_data['epoch'],
            'train_loss': epoch_data['train_loss'],
            'train_acc': epoch_data['train_acc'],
            'val_loss': epoch_data['val_loss'],
            'val_acc': epoch_data['val_acc'],
            'val_auc': epoch_data['val_auc'],
            'val_sample_acc': epoch_data['val_sample_acc']
        })
    
    # 添加最终测试结果
    final_result = {
        'epoch': 'final_test',
        'train_loss': 0.0,
        'train_acc': 0.0,
        'val_loss': final_loss,
        'val_acc': final_acc,
        'val_auc': final_auc,
        'val_sample_acc': final_sample_acc
    }
    
    # 添加水印重建结果
    if watermark_results_after:
        if watermark_results_after.get('method') == 'deltapcc':
            # ΔPCC方法结果
            final_result.update({
                'watermark_reconstruction_success': watermark_results_after.get('watermark_reconstruction_success', False),
                'infringement_detected': watermark_results_after.get('infringement_detected', False),
                'delta_pcc': watermark_results_after.get('delta_pcc', float('inf')),
                'perf_before': watermark_results_after.get('perf_before', float('inf')),
                'perf_after': watermark_results_after.get('perf_after', float('inf')),
                'perf_fail': watermark_results_after.get('perf_fail', float('inf')),
                'delta_perf': watermark_results_after.get('delta_perf', float('inf')),
                'tau': watermark_results_after.get('tau', float('inf')),
                'psnr': watermark_results_after.get('psnr', 0.0),
                'ssim': watermark_results_after.get('ssim', 0.0),
                'method': 'deltapcc'
            })
        else:
            # 传统方法结果
            final_result.update({
                'watermark_reconstruction_success': watermark_results_after.get('watermark_reconstruction_success', False),
                'infringement_detected': watermark_results_after.get('infringement_detected', False),
                'psnr_retention': watermark_results_after.get('psnr_retention', 0.0),
                'ssim_retention': watermark_results_after.get('ssim_retention', 0.0),
                'reconstruction_loss': watermark_results_after.get('reconstruction_loss', float('inf')),
                'original_reconstruction_loss': watermark_results_after.get('original_reconstruction_loss', float('inf')),
                'method': 'traditional'
            })
    
    results.append(final_result)
    
    # 保存CSV
    df = pd.DataFrame(results)
    csv_filename = f"finetune_attack_{model_args.model}_{model_args.dataset}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 添加模型信息作为注释
    watermark_info = ""
    if args.enable_watermark_reconstruction and watermark_results_after:
        watermark_info = f"""# 水印重建功能: 已启用
# 密钥矩阵目录: {args.key_matrix_dir}
# 自编码器权重目录: {args.autoencoder_weights_dir}
# 客户端ID: {args.client_id}
# 水印重建成功: {watermark_results_after.get('watermark_reconstruction_success', False)}
# 侵权检测: {watermark_results_after.get('infringement_detected', False)}
# PSNR保持率: {watermark_results_after.get('psnr_retention', 0.0):.2%}
# SSIM保持率: {watermark_results_after.get('ssim_retention', 0.0):.2%}
#
"""
    else:
        watermark_info = "# 水印重建功能: 未启用\n"
    
    model_info = f"""# 微调攻击实验结果
# 模型文件: {args.model_path}
# 模型类型: {model_args.model}
# 数据集: {model_args.dataset}
# 设备: {model_args.device}
# 批次大小: {model_args.batch_size}
# 微调轮数: {args.finetune_epochs}
# 微调学习率: {args.finetune_lr}
# 随机种子: {args.seed}
# 移除水印: {args.remove_watermark}
# 原始最佳准确率: {original_best_acc:.4f}
# 原始最佳AUC: {original_best_auc:.4f}
# 原始测试准确率: {original_acc:.2f}%
# 原始测试AUC: {original_auc:.4f}
# 微调后测试准确率: {final_acc:.2f}%
# 微调后测试AUC: {final_auc:.4f}
# 准确率变化: {acc_change:+.2f}%
# AUC变化: {auc_change:+.4f}
# 实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
#
{watermark_info}
"""
    
    # 将注释写入CSV文件开头
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(model_info)
        f.write(content)
    
    print(f"\n结果已保存到: {csv_path}")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"微调攻击测试完成")
    print(f"{'='*80}")
    print(f"模型: {model_args.model} on {model_args.dataset}")
    print(f"微调轮数: {args.finetune_epochs}")
    print(f"微调学习率: {args.finetune_lr}")
    print(f"原始性能: Acc={original_acc:.2f}%, AUC={original_auc:.4f}")
    print(f"微调后性能: Acc={final_acc:.2f}%, AUC={final_auc:.4f}")
    print(f"性能变化: Acc={acc_change:+.2f}%, AUC={auc_change:+.4f}")
    if key_manager is not None:
        print(f"水印状态: 微调前={has_watermark}, 微调后={has_watermark_after}")
        print(f"水印强度变化: {watermark_strength_after - watermark_strength:+.4f}")
    
    # 水印重建和侵权判断总结
    if args.enable_watermark_reconstruction and watermark_results_after:
        print(f"\n水印重建和侵权判断总结:")
        print(f"水印重建成功: {'是' if watermark_results_after.get('watermark_reconstruction_success', False) else '否'}")
        print(f"侵权检测: {'是' if watermark_results_after.get('infringement_detected', False) else '否'}")
        print(f"PSNR保持率: {watermark_results_after.get('psnr_retention', 0.0):.2%}")
        print(f"SSIM保持率: {watermark_results_after.get('ssim_retention', 0.0):.2%}")
        
        # 侵权风险评估
        psnr_retention = watermark_results_after.get('psnr_retention', 0.0)
        ssim_retention = watermark_results_after.get('ssim_retention', 0.0)
        infringement_detected = watermark_results_after.get('infringement_detected', False)
        
        if infringement_detected and psnr_retention >= 0.7 and ssim_retention >= 0.7:
            risk_level = "高风险 (确认侵权)"
        elif psnr_retention >= 0.5 or ssim_retention >= 0.5:
            risk_level = "中风险 (可能侵权)"
        else:
            risk_level = "低风险 (未侵权)"
        
        print(f"侵权风险等级: {risk_level}")

if __name__ == "__main__":
    main()
