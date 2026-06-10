#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印重建模块
用于从水印模型中提取参数，重建自编码器并评估性能
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from models.light_autoencoder import LightAutoencoder
from utils.key_matrix_utils import KeyMatrixManager


class WatermarkReconstructor:
    """水印重建器，用于从水印模型中重建自编码器"""
    
    def __init__(self, key_matrix_dir: str, autoencoder_weights_dir: str = './save/autoencoder', 
                 args=None):
        """
        初始化水印重建器
        
        Args:
            key_matrix_dir: 密钥矩阵目录
            autoencoder_weights_dir: 自编码器权重目录
            args: 参数对象（保留用于兼容性，但不再使用）
        """
        self.key_matrix_dir = key_matrix_dir
        self.autoencoder_weights_dir = autoencoder_weights_dir
        self.key_manager = KeyMatrixManager(key_matrix_dir, args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载原始自编码器作为参考
        self.original_autoencoder = self._load_original_autoencoder()
        
        # ΔPCC相关参数
        self.perf_before = None  # 基准性能 (perf_before)
        self.perf_fail = None    # 失效性能 (perf_fail)
        self.tau = None          # 阈值 τ = loss_fail - loss_before
        self.ds_loader = None    # 专用数据集加载器
        
    def _load_original_autoencoder(self) -> LightAutoencoder:
        """加载原始自编码器作为性能参考"""
        autoencoder = LightAutoencoder().to(self.device)
        
        autoencoder_path = os.path.join(self.autoencoder_weights_dir, 'autoencoder.pth')
        if os.path.exists(autoencoder_path):
            autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device, weights_only=False))
            print(f"✓ 已加载原始自编码器权重: {autoencoder_path}")
        else:
            print(f"⚠️  未找到原始自编码器权重: {autoencoder_path}")
        
        return autoencoder
    
    def extract_watermark_parameters(self, model_state_dict: Dict[str, torch.Tensor],
                                   client_id: int, check_pruning: bool = False,
                                   model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        从模型状态字典中提取水印参数

        Args:
            model_state_dict: 模型状态字典
            client_id: 客户端ID
            check_pruning: 是否检查剪枝影响
            model: 模型对象（推荐传递以确保参数顺序正确）
        """
        try:
            # 必须传递 model 参数以确保参数顺序与密钥矩阵生成时一致
            return self.key_manager.extract_watermark(model_state_dict, client_id, check_pruning, model=model)
        except ValueError as e:
            # 重新抛出 ValueError（包含明确的错误信息）
            raise
        except Exception as e:
            print(f"❌ 提取客户端 {client_id} 的水印参数失败: {e}")
            return torch.tensor([])
    
    def reconstruct_autoencoder_from_watermark(self, model_state_dict: Dict[str, torch.Tensor],
                                             client_id: int,
                                             model: Optional[torch.nn.Module] = None) -> LightAutoencoder:
        """
        从水印参数重建自编码器

        Args:
            model_state_dict: 模型状态字典
            client_id: 客户端ID
            model: 模型对象（推荐传递以确保参数顺序正确）
        """
        watermark_values = self.extract_watermark_parameters(model_state_dict, client_id, model=model)
        
        if len(watermark_values) == 0:
            print(f"❌ 无法从客户端 {client_id} 提取水印参数")
            return None
        
        reconstructed_autoencoder = LightAutoencoder().to(self.device)
        
        # 加载解码器权重
        decoder_path = os.path.join(self.autoencoder_weights_dir, 'decoder.pth')
        if os.path.exists(decoder_path):
            reconstructed_autoencoder.decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device, weights_only=False)
            )
            print(f"✓ 已加载解码器权重: {decoder_path}")
        
        # 重建编码器参数
        self._reconstruct_encoder_from_watermark(reconstructed_autoencoder.encoder, watermark_values)
        
        return reconstructed_autoencoder
    
    def reconstruct_autoencoder_from_all_clients(self, model_state_dict: Dict[str, torch.Tensor],
                                                model: Optional[torch.nn.Module] = None) -> LightAutoencoder:
        """
        从所有客户端的水印参数重建自编码器（用于侵权判断）

        Args:
            model_state_dict: 模型状态字典
            model: 模型对象（推荐传递以确保参数顺序正确）

        Returns:
            重建的自编码器
        """
        # 获取所有客户端ID
        all_client_ids = self.key_manager.list_clients()

        # 从所有客户端提取水印参数
        all_watermark_values = []
        successful_clients = []

        for client_id in all_client_ids:
            try:
                watermark_values = self.extract_watermark_parameters(model_state_dict, client_id, model=model)
                if len(watermark_values) > 0:
                    all_watermark_values.append(watermark_values)
                    successful_clients.append(client_id)
            except Exception as e:
                print(f"   ⚠️  客户端 {client_id} 提取水印失败: {e}")
        
        if not all_watermark_values:
            print("❌ 未能从任何客户端提取到水印参数")
            return None
        
        # 合并所有水印参数
        combined_watermark_values = torch.cat(all_watermark_values)
        
        # 检查参数数量是否匹配编码器
        encoder_params = list(LightAutoencoder().encoder.parameters())
        total_encoder_params = sum(param.numel() for param in encoder_params)
        
        if len(combined_watermark_values) != total_encoder_params:
            if len(combined_watermark_values) > total_encoder_params:
                # 截断多余的参数
                combined_watermark_values = combined_watermark_values[:total_encoder_params]
            else:
                # 填充不足的参数
                padding = torch.zeros(total_encoder_params - len(combined_watermark_values))
                combined_watermark_values = torch.cat([combined_watermark_values, padding])
        
        # 创建新的自编码器
        reconstructed_autoencoder = LightAutoencoder().to(self.device)
        
        # 加载解码器权重（保持不变）
        decoder_path = os.path.join(self.autoencoder_weights_dir, 'decoder.pth')
        if os.path.exists(decoder_path):
            reconstructed_autoencoder.decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device, weights_only=False)
            )
        else:
            print(f"⚠️  未找到解码器权重: {decoder_path}")
        
        # 重建编码器参数
        self._reconstruct_encoder_from_watermark(reconstructed_autoencoder.encoder, combined_watermark_values)
        
        return reconstructed_autoencoder
    
    def _reconstruct_encoder_from_watermark(self, encoder: nn.Module, watermark_values: torch.Tensor):
        """
        从水印值重建编码器参数
        
        Args:
            encoder: 编码器模块
            watermark_values: 水印值
        """
        # 确保水印值在正确的设备上
        device = next(encoder.parameters()).device
        watermark_values = watermark_values.to(device)
        
        # 获取编码器参数信息
        encoder_params = list(encoder.parameters())
        total_params = sum(param.numel() for param in encoder_params)
        
        # 确保水印参数数量与编码器参数数量匹配
        if len(watermark_values) != total_params:
            if len(watermark_values) > total_params:
                watermark_values = watermark_values[:total_params]
            else:
                padding = torch.zeros(total_params - len(watermark_values), device=device)
                watermark_values = torch.cat([watermark_values, padding])
        
        # 将水印值分配给编码器参数
        watermark_idx = 0
        for param in encoder_params:
            param_size = param.numel()
            param_values = watermark_values[watermark_idx:watermark_idx + param_size]
            
            # 重塑参数形状并确保在正确设备上
            param.data = param_values.reshape(param.shape).to(device)
            watermark_idx += param_size
    
    def evaluate_autoencoder_performance(self, autoencoder: LightAutoencoder, 
                                       test_loader) -> Dict[str, float]:
        """
        评估自编码器性能
        
        Args:
            autoencoder: 自编码器模型
            test_loader: 测试数据加载器
            
        Returns:
            性能指标字典
        """
        autoencoder.eval()
        total_loss = 0.0
        total_samples = 0
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                
                # 重建图像
                reconstructed = autoencoder(data)
                
                # 计算重建损失
                loss = criterion(reconstructed, data)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        
        # 计算平均重建损失
        avg_loss = total_loss / total_samples
        
        # 计算重建质量指标
        reconstruction_quality = self._calculate_reconstruction_quality(autoencoder, test_loader)
        
        return {
            'reconstruction_loss': avg_loss,
            'psnr': reconstruction_quality['psnr'],
            'ssim': reconstruction_quality['ssim'],
            'mse': avg_loss
        }
    
    def evaluate_classification_performance(self, model, test_loader) -> Dict[str, float]:
        """
        评估模型在分类任务上的性能（适用于ChestMNIST任务）
        
        Args:
            model: 分类模型（如ResNet18）
            test_loader: 测试数据加载器（ChestMNIST）
            
        Returns:
            分类性能指标字典
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # 获取模型预测
                output = model(data)
                
                # 使用sigmoid激活
                if len(target.shape) == 2 and target.shape[1] > 1:
                    pred_prob = torch.sigmoid(output)
                else:
                    pred_prob = torch.softmax(output, dim=1)
                
                all_predictions.append(pred_prob.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # 合并所有预测和目标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算AUC
        try:
            from sklearn.metrics import roc_auc_score
            if len(all_targets.shape) == 2 and all_targets.shape[1] > 1:
                # 计算每个标签的AUC，然后取平均
                auc_scores = []
                for i in range(all_targets.shape[1]):
                    if len(np.unique(all_targets[:, i])) > 1:  # 确保标签有变化
                        auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
                        auc_scores.append(auc)
                mean_auc = np.mean(auc_scores) if auc_scores else 0.0
            else:
                # 使用one-vs-rest策略
                mean_auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr')
        except ImportError:
            print("警告: sklearn未安装，无法计算AUC，使用准确率代替")
            # 使用准确率作为替代指标
            if len(all_targets.shape) == 2 and all_targets.shape[1] > 1:
                # 使用阈值0.5进行二值化
                pred_binary = (all_predictions > 0.5).astype(int)
                mean_auc = np.mean((pred_binary == all_targets).astype(float))
            else:
                # 使用argmax
                pred_labels = np.argmax(all_predictions, axis=1)
                mean_auc = np.mean((pred_labels == all_targets).astype(float))
        
        return {
            'auc': mean_auc,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def comprehensive_evaluation(self, model_state_dict: Dict[str, torch.Tensor], 
                               client_id: int, 
                               mnist_test_loader,
                               chestmnist_test_loader = None) -> Dict[str, any]:
        """
        综合评估：水印重建质量 + 主任务分类性能
        
        Args:
            model_state_dict: 水印模型状态字典
            client_id: 客户端ID
            mnist_test_loader: MNIST测试数据加载器（用于自编码器评估）
            chestmnist_test_loader: ChestMNIST测试数据加载器（用于主任务评估，可选）
            
        Returns:
            综合评估结果
        """
        results = {
            'client_id': client_id,
            'watermark_reconstruction': None,
            'classification_performance': None,
            'infringement_assessment': None
        }
        
        # 1. 水印重建评估（在MNIST上）
        print(f"🔍 评估客户端 {client_id} 的水印重建质量...")
        reconstructed_autoencoder = self.reconstruct_autoencoder_from_watermark(model_state_dict, client_id)
        
        if reconstructed_autoencoder is not None:
            # 评估重建的自编码器性能
            reconstruction_metrics = self.evaluate_autoencoder_performance(reconstructed_autoencoder, mnist_test_loader)
            
            # 与原始自编码器比较
            original_metrics = self.evaluate_autoencoder_performance(self.original_autoencoder, mnist_test_loader)
            comparison_results = self.compare_autoencoder_performance(original_metrics, reconstruction_metrics)
            
            # 侵权判断
            infringement_results = self.assess_infringement(comparison_results)
            
            results['watermark_reconstruction'] = {
                'reconstructed_metrics': reconstruction_metrics,
                'original_metrics': original_metrics,
                'comparison': comparison_results,
                'infringement': infringement_results
            }
            
            print(f"✅ 水印重建评估完成")
            print(f"   PSNR保持率: {comparison_results['retention']['psnr_retention']:.3f}")
            print(f"   SSIM保持率: {comparison_results['retention']['ssim_retention']:.3f}")
            print(f"   侵权判断: {'是' if infringement_results['overall_infringement'] else '否'}")
        else:
            print(f"❌ 客户端 {client_id} 水印重建失败")
            results['watermark_reconstruction'] = None
        
        # 2. 主任务分类性能评估（在ChestMNIST上，如果提供了数据）
        if chestmnist_test_loader is not None:
            print(f"🔍 评估客户端 {client_id} 的主任务分类性能...")
            
            # 创建主任务模型（需要根据实际情况调整）
            from models.resnet import resnet18
            main_task_model = resnet18(num_classes=7).to(self.device)  # ChestMNIST有7个类别
            
            # 加载模型权重（这里假设model_state_dict就是主任务模型的权重）
            main_task_model.load_state_dict(model_state_dict)
            
            # 评估分类性能
            classification_metrics = self.evaluate_classification_performance(main_task_model, chestmnist_test_loader)
            
            results['classification_performance'] = classification_metrics
            
            print(f"✅ 主任务分类性能评估完成")
            print(f"   AUC: {classification_metrics['auc']:.3f}")
        else:
            print("ℹ️  未提供ChestMNIST测试数据，跳过主任务性能评估")
        
        return results
    
    def _calculate_reconstruction_quality(self, autoencoder: LightAutoencoder, 
                                        test_loader) -> Dict[str, float]:
        """
        计算重建质量指标（PSNR, SSIM等）
        
        Args:
            autoencoder: 自编码器模型
            test_loader: 测试数据加载器
            
        Returns:
            质量指标字典
        """
        autoencoder.eval()
        
        # 获取一小批数据进行质量评估
        data, _ = next(iter(test_loader))
        data = data.to(self.device)
        
        with torch.no_grad():
            reconstructed = autoencoder(data)
            
            # 计算PSNR
            mse = torch.mean((data - reconstructed) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            else:
                psnr = torch.tensor(float('inf'))
            
            # 计算SSIM（简化版本）
            ssim = self._calculate_ssim(data, reconstructed)
        
        return {
            'psnr': psnr.item(),
            'ssim': ssim
        }
    
    def _calculate_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算SSIM（结构相似性指数）
        
        Args:
            x: 原始图像
            y: 重建图像
            
        Returns:
            SSIM值
        """
        # 简化的SSIM计算
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim.item()
    
    def compare_with_original_autoencoder(self, reconstructed_autoencoder: LightAutoencoder, 
                                        test_loader) -> Dict[str, float]:
        """
        将重建的自编码器与原始自编码器进行比较
        
        Args:
            reconstructed_autoencoder: 重建的自编码器
            test_loader: 测试数据加载器
            
        Returns:
            比较结果字典
        """
        # 评估重建的自编码器
        reconstructed_metrics = self.evaluate_autoencoder_performance(
            reconstructed_autoencoder, test_loader
        )
        
        # 评估原始自编码器
        original_metrics = self.evaluate_autoencoder_performance(
            self.original_autoencoder, test_loader
        )
        
        # 计算性能差异
        performance_diff = {
            'reconstruction_loss_diff': reconstructed_metrics['reconstruction_loss'] - original_metrics['reconstruction_loss'],
            'psnr_diff': reconstructed_metrics['psnr'] - original_metrics['psnr'],
            'ssim_diff': reconstructed_metrics['ssim'] - original_metrics['ssim'],
            'mse_diff': reconstructed_metrics['mse'] - original_metrics['mse']
        }
        
        # 计算性能保持率
        performance_retention = {
            'psnr_retention': reconstructed_metrics['psnr'] / original_metrics['psnr'] if original_metrics['psnr'] > 0 else 0,
            'ssim_retention': reconstructed_metrics['ssim'] / original_metrics['ssim'] if original_metrics['ssim'] > 0 else 0
        }
        
        return {
            'reconstructed': reconstructed_metrics,
            'original': original_metrics,
            'difference': performance_diff,
            'retention': performance_retention
        }
    
    def assess_infringement(self, comparison_results: Dict[str, float], 
                          thresholds: Dict[str, float] = None) -> Dict[str, bool]:
        """
        基于自编码器性能评估是否构成侵权
        
        Args:
            comparison_results: 比较结果
            thresholds: 侵权判断阈值
            
        Returns:
            侵权判断结果
        """
        if thresholds is None:
            # 默认阈值
            thresholds = {
                'psnr_retention_min': 0.7,  # PSNR保持率至少70%
                'ssim_retention_min': 0.7,  # SSIM保持率至少70%
                'reconstruction_loss_max_ratio': 2.0,  # 重建损失不超过原始损失的2倍
            }
        
        retention = comparison_results['retention']
        diff = comparison_results['difference']
        original = comparison_results['original']
        
        # 侵权判断标准
        infringement_criteria = {
            'psnr_retention_adequate': retention['psnr_retention'] >= thresholds['psnr_retention_min'],
            'ssim_retention_adequate': retention['ssim_retention'] >= thresholds['ssim_retention_min'],
            'reconstruction_loss_acceptable': (
                diff['reconstruction_loss_diff'] / original['reconstruction_loss'] 
                <= thresholds['reconstruction_loss_max_ratio']
            ),
        }
        
        # 综合判断
        infringement_criteria['overall_infringement'] = all(infringement_criteria.values())
        
        return infringement_criteria
    
    def setup_deltapcc_evaluation(self, ds_loader, perf_fail_ratio: float = 0.1):
        """
        设置ΔPCC评估参数
        
        Args:
            ds_loader: 专用数据集加载器 (Ds)
            perf_fail_ratio: 失效性能比例（相对于基准性能的倍数）
        """
        self.ds_loader = ds_loader
        
        # 计算基准性能 perf_before
        print("计算基准性能 perf_before...")
        self.perf_before = self.evaluate_autoencoder_performance(
            self.original_autoencoder, ds_loader
        )['mse']
        
        # 使用相对阈值计算方式（与剪枝攻击一致）
        self.perf_fail = self.perf_before * (1 + perf_fail_ratio)
        self.tau = self.perf_fail - self.perf_before
        
        print(f"ΔPCC评估参数设置完成:")
        print(f"  基准性能 (perf_before): {self.perf_before:.6f}")
        print(f"  失效性能 (perf_fail): {self.perf_fail:.6f}")
        print(f"  阈值 (τ): {self.tau:.6f}")
        
        if self.tau <= 0:
            print("⚠️  警告: 阈值τ <= 0，这可能导致ΔPCC计算异常")
    
    def evaluate_encoder_decoder_separated(self, model_state_dict: Dict[str, torch.Tensor], 
                                         client_id: int, test_loader) -> Dict[str, float]:
        """
        分离评估编码器和解码器（参考代码风格）
        编码器：从水印模型提取（可能被攻击）
        解码器：始终使用原始预训练权重
        
        Args:
            model_state_dict: 待测模型状态字典
            client_id: 客户端ID
            test_loader: 测试数据加载器
            
        Returns:
            评估结果
        """
        # 提取水印参数
        watermark_values = self.extract_watermark_parameters(model_state_dict, client_id)
        
        if len(watermark_values) == 0:
            return {
                'mse': float('inf'),
                'ssim': 0.0,
                'psnr': 0.0,
                'reconstruction_success': False,
                'watermark_damaged': False
            }
        
        # 创建新的自编码器
        autoencoder = LightAutoencoder().to(self.device)
        
        # 加载原始解码器权重（始终不变）
        decoder_path = os.path.join(self.autoencoder_weights_dir, 'decoder.pth')
        if os.path.exists(decoder_path):
            autoencoder.decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device, weights_only=False)
            )
        else:
            print(f"❌ 解码器权重文件不存在: {decoder_path}")
            return {
                'mse': float('inf'),
                'ssim': 0.0,
                'psnr': 0.0,
                'reconstruction_success': False,
                'watermark_damaged': False
            }
        
        # 重建编码器参数（从水印模型提取）
        self._reconstruct_encoder_from_watermark(autoencoder.encoder, watermark_values)
        
        # 评估性能
        metrics = self.evaluate_autoencoder_performance(autoencoder, test_loader)
        
        return {
            'mse': metrics['mse'],
            'ssim': metrics['ssim'],
            'psnr': metrics['psnr'],
            'reconstruction_success': True,
            'watermark_damaged': False
        }

    def calculate_deltapcc(self, model_state_dict: Dict[str, torch.Tensor], 
                          client_id: int, check_pruning: bool = True) -> Dict[str, float]:
        """
        计算ΔPCC值
        
        Args:
            model_state_dict: 待测模型状态字典
            client_id: 客户端ID
            check_pruning: 是否检查剪枝对水印的影响
            
        Returns:
            ΔPCC计算结果
        """
        if self.ds_loader is None or self.tau is None:
            raise ValueError("请先调用 setup_deltapcc_evaluation() 设置评估参数")
        
        # 检查水印是否被剪枝破坏
        watermark_values = self.extract_watermark_parameters(model_state_dict, client_id, check_pruning)
        
        # 如果启用了剪枝检查，检测水印完整性
        watermark_damaged = False
        if check_pruning and len(watermark_values) > 0:
            # 检查水印值是否完全等于0（被剪枝）
            damaged_count = (watermark_values == 0.0).sum().item()
            total_watermark_count = len(watermark_values)
            watermark_damaged = damaged_count > 0
            
            if watermark_damaged:
                print(f"⚠️  检测到水印被剪枝破坏: {damaged_count}/{total_watermark_count} 个水印位置被剪掉")
                # 注意：水印破坏信息仅用于记录，不影响侵权判断
                # 侵权判断将完全基于PCC值
        
        # 从待测模型重建自编码器
        reconstructed_autoencoder = self.reconstruct_autoencoder_from_watermark(
            model_state_dict, client_id
        )
        
        if reconstructed_autoencoder is None:
            return {
                'delta_pcc': float('inf'),
                'perf_after': float('inf'),
                'delta_perf': float('inf'),
                'infringement_detected': False,
                'reconstruction_success': False,
                'watermark_damaged': watermark_damaged
            }
        
        # 在专用数据集上测试重建后的性能
        perf_after_metrics = self.evaluate_autoencoder_performance(
            reconstructed_autoencoder, self.ds_loader
        )
        perf_after = perf_after_metrics['mse']
        
        # 计算性能变化 Δperf = |perf_after - perf_before|
        delta_perf = abs(perf_after - self.perf_before)
        
        # 计算ΔPCC = Δperf / τ
        delta_pcc = delta_perf / self.tau if self.tau > 0 else float('inf')
        
        # 侵权判断: 只基于ΔPCC < 1 表示侵权
        infringement_detected = delta_pcc < 1.0
        
        return {
            'delta_pcc': delta_pcc,
            'perf_before': self.perf_before,
            'perf_after': perf_after,
            'perf_fail': self.perf_fail,
            'delta_perf': delta_perf,
            'tau': self.tau,
            'infringement_detected': infringement_detected,
            'reconstruction_success': True,
            'watermark_damaged': watermark_damaged,
            'damaged_ratio': damaged_count / total_watermark_count if watermark_damaged else 0.0,
            'psnr': perf_after_metrics['psnr'],
            'ssim': perf_after_metrics['ssim']
        }
    
    def batch_evaluate_deltapcc(self, model_state_dicts: List[Dict[str, torch.Tensor]], 
                               client_ids: List[int]) -> Dict[str, List[float]]:
        """
        批量评估多个模型的ΔPCC
        
        Args:
            model_state_dicts: 模型状态字典列表
            client_ids: 客户端ID列表
            
        Returns:
            批量评估结果
        """
        if self.ds_loader is None or self.tau is None:
            raise ValueError("请先调用 setup_deltapcc_evaluation() 设置评估参数")
        
        results = {
            'delta_pcc': [],
            'perf_before': [],
            'perf_after': [],
            'perf_fail': [],
            'delta_perf': [],
            'tau': [],
            'infringement_detected': [],
            'reconstruction_success': [],
            'psnr': [],
            'ssim': []
        }
        
        print(f"批量评估 {len(model_state_dicts)} 个模型的ΔPCC...")
        
        for i, (model_state_dict, client_id) in enumerate(zip(model_state_dicts, client_ids)):
            print(f"评估模型 {i+1}/{len(model_state_dicts)} (客户端 {client_id})...")
            
            try:
                result = self.calculate_deltapcc(model_state_dict, client_id)
                
                for key, value in result.items():
                    results[key].append(value)
                    
            except Exception as e:
                print(f"❌ 模型 {i+1} 评估失败: {e}")
                # 添加失败结果
                for key in results.keys():
                    if key == 'infringement_detected' or key == 'reconstruction_success':
                        results[key].append(False)
                    else:
                        results[key].append(float('inf'))
        
        return results
    
    def analyze_deltapcc_results(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """
        分析ΔPCC评估结果
        
        Args:
            results: 批量评估结果
            
        Returns:
            分析结果统计
        """
        delta_pcc_values = [v for v in results['delta_pcc'] if v != float('inf')]
        infringement_detected = results['infringement_detected']
        reconstruction_success = results['reconstruction_success']
        
        if not delta_pcc_values:
            return {
                'total_models': len(results['delta_pcc']),
                'successful_reconstructions': 0,
                'infringement_rate': 0.0,
                'avg_delta_pcc': float('inf'),
                'min_delta_pcc': float('inf'),
                'max_delta_pcc': float('inf'),
                'std_delta_pcc': float('inf')
            }
        
        successful_reconstructions = sum(reconstruction_success)
        infringement_count = sum(infringement_detected)
        
        return {
            'total_models': len(results['delta_pcc']),
            'successful_reconstructions': successful_reconstructions,
            'reconstruction_success_rate': successful_reconstructions / len(results['delta_pcc']),
            'infringement_count': infringement_count,
            'infringement_rate': infringement_count / successful_reconstructions if successful_reconstructions > 0 else 0.0,
            'avg_delta_pcc': np.mean(delta_pcc_values),
            'min_delta_pcc': np.min(delta_pcc_values),
            'max_delta_pcc': np.max(delta_pcc_values),
            'std_delta_pcc': np.std(delta_pcc_values),
            'median_delta_pcc': np.median(delta_pcc_values)
        }


def create_test_loader_for_autoencoder(batch_size: int = 128, num_samples: int = 1000):
    """
    为自编码器创建测试数据加载器
    
    Args:
        batch_size: 批大小
        num_samples: 样本数量
        
    Returns:
        测试数据加载器
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    
    # 使用MNIST数据集作为自编码器的测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载MNIST测试集
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 限制样本数量
    if num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_dataset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def test_watermark_reconstruction(key_matrix_dir=None, autoencoder_weights_dir=None):
    """测试水印重建功能"""
    print("测试水印重建功能...")
    
    # 检查必要的文件
    if key_matrix_dir is None:
        # 使用环境变量或默认路径
        key_matrix_dir = os.environ.get('KEY_MATRIX_DIR', './save/key_matrix/resnet/client10')
    if autoencoder_weights_dir is None:
        autoencoder_weights_dir = os.environ.get('AUTOENCODER_DIR', './save/autoencoder')
    
    if not os.path.exists(key_matrix_dir):
        print(f"❌ 密钥矩阵目录不存在: {key_matrix_dir}")
        return
    
    if not os.path.exists(autoencoder_weights_dir):
        print(f"❌ 自编码器权重目录不存在: {autoencoder_weights_dir}")
        return
    
    # 创建水印重建器
    reconstructor = WatermarkReconstructor(key_matrix_dir, autoencoder_weights_dir)
    
    # 创建测试数据加载器
    test_loader = create_test_loader_for_autoencoder()
    
    # 设置ΔPCC评估参数
    print("设置ΔPCC评估参数...")
    reconstructor.setup_deltapcc_evaluation(test_loader, perf_fail_ratio=0.1)
    
    # 模拟一个水印模型（这里使用随机参数）
    print("创建模拟水印模型...")
    from models.resnet import resnet18
    model = resnet18(num_classes=14, in_channels=1, input_size=28)
    model_state_dict = model.state_dict()
    
    # 测试ΔPCC计算
    client_id = 0
    print(f"测试客户端 {client_id} 的ΔPCC计算...")
    
    deltapcc_result = reconstructor.calculate_deltapcc(model_state_dict, client_id)
    
    if deltapcc_result['reconstruction_success']:
        print("✓ ΔPCC计算成功")
        
        # 简化的ΔPCC评估结果
        infringement_status = "侵权" if deltapcc_result['infringement_detected'] else "未侵权"
        print(f"ΔPCC: {deltapcc_result['delta_pcc']:.3f} | 侵权判断: {infringement_status}")
        
    else:
        print("❌ ΔPCC计算失败")
    
    # 测试传统方法对比
    print("\n=== 传统方法对比 ===")
    reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_watermark(
        model_state_dict, client_id
    )
    
    if reconstructed_autoencoder is not None:
        comparison_results = reconstructor.compare_with_original_autoencoder(
            reconstructed_autoencoder, test_loader
        )
        
        print(f"重建自编码器 - PSNR: {comparison_results['reconstructed']['psnr']:.2f}, "
              f"SSIM: {comparison_results['reconstructed']['ssim']:.4f}")
        print(f"原始自编码器 - PSNR: {comparison_results['original']['psnr']:.2f}, "
              f"SSIM: {comparison_results['original']['ssim']:.4f}")
        
        print(f"\n性能保持率:")
        print(f"PSNR保持率: {comparison_results['retention']['psnr_retention']:.2%}")
        print(f"SSIM保持率: {comparison_results['retention']['ssim_retention']:.2%}")
        
        # 传统侵权判断
        infringement_results = reconstructor.assess_infringement(comparison_results)
        print(f"\n传统侵权判断: {'侵权' if infringement_results['overall_infringement'] else '未侵权'}")
        
    else:
        print("❌ 传统方法重建失败")


def test_deltapcc_batch_evaluation():
    """测试ΔPCC批量评估功能"""
    print("\n测试ΔPCC批量评估功能...")
    
    # 检查必要的文件
    key_matrix_dir = os.environ.get('KEY_MATRIX_DIR', './save/key_matrix/resnet/client10')
    autoencoder_weights_dir = os.environ.get('AUTOENCODER_DIR', './save/autoencoder')
    
    if not os.path.exists(key_matrix_dir):
        print(f"❌ 密钥矩阵目录不存在: {key_matrix_dir}")
        return
    
    if not os.path.exists(autoencoder_weights_dir):
        print(f"❌ 自编码器权重目录不存在: {autoencoder_weights_dir}")
        return
    
    # 创建水印重建器
    reconstructor = WatermarkReconstructor(key_matrix_dir, autoencoder_weights_dir)
    
    # 创建测试数据加载器
    test_loader = create_test_loader_for_autoencoder()
    
    # 设置ΔPCC评估参数
    reconstructor.setup_deltapcc_evaluation(test_loader, perf_fail_ratio=0.1)
    
    # 创建多个模拟模型
    print("创建多个模拟水印模型...")
    from models.resnet import resnet18
    model_state_dicts = []
    client_ids = list(range(5))  # 测试5个客户端
    
    for i in range(5):
        model = resnet18(num_classes=14, in_channels=1, input_size=28)
        model_state_dicts.append(model.state_dict())
    
    # 批量评估
    results = reconstructor.batch_evaluate_deltapcc(model_state_dicts, client_ids)
    
    # 分析结果
    analysis = reconstructor.analyze_deltapcc_results(results)
    
    print("\n=== 批量评估结果分析 ===")
    print(f"总模型数: {analysis['total_models']}")
    print(f"成功重建数: {analysis['successful_reconstructions']}")
    print(f"重建成功率: {analysis['reconstruction_success_rate']:.2%}")
    print(f"侵权检测数: {analysis['infringement_count']}")
    print(f"侵权检测率: {analysis['infringement_rate']:.2%}")
    print(f"平均ΔPCC: {analysis['avg_delta_pcc']:.6f}")
    print(f"最小ΔPCC: {analysis['min_delta_pcc']:.6f}")
    print(f"最大ΔPCC: {analysis['max_delta_pcc']:.6f}")
    print(f"ΔPCC标准差: {analysis['std_delta_pcc']:.6f}")
    print(f"ΔPCC中位数: {analysis['median_delta_pcc']:.6f}")


if __name__ == '__main__':
    test_watermark_reconstruction()
    test_deltapcc_batch_evaluation()
