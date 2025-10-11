"""
ΔPCC (Delta Performance Change Coefficient) 计算工具

统一支持微调和剪枝实验的ΔPCC计算函数
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union


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
            reconstructed = autoencoder(data)
            mse = nn.functional.mse_loss(reconstructed, data)
            total_mse += mse.item() * data.size(0)
            total_samples += data.size(0)
    
    avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')
    return avg_mse


def evaluate_delta_pcc(original_model_state, current_model_state, reconstructor, 
                      test_loader, device: str = 'cuda', perf_fail_ratio: float = 0.1, 
                      fixed_tau: Optional[float] = None):
    """
    评估ΔPCC（性能变化百分比）- 统一版本
    
    支持微调和剪枝实验的ΔPCC计算，可以同时使用固定阈值和动态阈值
    
    Args:
        original_model_state: 原始模型状态（用于重建基准自编码器）
        current_model_state: 当前模型状态（微调后或剪枝后的模型状态）
        reconstructor: 水印重建器实例
        test_loader: MNIST测试数据加载器
        device: 设备类型
        perf_fail_ratio: 失效性能比例（仅在fixed_tau为None时使用）
        fixed_tau: 固定阈值τ（如果提供，则使用此值而不是动态计算）
        
    Returns:
        dict: 包含ΔPCC评估结果的字典，如果失败返回None
    """
    try:
        # 1. 从原始模型重建自编码器作为基准
        original_reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(original_model_state)
        
        if original_reconstructed_autoencoder is None:
            print("❌ 无法从原始模型重建自编码器")
            return None
        
        # 2. 测试原始重建自编码器的基准性能
        perf_before = test_autoencoder_mse(original_reconstructed_autoencoder, test_loader, device)
        
        # 3. 计算阈值τ和失效性能
        if fixed_tau is not None:
            # 使用固定阈值
            tau = fixed_tau
            perf_fail = perf_before + tau
        else:
            # 使用动态阈值计算
            # 对于MSE损失，性能下降意味着损失增加，所以失效性能应该比基准性能大
            perf_fail = perf_before * (1 + perf_fail_ratio)
            tau = perf_fail - perf_before
        
        # 4. 从当前模型重建自编码器
        current_reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(current_model_state)
        
        if current_reconstructed_autoencoder is None:
            print("❌ 自编码器重建失败")
            return None
        
        # 5. 测试重建自编码器的性能
        perf_after = test_autoencoder_mse(current_reconstructed_autoencoder, test_loader, device)
        
        # 6. 计算性能变化
        delta_perf = abs(perf_after - perf_before)
        
        # 调试信息
        print(f"    性能: 原始={perf_before:.6f}, 当前={perf_after:.6f}, 变化={delta_perf:.6f}")
        print(f"    阈值τ={tau:.6f}, 失效性能={perf_fail:.6f}")
        
        # 7. 计算ΔPCC
        if tau > 0:
            delta_pcc = delta_perf / tau
        else:
            delta_pcc = float('inf')
        
        # 8. 判断侵权
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
        print(f"❌ ΔPCC评估失败: {e}")
        return None


def calculate_fixed_tau(original_model_state, reconstructor, test_loader, 
                       device: str = 'cuda', perf_fail_ratio: float = 0.1):
    """
    计算固定阈值τ（基于原始模型）
    
    Args:
        original_model_state: 原始模型状态
        reconstructor: 水印重建器实例
        test_loader: MNIST测试数据加载器
        device: 设备类型
        perf_fail_ratio: 失效性能比例
        
    Returns:
        float: 固定阈值τ，如果计算失败返回None
    """
    try:
        # 从原始模型重建自编码器
        original_reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(original_model_state)
        
        if original_reconstructed_autoencoder is None:
            print("❌ 无法从原始模型重建自编码器")
            return None
        
        # 测试基准性能
        perf_before = test_autoencoder_mse(original_reconstructed_autoencoder, test_loader, device)
        
        # 计算固定阈值
        perf_fail = perf_before * (1 + perf_fail_ratio)
        fixed_tau = perf_fail - perf_before
        
        print(f"✓ 固定阈值τ={fixed_tau:.6f} (基准性能={perf_before:.6f})")
        return fixed_tau
        
    except Exception as e:
        print(f"❌ 固定阈值计算失败: {e}")
        return None


def format_delta_pcc_result(delta_pcc_result: Optional[Dict], default_values: Optional[Dict] = None):
    """
    格式化ΔPCC结果为标准格式
    
    Args:
        delta_pcc_result: ΔPCC评估结果
        default_values: 默认值字典
        
    Returns:
        dict: 格式化的结果字典
    """
    if default_values is None:
        default_values = {
            'perf_before': None,
            'perf_after': None,
            'perf_fail': None,
            'tau': None,
            'delta_perf': None,
            'delta_pcc': None,
            'is_infringement': None,
            'result_text': 'N/A'
        }
    
    if delta_pcc_result is not None:
        return {
            'perf_before': delta_pcc_result['perf_before'],
            'perf_after': delta_pcc_result['perf_after'],
            'perf_fail': delta_pcc_result['perf_fail'],
            'tau': delta_pcc_result['tau'],
            'delta_perf': delta_pcc_result['delta_perf'],
            'delta_pcc': delta_pcc_result['delta_pcc'],
            'is_infringement': delta_pcc_result['is_infringement'],
            'result_text': delta_pcc_result['result_text']
        }
    else:
        return default_values.copy()


def print_delta_pcc_summary(delta_pcc_result: Optional[Dict], prefix: str = ""):
    """
    打印ΔPCC结果摘要
    
    Args:
        delta_pcc_result: ΔPCC评估结果
        prefix: 输出前缀
    """
    if delta_pcc_result is not None:
        print(f"{prefix}性能基准: {delta_pcc_result['perf_before']:.6f} | "
              f"性能变化: {delta_pcc_result['delta_perf']:.6f}")
        print(f"{prefix}ΔPCC: {delta_pcc_result['delta_pcc']:.6f} | "
              f"侵权判断: {delta_pcc_result['result_text']}")
    else:
        print(f"{prefix}ΔPCC: N/A | 侵权判断: N/A")
