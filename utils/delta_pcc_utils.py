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
            
            # 检查自编码器期望的输入通道数
            if autoencoder.input_channels == 1 and data.size(1) == 3:
                # 自编码器期望单通道，但数据是3通道，转换为灰度图
                data = torch.mean(data, dim=1, keepdim=True)
            elif autoencoder.input_channels == 3 and data.size(1) == 1:
                # 自编码器期望3通道，但数据是单通道，复制为3通道
                data = data.repeat(1, 3, 1, 1)
            
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
        
        # 🔍 调试：检查提取的水印参数是否变化
        import hashlib
        # 使用第一个客户端进行调试（客户端ID从0开始）
        debug_client_id = 0
        orig_wm = reconstructor.key_manager.extract_watermark(original_model_state, client_id=debug_client_id)
        curr_wm = reconstructor.key_manager.extract_watermark(current_model_state, client_id=debug_client_id)
        
        def get_tensor_hash(tensor):
            if len(tensor) == 0:
                return "empty"
            return hashlib.md5(f"{tensor.sum().item()}_{tensor.std().item()}".encode()).hexdigest()[:8]
        
        orig_wm_hash = get_tensor_hash(orig_wm)
        curr_wm_hash = get_tensor_hash(curr_wm)
        print(f"    🔍 水印参数哈希: 原始={orig_wm_hash}, 当前={curr_wm_hash}, 相同={orig_wm_hash==curr_wm_hash}")
        print(f"    🔍 水印参数统计: 原始sum={orig_wm.sum().item():.6f}, 当前sum={curr_wm.sum().item():.6f}")
        print(f"    🔍 水印参数零值: 原始={(orig_wm==0).sum().item()}/{len(orig_wm)}, 当前={(curr_wm==0).sum().item()}/{len(curr_wm)}")
        print(f"    🔍 水印参数差异: max_diff={torch.abs(curr_wm - orig_wm).max().item():.9f}, mean_diff={torch.abs(curr_wm - orig_wm).mean().item():.9f}")
        
        # 5. 测试重建自编码器的性能
        perf_after = test_autoencoder_mse(current_reconstructed_autoencoder, test_loader, device)
        
        # 🔍 调试：直接比较两个自编码器的编码器参数
        orig_encoder_params = torch.cat([p.view(-1) for p in original_reconstructed_autoencoder.encoder.parameters()])
        curr_encoder_params = torch.cat([p.view(-1) for p in current_reconstructed_autoencoder.encoder.parameters()])
        encoder_diff = torch.abs(orig_encoder_params - curr_encoder_params)
        print(f"    🔍 自编码器编码器参数: 原始sum={orig_encoder_params.sum().item():.6f}, 当前sum={curr_encoder_params.sum().item():.6f}")
        print(f"    🔍 编码器参数差异: max={encoder_diff.max().item():.9f}, mean={encoder_diff.mean().item():.9f}, 非零差异={(encoder_diff > 1e-9).sum().item()}/{len(encoder_diff)}")
        
        # 🔍 检查被剪枝的水印参数的值
        zero_mask = (curr_wm == 0) & (orig_wm != 0)
        if zero_mask.sum() > 0:
            pruned_values = orig_wm[zero_mask]
            print(f"    🔍 被剪枝的{zero_mask.sum().item()}个水印参数: mean={pruned_values.mean().item():.9f}, max={pruned_values.abs().max().item():.9f}, min={pruned_values.abs().min().item():.9f}")
        
        # 🔍 调试：检查模型状态字典是否真的不同
        import hashlib
        def get_state_hash(state_dict):
            """计算状态字典的哈希值"""
            # 将所有参数连接成一个字符串并计算哈希
            param_str = ''.join([f"{k}:{v.sum().item()}" for k, v in state_dict.items()])
            return hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        orig_hash = get_state_hash(original_model_state)
        curr_hash = get_state_hash(current_model_state)
        print(f"    🔍 模型状态哈希: 原始={orig_hash}, 当前={curr_hash}, 是否相同={orig_hash==curr_hash}")
        
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
