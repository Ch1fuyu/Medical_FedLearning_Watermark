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


def load_mnist_test_data(batch_size: int = 128, data_dir: str = './data'):
    """
    加载MNIST测试数据，使用与训练时相同的数据预处理策略
    
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ 已加载MNIST测试集: {len(test_dataset)} 个样本")
    return test_loader

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
            
            # 计算MSE loss (使用sum reduction，然后手动平均)
            mse_loss = torch.nn.functional.mse_loss(reconstructed, data, reduction='sum')
            total_mse += mse_loss.item()
            total_samples += data.size(0)  # 累加样本数
    
    avg_mse = total_mse / total_samples
    return avg_mse

def evaluate_delta_pcc(original_model, pruned_model, reconstructor, test_loader, device: str = 'cuda', 
                      perf_fail_ratio: float = 0.1):
    """
    评估ΔPCC（性能变化百分比）
    
    Args:
        original_model: 剪枝前的模型（嵌入水印但未剪枝）
        pruned_model: 剪枝后的模型
        reconstructor: 水印重建器实例
        test_loader: MNIST测试数据加载器
        device: 设备类型
        perf_fail_ratio: 失效性能比例（相对于基准性能的倍数）
        
    Returns:
        包含ΔPCC评估结果的字典
    """
    try:
        # 1. 从剪枝前的模型重建自编码器作为基准
        original_model_state_dict = original_model.state_dict()
        original_reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(original_model_state_dict)
        
        if original_reconstructed_autoencoder is None:
            print("❌ 无法从剪枝前模型重建自编码器")
            return None
        
        # 2. 测试剪枝前重建自编码器的基准性能
        perf_before = test_autoencoder_mse(original_reconstructed_autoencoder, test_loader, device)
        
        # 3. 计算阈值τ（性能下降容忍度）
        # 对于MSE损失，性能下降意味着损失增加，所以失效性能应该比基准性能大
        perf_fail = perf_before * (1 + perf_fail_ratio)
        tau = perf_fail - perf_before
        
        # 4. 从剪枝后的模型重建自编码器
        pruned_model_state_dict = pruned_model.state_dict()
        
        # 重建自编码器
        reconstructed_autoencoder = reconstructor.reconstruct_autoencoder_from_all_clients(pruned_model_state_dict)
        
        if reconstructed_autoencoder is None:
            print("❌ 自编码器重建失败")
            return None
        
        # 5. 测试重建自编码器的性能
        perf_after = test_autoencoder_mse(reconstructed_autoencoder, test_loader, device)
        
        # 6. 计算性能变化
        delta_perf = abs(perf_after - perf_before)
        
        # 调试信息
        print(f"    性能: 剪枝前={perf_before:.6f}, 剪枝后={perf_after:.6f}, 变化={delta_perf:.6f}")
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
    model.eval()
    
    return model

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

def save_pruning_results(results, save_dir='./save/pruning_results'):
    """
    保存剪枝攻击实验结果
    
    Args:
        results: 实验结果列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细结果（pickle格式）
    results_file = os.path.join(save_dir, f'pruning_attack_results_{timestamp}.pkl')
    torch.save(results, results_file)
    
    # 保存CSV格式的简化结果
    csv_file = os.path.join(save_dir, f'pruning_attack_summary_{timestamp}.csv')
    
    df_data = []
    for result in results:
        df_data.append({
            'pruning_ratio': result['pruning_ratio'],
            'watermark_integrity': result['watermark_integrity'],
            'total_watermark_params': result['total_watermark_params'],
            'damaged_watermark_params': result['damaged_watermark_params'],
            'perf_before': result.get('perf_before', float('inf')),
            'perf_after': result.get('perf_after', float('inf')),
            'delta_pcc': result.get('delta_pcc', float('inf')),
            'is_infringement': result.get('is_infringement', False),
            'result_text': result.get('result_text', '评估失败')
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
    # 模型路径
    model_path = './save/resnet/chestmnist/202510091107_Dp_False_sig_0.1_iid_True_ns_1_wt_gamma_lt_sign_bit_20_alp_0.2_nb_1_type_True_tri_40_ep_10_le_2_cn_10_fra_1.0000_acc_0.5892_enhanced.pkl'
    
    # 密钥矩阵目录
    key_matrix_dir = './save/key_matrix'
    autoencoder_dir = './save/autoencoder'
    
    # 加载MNIST测试数据
    test_loader = load_mnist_test_data(batch_size=128)
    
    # 加载主任务模型
    model = load_main_task_model(model_path, device)
    
    if model is not None:
        print(f"开始剪枝攻击实验 (设备: {device})")
        
        # 初始化水印重建器（只初始化一次）
        reconstructor = WatermarkReconstructor(key_matrix_dir, autoencoder_dir)
        
        # 定义剪枝比例：从0%到100%，步长10%
        pruning_ratios = [i/10.0 for i in range(0, 11)]  # [0.0, 0.1, 0.2, ..., 1.0]
        
        # 存储实验结果
        results = []
        
        for ratio in pruning_ratios:
            
            # 对模型进行剪枝
            pruned_model = threshold_pruning(model, ratio)
            
            
            # 评估水印完整性
            watermark_result = evaluate_watermark_after_pruning(pruned_model, reconstructor)
            
            # 评估ΔPCC
            # 使用原始模型作为基准，比较剪枝前后的性能
            delta_pcc_result = evaluate_delta_pcc(
                model, pruned_model, reconstructor, test_loader, device, 
                perf_fail_ratio=0.1
            )
            
            # 记录结果
            result = {
                'pruning_ratio': ratio
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
            
            if delta_pcc_result is not None:
                result.update({
                    'perf_before': delta_pcc_result['perf_before'],
                    'perf_after': delta_pcc_result['perf_after'],
                    'delta_pcc': delta_pcc_result['delta_pcc'],
                    'is_infringement': delta_pcc_result['is_infringement'],
                    'result_text': delta_pcc_result['result_text']
                })
            else:
                result.update({
                    'perf_before': float('inf'),
                    'perf_after': float('inf'),
                    'delta_pcc': float('inf'),
                    'is_infringement': False,
                    'result_text': '评估失败'
                })
            
            results.append(result)
            
            # 简化的实验结果输出
            print(f"剪枝{ratio:.0%}: 水印完整性{result['watermark_integrity']:.2%} | ΔPCC{result['delta_pcc']:.6f} | {result['result_text']}")
        
        # 简化的总结表格
        print(f"\n{'='*80}")
        print("实验结果总结")
        print(f"{'='*80}")
        print(f"{'剪枝%':<6} {'水印完整性%':<13} {'ΔPCC':<9} {'侵权判断':<8}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['pruning_ratio']:>4.0%} "
                  f"{result['watermark_integrity']:>11.2%} {result['delta_pcc']:>9.6f} "
                  f"{result['result_text']:>6}")
        
        # 保存实验结果
        print("\n保存实验结果...")
        save_pruning_results(results)
            
    else:
        print("主任务模型加载失败")

if __name__ == '__main__':
    main()
