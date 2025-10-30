"""
自编码器微调模块
用于在联邦学习过程中微调自编码器的编码器部分
注意：只在内存中更新编码器参数，不修改原始.pth文件
解码器参数保持不变，由第三方保管
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import gc
import os



class AutoencoderFinetuner:
    """自编码器微调器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._mnist_dataset = None
        self._mnist_loader = None
        
    def _get_mnist_loader(self, batch_size=32):  # 减少默认批处理大小
        """获取MNIST数据加载器，使用缓存避免重复加载"""
        if self._mnist_loader is None:
            try:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                # 使用MNIST训练集进行微调，确保只下载一次
                self._mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                self._mnist_loader = torch.utils.data.DataLoader(
                    self._mnist_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    num_workers=0,  # 避免多进程问题
                    pin_memory=False,  # 减少内存使用
                    drop_last=True,  # 丢弃最后一个不完整的batch
                    persistent_workers=False  # 避免内存泄漏
                )
                pass  # 静默加载
            except Exception as e:
                print(f"⚠️ 加载MNIST数据集失败: {e}")
                return None
        
        return self._mnist_loader
    
    def _check_gpu_memory(self):
        """检查GPU内存使用情况"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                max_allocated = torch.cuda.max_memory_allocated()
                if max_allocated > 0:
                    return allocated / max_allocated
                else:
                    # 如果max_memory_allocated为0，使用reserved_memory
                    reserved = torch.cuda.memory_reserved()
                    max_reserved = torch.cuda.max_memory_reserved()
                    return reserved / max_reserved if max_reserved > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    def _force_cleanup(self):
        """强制清理内存"""
        try:
            # 清理Python垃圾回收
            gc.collect()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 内存清理完成
        except Exception as e:
            print(f"⚠️ 内存清理失败: {e}")

    def finetune_encoder(self, autoencoder, epochs=1, lr=0.005, batch_size=32):  # 减少默认批处理大小
        """
        微调自编码器的编码器部分
        
        Args:
            autoencoder: 自编码器模型
            epochs: 微调轮数
            lr: 学习率
            batch_size: 批次大小
            
        Returns:
            bool: 微调是否成功
        """
        if autoencoder is None:
            print("⚠️ 自编码器模型为空，无法进行微调")
            return False
            
        try:
            # 获取MNIST数据加载器
            mnist_loader = self._get_mnist_loader(batch_size)
            if mnist_loader is None:
                return False
            
            # 只对编码器参数设置优化器，解码器参数不参与更新
            encoder_params = list(autoencoder.encoder.parameters())
            optimizer = torch.optim.Adam(encoder_params, lr=lr, weight_decay=1e-5)
            criterion = torch.nn.MSELoss()
            
            # 冻结解码器参数，确保不被更新
            decoder_grad_state = []
            for param in autoencoder.decoder.parameters():
                decoder_grad_state.append(param.requires_grad)
                param.requires_grad = False
            
            # 微调训练（只更新编码器）
            autoencoder.train()
            for epoch in range(epochs):
                total_loss = 0.0
                batch_count = 0
                
                for data, _ in mnist_loader:
                    try:
                        data = data.to(self.device, non_blocking=True)
                        optimizer.zero_grad()
                        output = autoencoder(data)
                        loss = criterion(output, data)
                        loss.backward()
                        
                        # 梯度裁剪防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)
                        
                        optimizer.step()
                        total_loss += loss.item() * data.size(0)
                        batch_count += 1
                        
                        # 更频繁的内存清理
                        if batch_count % 20 == 0:  # 从50改为20
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                        # 进一步限制微调的batch数量
                        if batch_count >= 50:  # 从100减少到50
                            break
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"⚠️ GPU内存不足，跳过当前batch: {e}")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                
                # 静默训练，不输出日志
                avg_loss = total_loss / (batch_count * batch_size) if batch_count > 0 else 0.0
            
            # 恢复解码器参数的原始梯度状态
            for param, original_grad_state in zip(autoencoder.decoder.parameters(), decoder_grad_state):
                param.requires_grad = original_grad_state
            
            # 清理优化器状态
            del optimizer
            del criterion
            
            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"⚠️ 编码器微调失败: {e}")
            print("继续使用原始编码器参数")
            # 确保清理资源
            if 'optimizer' in locals():
                del optimizer
            if 'criterion' in locals():
                del criterion
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            return False
    
    def evaluate_encoder_performance(self, autoencoder, test_samples=500):
        """
        评估编码器性能
        
        Args:
            autoencoder: 自编码器模型
            test_samples: 测试样本数量
            
        Returns:
            float: 平均重建损失
        """
        if autoencoder is None:
            return float('inf')
            
        try:
            # 使用缓存的训练数据或创建测试数据加载器
            if self._mnist_loader is not None:
                # 使用现有的数据加载器，但限制样本数量
                test_loader = self._mnist_loader
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                # 使用MNIST测试集进行评估
                mnist_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
                test_loader = torch.utils.data.DataLoader(
                    mnist_dataset, 
                    batch_size=64,  # 减少批次大小
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
            
            autoencoder.eval()
            criterion = torch.nn.MSELoss()
            total_loss = 0.0
            sample_count = 0
            batch_count = 0
            
            with torch.no_grad():
                for data, _ in test_loader:
                    if sample_count >= test_samples:
                        break
                        
                    try:
                        data = data.to(self.device, non_blocking=True)
                        output = autoencoder(data)
                        loss = criterion(output, data)
                        total_loss += loss.item() * data.size(0)
                        sample_count += data.size(0)
                        batch_count += 1
                        
                        # 每10个batch清理一次内存
                        if batch_count % 10 == 0:
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"⚠️ GPU内存不足，跳过评估batch: {e}")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            
            avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
            
            # 清理内存
            del criterion
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return avg_loss
            
        except Exception as e:
            print(f"⚠️ 编码器性能评估失败: {e}")
            return float('inf')
    
    def get_encoder_parameters(self, autoencoder):
        """
        提取编码器参数
        
        Args:
            autoencoder: 自编码器模型
            
        Returns:
            torch.Tensor: 编码器参数向量
        """
        if autoencoder is None:
            return None
            
        encoder_params = []
        for param in autoencoder.encoder.parameters():
            encoder_params.append(param.data.view(-1))
        
        return torch.cat(encoder_params)
    
    def cleanup(self):
        """清理内存和缓存"""
        # 开始清理自编码器微调器资源
        
        # 清理数据加载器
        if self._mnist_loader is not None:
            del self._mnist_loader
            self._mnist_loader = None
            
        if self._mnist_dataset is not None:
            del self._mnist_dataset
            self._mnist_dataset = None
        
        # 强制清理内存
        self._force_cleanup()


def finetune_autoencoder_encoder(autoencoder, device='cuda' if torch.cuda.is_available() else 'cpu', 
                                epochs=1, lr=0.005, batch_size=128):
    """
    便捷函数：微调自编码器的编码器部分
    
    Args:
        autoencoder: 自编码器模型
        device: 设备
        epochs: 微调轮数
        lr: 学习率
        batch_size: 批次大小
        
    Returns:
        bool: 微调是否成功
    """
    finetuner = AutoencoderFinetuner(device)
    return finetuner.finetune_encoder(autoencoder, epochs, lr, batch_size)


def extract_encoder_parameters(autoencoder):
    """
    便捷函数：提取编码器参数
    
    Args:
        autoencoder: 自编码器模型
        
    Returns:
        torch.Tensor: 编码器参数向量
    """
    finetuner = AutoencoderFinetuner()
    return finetuner.get_encoder_parameters(autoencoder)
