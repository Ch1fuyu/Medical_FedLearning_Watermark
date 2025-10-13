"""
自编码器微调模块
用于在联邦学习过程中微调自编码器的编码器部分
注意：只在内存中更新编码器参数，不修改原始.pth文件
解码器参数保持不变，由第三方保管
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms


class AutoencoderFinetuner:
    """自编码器微调器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def finetune_encoder(self, autoencoder, epochs=1, lr=0.005, batch_size=128):
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
            # 创建MNIST数据加载器用于自编码器训练
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 使用MNIST训练集进行微调
            mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
            
            # 只对编码器参数设置优化器，解码器参数不参与更新
            encoder_params = list(autoencoder.encoder.parameters())
            optimizer = torch.optim.Adam(encoder_params, lr=lr)
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
                for data, _ in mnist_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = autoencoder(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * data.size(0)
                
                avg_loss = total_loss / len(mnist_loader.dataset)
                # 简化输出：只在最后一个epoch显示
                if epoch + 1 == epochs:
                    print(f"编码器微调完成, Loss: {avg_loss:.6f}")
            
            # 恢复解码器参数的原始梯度状态
            for param, original_grad_state in zip(autoencoder.decoder.parameters(), decoder_grad_state):
                param.requires_grad = original_grad_state
            
            # 简化输出：移除冗余信息
            return True
            
        except Exception as e:
            print(f"⚠️ 编码器微调失败: {e}")
            print("继续使用原始编码器参数")
            return False
    
    def evaluate_encoder_performance(self, autoencoder, test_samples=1000):
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
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 使用MNIST测试集进行评估
            mnist_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=128, shuffle=False)
            
            autoencoder.eval()
            criterion = torch.nn.MSELoss()
            total_loss = 0.0
            sample_count = 0
            
            with torch.no_grad():
                for data, _ in test_loader:
                    if sample_count >= test_samples:
                        break
                        
                    data = data.to(self.device)
                    output = autoencoder(data)
                    loss = criterion(output, data)
                    total_loss += loss.item() * data.size(0)
                    sample_count += data.size(0)
            
            avg_loss = total_loss / sample_count
            # 简化输出：只在需要时显示性能评估
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
