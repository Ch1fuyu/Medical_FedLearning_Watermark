import os

import numpy as np
import torch
import torch.nn.functional as F

from models.light_autoencoder import LightAutoencoder
from models.losses.multi_loss import FocalLoss
from utils.key_matrix_utils import KeyMatrixManager
from utils.trainer_private_enhanced import TesterPrivate


def accuracy(output, target):
    """计算多标签分类的准确率"""
    with torch.no_grad():
        pred_prob = torch.sigmoid(output)
        pred_binary = pred_prob > 0.5
        
        # 标签级准确率
        label_correct = (pred_binary == target).float().mean()
        # 样本级准确率
        sample_correct = torch.all(pred_binary == target, dim=1).float().mean()
        
        return [label_correct * 100.0, sample_correct * 100.0]


class TrainerAblation:
    """
    消融实验训练器：只使用主任务损失，不使用三个正则项
    但保持水印嵌入逻辑不变（使用KeyMatrixManager进行参数替换）
    """
    
    def __init__(self, model, device, dp, sigma, args=None):
        self.optimizer = None
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tester = TesterPrivate(model, device, args=args)
        self.dp = dp
        self.sigma = sigma
        self.args = args
        self._key_manager = None
        
        # 初始化FocalLoss（用于多标签任务）
        focal_gamma = 1.0
        focal_reduction = 'mean'
        if args and hasattr(args, 'focal_gamma'):
            focal_gamma = args.focal_gamma
        if args and hasattr(args, 'focal_reduction'):
            focal_reduction = args.focal_reduction
        self.focal_loss = FocalLoss(alpha=None, gamma=focal_gamma, reduction=focal_reduction)

    def get_loss_function(self, pred, target):
        """
        计算损失函数，根据任务类型分支
        multiclass 使用交叉熵，multi-label/binary 使用 FocalLoss（ChestMNIST）
        
        注意：消融实验只使用主任务损失，不使用正则项
        """
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            return F.cross_entropy(pred, target)
        
        # 对于ChestMNIST等multi-label任务，使用FocalLoss
        if self.args is None or not self.args.class_weights:
            return self.focal_loss(pred, target)
        
        pos_counts = target.sum(dim=0)
        neg_counts = target.shape[0] - pos_counts
        pos_weights = torch.zeros_like(pos_counts, dtype=torch.float32, device=pred.device)
        
        for i in range(len(pos_counts)):
            if pos_counts[i] > 0 and neg_counts[i] > 0:
                pos_weights[i] = (neg_counts[i] / pos_counts[i]) * self.args.pos_weight_factor
            else:
                pos_weights[i] = 1.0
        
        pos_weights = torch.clamp(pos_weights, min=0.1, max=10.0)
        return F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=pos_weights)

    def _compute_accuracy(self, pred, target):
        """根据任务类型计算准确率（百分比）"""
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            preds_top1 = pred.argmax(dim=1)
            return (preds_top1 == target).float().mean() * 100.0
        # multilabel/binary 使用原有accuracy()
        return accuracy(pred, target)[0]

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """
        本地更新
        注意：消融实验只使用主任务损失，不使用MultiLoss的三个正则项
        """
        self.model.to(self.device)  # 确保模型在正确的设备上
        self.model.train()
        
        # 如果启用学习率调度器，根据全局轮次计算当前学习率
        if self.args and getattr(self.args, 'use_lr_scheduler', False):
            import math
            # 余弦退火：基于全局训练轮次
            progress = current_epoch / total_epochs  # 0 到 1
            lr_min = lr * 0.01
            adjusted_lr = lr_min + (lr - lr_min) * (1 + math.cos(math.pi * progress)) / 2
        else:
            adjusted_lr = lr
        
        # 根据args.optim参数选择优化器（使用调整后的学习率）
        if self.args and hasattr(self.args, 'optim'):
            if self.args.optim.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=adjusted_lr, weight_decay=getattr(self.args, 'wd', 0.0))
            else:  # 默认使用SGD
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=adjusted_lr, momentum=0.9, weight_decay=getattr(self.args, 'wd', 0.0))
        else:
            # 向后兼容：如果没有args或optim参数，使用SGD
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=adjusted_lr, momentum=0.9)

        epoch_loss, epoch_acc = [], []

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(x)
                # 消融实验：只使用主任务损失，不使用正则项
                main_loss = self.get_loss_function(pred, y)
                total_loss = main_loss  # 直接使用主任务损失

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)
                
                # 定期清理内存，防止内存泄漏
                if batch_idx % 3 == 0:
                    torch.cuda.empty_cache()

            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            
            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            epoch_acc.append(acc_meter)

            # 简洁输出：只在最后epoch打印
            if epoch + 1 == local_ep:
                from tqdm import tqdm
                tqdm.write(f"C{client_id} E{epoch+1}/{local_ep}: L={loss_meter:.4f} A={acc_meter:.4f} LR={adjusted_lr:.6f}")

        # 本地训练结束后，进行水印嵌入（消融实验保持水印嵌入逻辑不变）
        if self.args is not None and getattr(self.args, 'use_key_matrix', False):
            try:
                # 获取该客户端的水印位置：优先使用保存的密钥矩阵；否则回退到随机位置
                if self._key_manager is None:
                    try:
                        # 初始化KeyMatrixManager，支持水印缩放
                        self._key_manager = KeyMatrixManager(
                            self.args.key_matrix_path,
                            args=self.args
                        )
                    except Exception as e:
                        print(f"[Watermark Warning] Failed to load KeyMatrixManager: {e}. Fallback to random positions.")
                        self._key_manager = None
                
                if self._key_manager is not None:
                    # 加载编码器
                    encoder = LightAutoencoder().encoder.to(self.device)
                    if self.args.encoder_path and torch.cuda.is_available():
                        encoder.load_state_dict(torch.load(self.args.encoder_path, weights_only=False))
                    else:
                        encoder.load_state_dict(torch.load(self.args.encoder_path, map_location=self.device, weights_only=False))

                    with torch.no_grad():
                        encoder_flat = torch.cat([param.view(-1) for param in encoder.parameters()])
                        
                        # 使用KeyMatrixManager的embed_watermark方法，自动处理缩放
                        model_params = dict(self.model.named_parameters())
                        watermarked_params = self._key_manager.embed_watermark(
                            model_params, client_id, encoder_flat
                        )
                        
                        # 将水印参数更新到模型中
                        for name, param in self.model.named_parameters():
                            if name in watermarked_params:
                                param.data.copy_(watermarked_params[name])
                        
                        # 静默嵌入，减少日志输出
                        pass
                                
            except Exception as e:
                print(f"[Watermark Warning] Failed to embed watermark for client {client_id}: {e}")

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)
    
    def _cleanup_memory(self):
        """清理内存和GPU缓存"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ 内存清理失败: {e}")

    def test(self, dataloader):
        """测试模型"""
        return self.tester.test(dataloader)

