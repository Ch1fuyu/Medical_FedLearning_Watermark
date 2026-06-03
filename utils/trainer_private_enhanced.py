import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.light_autoencoder import LightAutoencoder
from models.losses.multi_loss import MultiLoss, FocalLoss
from utils.key_matrix_utils import KeyMatrixManager


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

class TesterPrivate:
    """测试器类，用于模型评估"""

    def __init__(self, model, device, verbose=False, args=None, loss_fn=None, get_loss_fn=None):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.args = args
        self.loss_fn = loss_fn  # 可选的外部损失函数（如FocalLoss）
        self.get_loss_fn = get_loss_fn  # 获取正确损失函数的回调

    def _compute_loss(self, pred, target):
        """根据任务类型计算损失，支持外部损失函数"""
        # 如果有 get_loss_fn 回调（优先级最高），使用它获取正确的损失函数
        if self.get_loss_fn is not None:
            return self.get_loss_fn(pred, target)

        # 如果有外部损失函数，优先使用
        if self.loss_fn is not None:
            return self.loss_fn(pred, target)

        # 默认使用标准损失函数
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            return F.cross_entropy(pred, target, reduction='mean')
        return F.binary_cross_entropy_with_logits(pred, target.float(), reduction='mean')

    def _compute_loss_sum(self, pred, target):
        """计算损失的 sum 版本，用于累积"""
        # 如果有 get_loss_fn 回调（优先级最高），使用它获取正确的损失函数
        if self.get_loss_fn is not None:
            return self.get_loss_fn(pred, target) * target.size(0)

        # 如果有外部损失函数，转换为 sum
        if self.loss_fn is not None:
            return self.loss_fn(pred, target) * target.size(0)

        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            return F.cross_entropy(pred, target, reduction='sum')
        return F.binary_cross_entropy_with_logits(pred, target.float(), reduction='sum')

    def test(self, dataloader):
        """测试模型性能"""
        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        sample_acc_meter = 0
        run_count = 0
        
        all_y_true = []
        all_y_score = []

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)

                if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
                    loss_meter += self._compute_loss_sum(pred, target).item()
                    # top-1 accuracy
                    preds_top1 = pred.argmax(dim=1)
                    label_acc = (preds_top1 == target).float().mean() * 100.0
                    sample_acc = label_acc
                    acc_meter += label_acc.item() * data.size(0) / 100.0
                    sample_acc_meter += sample_acc.item() * data.size(0) / 100.0
                    probs = torch.softmax(pred, dim=1)
                    if self.verbose and run_count == 0:
                        correct_first = (preds_top1 == target).sum().item()
                        print(f"First batch - Top1 Acc: {label_acc:.4f}% ({correct_first}/{data.size(0)})")
                    all_y_true.append(target.detach().cpu().numpy())
                    all_y_score.append(probs.detach().cpu().numpy())
                else:
                    # multilabel/binary
                    loss_meter += self._compute_loss_sum(pred, target).item()
                    acc_results = accuracy(pred, target)
                    label_acc = acc_results[0]
                    sample_acc = acc_results[1]
                    acc_meter += label_acc.item() * data.size(0) / 100.0
                    sample_acc_meter += sample_acc.item() * data.size(0) / 100.0
                    pred_prob = torch.sigmoid(pred)
                    if self.verbose and run_count == 0:
                        pred_normal = torch.all(pred_prob < 0.5, dim=1).sum().item()
                        print(f"First batch - Label-level accuracy: {label_acc:.4f}%, Sample-level accuracy: {sample_acc:.4f}%, Predicted normal samples: {pred_normal}/{data.size(0)}")
                    all_y_true.append(target.detach().cpu().numpy())
                    all_y_score.append(pred_prob.detach().cpu().numpy())
                run_count += data.size(0)

        loss_meter /= run_count
        acc_meter /= run_count
        sample_acc_meter /= run_count

        if hasattr(acc_meter, 'item'):
            acc_meter = acc_meter.item()

        # 计算AUC
        auc_val = 0.0
        if len(all_y_true) > 0:
            try:
                y_true_all = np.concatenate(all_y_true, axis=0)
                y_score_all = np.concatenate(all_y_score, axis=0)
                if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
                    # 使用一对多宏平均AUC，如果标签单一或异常则回退0.0
                    try:
                        auc_val = roc_auc_score(y_true_all, y_score_all, multi_class='ovr', average='macro')
                    except Exception:
                        auc_val = 0.0
                else:
                    valid_classes = []
                    for i in range(y_true_all.shape[1]):
                        if len(np.unique(y_true_all[:, i])) > 1:
                            valid_classes.append(i)
                    if len(valid_classes) > 0:
                        auc_scores = []
                        for i in valid_classes:
                            try:
                                auc_i = roc_auc_score(y_true_all[:, i], y_score_all[:, i])
                                auc_scores.append(auc_i)
                            except Exception:
                                continue
                        if len(auc_scores) > 0:
                            auc_val = np.mean(auc_scores)
            except Exception as e:
                print(f"AUC计算错误: {e}")
                auc_val = 0.0

        return loss_meter, acc_meter, float(auc_val), sample_acc_meter

class TrainerPrivateEnhanced:
    """增强版训练器，支持MultiLoss和自编码器水印"""
    
    def __init__(self, model, device, dp, sigma, args=None):
        self.optimizer = None
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tester = TesterPrivate(model, device, args=args)
        self.dp = dp
        self.sigma = sigma
        self.args = args
        self._key_manager = None
        
        # MultiLoss和掩码管理器
        self.multi_loss = MultiLoss(model=self.model)
        # 初始化参数快照，便于后续观察漂移
        if hasattr(self.multi_loss, 'update_param_snapshot'):
            self.multi_loss.update_param_snapshot(self.model)
        self.mask_manager = None
        self.autoencoder = None

        # 初始化FocalLoss，降低gamma避免过度关注困难样本，提升训练稳定性
        focal_gamma = 1.0  # 降低到1.0，更接近标准BCE
        focal_reduction = 'mean'
        if args and hasattr(args, 'focal_gamma'):
            focal_gamma = args.focal_gamma
        if args and hasattr(args, 'focal_reduction'):
            focal_reduction = args.focal_reduction
        self.focal_loss = FocalLoss(alpha=None, gamma=focal_gamma, reduction=focal_reduction)

        # 传递focal_loss和get_loss_function方法给TesterPrivate，确保test时使用正确的损失函数
        self.tester = TesterPrivate(model, device, args=args, loss_fn=self.focal_loss, get_loss_fn=self.get_loss_function)
        
        # 初始化掩码管理器
        if args and getattr(args, 'use_key_matrix', False):
            from utils.mask_utils import create_mask_manager
            self.mask_manager = create_mask_manager(model, args.key_matrix_path, args)
        
        # 初始化自编码器（如果使用增强水印模式）
        if args and getattr(args, 'watermark_mode', '') == 'enhanced':
            self._initialize_autoencoder()

    def get_loss_function(self, pred, target):
        """计算损失函数，根据任务类型分支；
        multiclass 使用交叉熵，multi-label/binary 使用 FocalLoss（ChestMNIST）。
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
        # 仍然使用加权BCEWithLogits作为备选
        return F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=pos_weights)

    def _compute_accuracy(self, pred, target):
        """根据任务类型计算准确率（百分比）。"""
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            preds_top1 = pred.argmax(dim=1)
            return (preds_top1 == target).float().mean() * 100.0
        # multilabel/binary 使用原有accuracy()
        return accuracy(pred, target)[0]

    def _initialize_autoencoder(self):
        """初始化自编码器，分别加载编码器和解码器"""
        if self.autoencoder is None:
            # 使用单通道输入的自编码器（MNIST训练）
            self.autoencoder = LightAutoencoder(input_channels=1).to(self.device)
            
            # 检查是否有自编码器权重
            weights_dir = './save/autoencoder'
            encoder_path = os.path.join(weights_dir, 'encoder.pth')
            decoder_path = os.path.join(weights_dir, 'decoder.pth')
            
            if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                load_weights = True
            else:
                raise FileNotFoundError(f"自编码器权重文件不存在: {encoder_path} 或 {decoder_path}")
            
            # 加载自编码器权重
            if load_weights:
                try:
                    # 加载编码器
                    if os.path.exists(encoder_path):
                        self.autoencoder.encoder.load_state_dict(
                            torch.load(encoder_path, map_location=self.device, weights_only=False)
                        )
                    
                    # 加载解码器
                    if os.path.exists(decoder_path):
                        self.autoencoder.decoder.load_state_dict(
                            torch.load(decoder_path, map_location=self.device, weights_only=False)
                        )
                        
                except Exception as e:
                    raise RuntimeError(f"自编码器权重加载失败: {e}")
            # 使用随机初始化的自编码器权重
    
    def _fine_tune_autoencoder(self, epochs=1, lr=0.005):
        """在每轮联邦学习开始前微调自编码器，确保性能稳定
        注意：只在内存中更新编码器参数，不修改原始.pth文件
        解码器参数保持不变，由第三方保管
        """
        if self.autoencoder is None:
            self._initialize_autoencoder()
        
        # 使用外部自编码器微调模块
        from .autoencoder_finetuner import finetune_autoencoder_encoder
        
        finetune_autoencoder_encoder(
            autoencoder=self.autoencoder,
            device=self.device,
            epochs=epochs,
            lr=lr,
            batch_size=128
        )
        
        # 简化输出：移除冗余信息

    def _extract_encoder_parameters(self):
        """提取编码器参数作为水印"""
        if self.autoencoder is None:
            self._initialize_autoencoder()
        
        # 使用外部模块提取编码器参数
        from .autoencoder_finetuner import extract_encoder_parameters
        return extract_encoder_parameters(self.autoencoder)

    def _embed_watermark(self, client_id, current_epoch):
        """嵌入水印到目标模型"""
        if not self.args or not getattr(self.args, 'use_key_matrix', False):
            return
        
        try:
            encoder_params = self._extract_encoder_parameters()
            if encoder_params is None:
                return
            
            if self._key_manager is None:
                try:
                    self._key_manager = KeyMatrixManager(
                        self.args.key_matrix_path,
                        args=self.args
                    )
                except Exception as e:
                    print(f"加载密钥矩阵管理器失败: {e}")
                    return
            
            with torch.no_grad():
                # 必须传递 self.model 以确保参数顺序与密钥矩阵生成时一致
                model_params = dict(self.model.named_parameters())
                watermarked_params = self._key_manager.embed_watermark(
                    model_params, client_id, encoder_params, model=self.model
                )
                
                for name, param in self.model.named_parameters():
                    if name in watermarked_params:
                        param.data.copy_(watermarked_params[name])
                # 静默嵌入，减少日志输出
                pass
                                
        except Exception as e:
            print(f"⚠️ 水印嵌入失败: {e}")

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """本地更新，支持MultiLoss和自编码器训练"""
        self.model.to(self.device)
        self.model.train()

        if self.mask_manager:
            try:
                self.mask_manager.update_encoder_mask(client_id)
            except Exception as e:
                print(f"⚠️ 客户端{client_id}掩码更新失败: {e}")
        
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
                main_loss = self.get_loss_function(pred, y)

                # 计算最终损失并反向传播（包含对比正则项）
                if current_epoch == 0:
                    # 第一轮只使用主任务损失
                    total_loss = main_loss
                    total_loss.backward()
                else:
                    # 获取掩码用于对比正则项
                    if self.mask_manager:
                        target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                    else:
                        target_mask, encoder_mask, effective_mask = None, None, None
                    
                    # 使用 compute_loss_and_backward 正确应用对比正则项
                    total_loss = self.multi_loss.compute_loss_and_backward(
                        main_loss,
                        target_mask,
                        encoder_mask,
                        effective_mask,
                        current_epoch,
                        total_epochs
                    )
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)
                
                # 定期清理内存，防止内存泄漏
                if batch_idx % 3 == 0:  # 更频繁的内存清理
                    torch.cuda.empty_cache()
                
                # 每20个批次进行激进内存清理
                if batch_idx % 20 == 0 and batch_idx > 0:
                    self._aggressive_memory_cleanup()

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

        # 差分隐私噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def get_gradient_stats(self):
        """获取梯度统计量"""
        if hasattr(self, 'multi_loss'):
            return self.multi_loss.get_stats()
        return {}
    
    def _cleanup_memory(self):
        """清理内存和GPU缓存（保留梯度数据）"""
        try:
            # 不清理梯度数据，保留用于水印嵌入
            # 梯度数据将在水印嵌入完成后统一清理
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 内存清理完成
        except Exception as e:
            print(f"⚠️ 内存清理失败: {e}")
    
    def _aggressive_memory_cleanup(self):
        """激进的内存清理策略（保留梯度数据）"""
        try:
            # 不清理梯度数据，保留用于水印嵌入
            # 梯度数据将在水印嵌入完成后统一清理
            
            # 清理模型梯度
            if hasattr(self, 'model'):
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = None
            
            # 清理优化器状态
            if hasattr(self, 'optimizer'):
                self.optimizer.zero_grad()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 激进内存清理完成
        except Exception as e:
            print(f"⚠️ 激进内存清理失败: {e}")
    
    def reset_gradient_stats(self):
        """重置梯度统计量"""
        if hasattr(self, 'multi_loss'):
            self.multi_loss.reset_stats()

    def test(self, dataloader):
        """测试模型"""
        return self.tester.test(dataloader)
