import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.light_autoencoder import LightAutoencoder
from models.losses.multi_loss import MultiLoss
from utils.key_matrix_utils import KeyMatrixManager
from utils.mask_utils import MaskManager


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
    
    def __init__(self, model, device, verbose=False, args=None):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.args = args

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
                    loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
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
                    loss_meter += F.binary_cross_entropy_with_logits(pred, target.float(), reduction='sum').item()
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
        self.multi_loss = MultiLoss()
        self.mask_manager = None
        self.autoencoder = None
        
        # 初始化掩码管理器
        if args and getattr(args, 'use_key_matrix', False):
            from utils.mask_utils import create_mask_manager
            # 在失败时会直接退出程序
            self.mask_manager = create_mask_manager(model, args.key_matrix_path, args)
            # 初始化时更新所有客户端的编码器掩码
            if self.mask_manager:
                self.mask_manager.update_encoder_mask()
        
        # 初始化自编码器（如果使用增强水印模式）
        if args and getattr(args, 'watermark_mode', '') == 'enhanced':
            try:
                self._initialize_autoencoder()
                print("自编码器已自动初始化")
            except Exception as e:
                print(f"错误: 初始化自编码器失败!")
                print(f"  错误详情: {e}")
                print(f"  请确保自编码器权重文件存在且格式正确")
                raise RuntimeError(f"自编码器初始化失败: {e}")

    def get_loss_function(self, pred, target):
        """计算损失函数，根据任务类型分支；
        multiclass 使用交叉熵，multi-label/binary 使用 BCEWithLogits。
        """
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            return F.cross_entropy(pred, target)
        if self.args is None or not self.args.class_weights:
            return F.binary_cross_entropy_with_logits(pred, target.float())
        
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
                print("自编码器权重已加载")
                load_weights = True
            else:
                print(f"错误: 自编码器权重文件不存在!")
                print(f"  编码器权重路径: {encoder_path}")
                print(f"  解码器权重路径: {decoder_path}")
                print(f"  请先运行 train_autoencoder.py 训练自编码器")
                raise FileNotFoundError(f"自编码器权重文件不存在: {encoder_path} 或 {decoder_path}")
            
            # 加载自编码器权重
            if load_weights:
                try:
                    # 加载编码器
                    if os.path.exists(encoder_path):
                        self.autoencoder.encoder.load_state_dict(
                            torch.load(encoder_path, map_location=self.device, weights_only=False)
                        )
                        print(f"✓ 编码器权重已加载: {encoder_path}")
                    
                    # 加载解码器
                    if os.path.exists(decoder_path):
                        self.autoencoder.decoder.load_state_dict(
                            torch.load(decoder_path, map_location=self.device, weights_only=False)
                        )
                        print(f"✓ 解码器权重已加载: {decoder_path}")
                        
                except Exception as e:
                    print(f"错误: 加载自编码器权重失败!")
                    print(f"  编码器路径: {encoder_path}")
                    print(f"  解码器路径: {decoder_path}")
                    print(f"  错误详情: {e}")
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
            
            try:
                position_dict = self._key_manager.load_positions(client_id)
            except Exception as e:
                print(f"加载客户端 {client_id} 位置信息失败: {e}")
                return
            
            with torch.no_grad():
                model_params = dict(self.model.named_parameters())
                watermarked_params = self._key_manager.embed_watermark(
                    model_params, client_id, encoder_params
                )
                
                for name, param in self.model.named_parameters():
                    if name in watermarked_params:
                        param.data.copy_(watermarked_params[name])
                
                print(f"水印嵌入完成，使用KeyMatrixManager自动缩放")
                
                # 水印嵌入完成后清理梯度数据
                if hasattr(self, 'gradient_batch_data') and len(self.gradient_batch_data) > 0:
                    print(f"清理客户端 {client_id} 的梯度数据 ({len(self.gradient_batch_data)} 个批次)")
                    self.clear_gradient_batch_data()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                                
        except Exception as e:
            print(f"嵌入水印失败: {e}")

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """本地更新，支持MultiLoss和自编码器训练"""
        self.model.train()
        
        # 根据args.optim参数选择优化器
        if self.args and hasattr(self.args, 'optim'):
            if self.args.optim.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=getattr(self.args, 'wd', 0.0))
            else:  # 默认使用SGD
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=getattr(self.args, 'wd', 0.0))
        else:
            # 向后兼容：如果没有args或optim参数，使用SGD
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        epoch_loss, epoch_acc = [], []
        
        # 初始化梯度统计变量（实时计算，不累积存储）
        total_gradients = None
        total_encoder_gradients = None
        total_target_mask = None
        total_encoder_mask = None
        total_effective_mask = None
        batch_count = 0

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(x)
                main_loss = self.get_loss_function(pred, y)

                # 计算最终损失
                if current_epoch == 0:
                    total_loss = main_loss
                else:
                    total_loss = self.multi_loss.compute_loss(main_loss, current_epoch, total_epochs)

                total_loss.backward()

                # 梯度统计收集：在训练过程中收集梯度信息（用于统计，不影响训练）
                if (self.mask_manager and current_epoch >= 0 and batch_idx % 5 == 0):  # 每5个batch收集一次
                    try:
                        # 只收集卷积层参数的梯度
                        conv_gradients = []
                        for name, param in self.model.named_parameters():
                            if 'conv' in name and 'weight' in name and param.grad is not None:
                                conv_gradients.append(param.grad.view(-1))
                        
                        if conv_gradients:
                            gradients = torch.cat(conv_gradients)
                            target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                            
                            # 计算编码器区域的梯度（用于prevGH）
                            encoder_gradients = torch.mul(gradients, effective_mask)
                            
                            # 累积梯度数据用于统计
                            if total_gradients is None:
                                total_gradients = gradients.detach()
                                total_encoder_gradients = encoder_gradients.detach()

                                total_target_mask = target_mask.detach()
                                total_encoder_mask = encoder_mask.detach()
                                total_effective_mask = effective_mask.detach()
                            else:
                                total_gradients += gradients.detach()
                                total_encoder_gradients += encoder_gradients.detach()
                            
                            batch_count += 1
                            
                            # 清理临时变量
                            del gradients, encoder_gradients, target_mask, encoder_mask, effective_mask
                        
                    except Exception as e:
                        print(f"更新梯度统计失败: {e}")

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

            if epoch + 1 == local_ep:
                print(f"Client {client_id} - Epoch {epoch+1}/{local_ep}: "
                      f"Loss={loss_meter:.4f}, Acc={acc_meter:.4f}, LR={lr:.6f}")

        # 每5个epoch进行水印融合
        if (current_epoch + 1) % 5 == 0:
            self._embed_watermark(client_id, current_epoch)

        # 本地训练结束后，使用累积的梯度数据更新统计量
        if (self.mask_manager and current_epoch >= 0 and batch_count > 0):
            try:
                # 计算平均梯度数据
                avg_gradients = total_gradients / batch_count
                avg_encoder_gradients = total_encoder_gradients / batch_count
                
                # 更新梯度统计量
                self.multi_loss.update_gradient_stats(
                    avg_gradients,
                    avg_encoder_gradients,
                    total_target_mask,
                    total_encoder_mask,
                    total_effective_mask
                )
                
            except Exception as e:
                print(f"Client {client_id} 梯度统计更新失败: {e}")
                print("梯度统计更新失败，终止训练")
                raise RuntimeError(f"梯度统计更新失败，训练终止: {e}")
            finally:
                # 确保清理梯度数据，防止内存泄漏
                if 'total_gradients' in locals():
                    del total_gradients
                if 'total_encoder_gradients' in locals():
                    del total_encoder_gradients
                if 'total_target_mask' in locals():
                    del total_target_mask
                if 'total_encoder_mask' in locals():
                    del total_encoder_mask
                if 'total_effective_mask' in locals():
                    del total_effective_mask
                if 'avg_gradients' in locals():
                    del avg_gradients
                if 'avg_encoder_gradients' in locals():
                    del avg_encoder_gradients
                
                # 强制垃圾回收
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 定期清理内存
        if current_epoch > 0 and current_epoch % 10 == 0:
            self._cleanup_memory()

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
            self.multi_loss.reset_batch_stats()

    def test(self, dataloader):
        """测试模型"""
        return self.tester.test(dataloader)
