import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models.light_autoencoder import LightAutoencoder
from utils.key_matrix_utils import KeyMatrixManager
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        # 使用sigmoid和阈值
        pred_prob = torch.sigmoid(output)
        pred_binary = pred_prob > 0.5
        
        # 标签级准确率
        label_correct = (pred_binary == target).float().mean()
        
        # 样本级准确率
        sample_correct = torch.all(pred_binary == target, dim=1).float().mean()
        
        # 返回标签级准确率作为主要指标
        return [label_correct * 100.0, sample_correct * 100.0]

class TesterPrivate(object):
    def __init__(self, model, device, verbose=False, args=None):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.args = args

    def test(self, dataloader):
        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0  # 标签级准确率
        sample_acc_meter = 0  # 样本级准确率
        run_count = 0
        
        # 用于AUC计算的累积数据
        all_y_true = []
        all_y_score = []

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)

                if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
                    # 多分类
                    loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
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
                    # 多标签/二分类
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

        # 确保acc_meter是标量
        if hasattr(acc_meter, 'item'):
            acc_meter = acc_meter.item()

        # 计算整体AUC
        auc_val = 0.0
        if len(all_y_true) > 0 and roc_auc_score is not None:
            try:
                # 合并所有batch的数据
                y_true_all = np.concatenate(all_y_true, axis=0)
                y_score_all = np.concatenate(all_y_score, axis=0)
                if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
                    try:
                        auc_val = roc_auc_score(y_true_all, y_score_all, multi_class='ovr', average='macro')
                    except Exception:
                        auc_val = 0.0
                else:
                    # 多标签宏平均
                    valid_classes = []
                    for i in range(y_true_all.shape[1]):
                        if len(np.unique(y_true_all[:, i])) > 1:  # 确保有0和1两种标签
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
                            auc_val = np.mean(auc_scores)  # 宏平均
            except Exception as e:
                print(f"AUC计算错误: {e}")
                auc_val = 0.0

        # 为保持接口一致性并扩展多指标，返回 (loss, acc_label, auc, acc_sample)
        return loss_meter, acc_meter, float(auc_val), sample_acc_meter

class TrainerPrivate(object):
    def __init__(self, model, device, dp, sigma, random_positions, args=None):
        self.optimizer = None
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tester = TesterPrivate(model, device, args=args)
        self.dp = dp
        self.sigma = sigma
        self.random_positions = random_positions  # 加入随机位置列表
        self.args = args  # 添加参数对象
        self._key_manager = None
        self.position_dict = random_positions  # 为了兼容性，添加这个属性

    def get_loss_function(self, pred, target):
        """根据任务类型计算损失：multiclass 用 CrossEntropy；其他用 BCEWithLogits。"""
        if self.args is not None and getattr(self.args, 'task_type', 'multiclass') == 'multiclass':
            return F.cross_entropy(pred, target)
        if self.args is None or not self.args.class_weights:
            return F.binary_cross_entropy_with_logits(pred, target.float())
        
        # 计算类别权重
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
        return accuracy(pred, target)[0]

    def local_update(self, dataloader, local_ep, lr, client_id, **kwargs):
        self.model.train()
        
        # 提取 current_epoch 和 total_epochs（用于全局学习率调度）
        current_epoch = kwargs.get('current_epoch', 0)
        total_epochs = kwargs.get('total_epochs', 100)
        
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
            run_count = 0  # 统计总样本数

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                pred = self.model(x)

                # 计算损失
                loss = self.get_loss_function(pred, y)

                # 计算准确率（按样本加权）
                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 记录损失（按样本加权）
                loss_meter += loss.item() * x.size(0)
                run_count += x.size(0)

            # 样本平均
            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            epoch_acc.append(acc_meter)

            # 简洁输出：只在最后epoch打印
            if epoch + 1 == local_ep:
                from tqdm import tqdm
                tqdm.write(f"C{client_id} E{epoch+1}/{local_ep}: L={loss_meter:.4f} A={acc_meter:.4f} LR={adjusted_lr:.6f}")

        # 本地训练结束后，进行水印嵌入
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
                    try:
                        position_dict = self._key_manager.load_positions(client_id)
                    except Exception as e:
                        print(f"[Watermark Warning] Failed to load positions for client {client_id}: {e}. Fallback to random positions.")
                        position_dict = self.random_positions[client_id]
                else:
                    position_dict = self.random_positions[client_id]

                # 加载编码器
                encoder = LightAutoencoder().encoder.to(self.device)
                if self.args.encoder_path and torch.cuda.is_available():
                    encoder.load_state_dict(torch.load(self.args.encoder_path, weights_only=False))
                else:
                    encoder.load_state_dict(torch.load(self.args.encoder_path, map_location=self.device, weights_only=False))

                with torch.no_grad():
                    encoder_flat = torch.cat([param.view(-1) for param in encoder.parameters()])

                    # ==================== 水印参数自适应缩放 ====================
                    # 分析主任务参数和水印参数的数值范围
                    main_params = []
                    for name, param in self.model.named_parameters():
                        main_params.extend(param.view(-1).tolist())
                    
                    main_params = torch.tensor(main_params)
                    main_std = main_params.std().item()
                    main_mean_abs = main_params.abs().mean().item()
                    watermark_std = encoder_flat.std().item()
                    watermark_mean_abs = encoder_flat.abs().mean().item()
                    
                    # 使用KeyMatrixManager的embed_watermark方法，自动处理缩放
                    model_params = dict(self.model.named_parameters())
                    watermarked_params = self._key_manager.embed_watermark(
                        model_params, client_id, encoder_flat
                    )
                    
                    # 将水印参数更新到模型中
                    for name, param in self.model.named_parameters():
                        if name in watermarked_params:
                            param.data.copy_(watermarked_params[name])
                    
                    print(f"🔧 水印嵌入完成，使用KeyMatrixManager自动缩放")
                        
            except Exception as e:
                print(f"[Watermark Warning] Failed to embed watermark for client {client_id}: {e}")

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def local_update_with_no_fl(self, dataloader, local_ep, lr):
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

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0  # 统计总样本数

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                pred = self.model(x)

                # 计算损失
                loss = self.get_loss_function(pred, y)

                # 计算准确率（按样本加权）
                batch_acc = self._compute_accuracy(pred, y)
                acc_meter += batch_acc.item() * x.size(0) / 100.0

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 记录损失（按样本加权）
                loss_meter += loss.item() * x.size(0)
                run_count += x.size(0)

            # 样本平均
            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            epoch_acc.append(acc_meter)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)


    def test(self, dataloader):
        return self.tester.test(dataloader)
