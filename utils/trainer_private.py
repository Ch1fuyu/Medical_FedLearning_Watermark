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
        # 检查是否为多标签分类（ChestMNIST: batch_size x 14）
        if len(target.shape) == 2 and target.shape[1] == 14:
            # 多标签分类：使用sigmoid和阈值
            pred_prob = torch.sigmoid(output)
            pred_binary = pred_prob > 0.5
            
            # 标签级准确率
            label_correct = (pred_binary == target).float().mean()
            
            # 样本级准确率
            sample_correct = torch.all(pred_binary == target, dim=1).float().mean()
            
            # 返回标签级准确率作为主要指标
            return [label_correct * 100.0, sample_correct * 100.0]
        else:
            # 单标签分类：使用top-k准确率
            max_k = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

class TesterPrivate(object):
    def __init__(self, model, device, verbose=False):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test(self, dataloader):
        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0  # 标签级准确率（多标签时）或普通准确率（单标签）
        sample_acc_meter = 0  # 样本级准确率（多标签时），单标签退化为与acc相同
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
                
                # 检查是否为多标签分类
                if len(target.shape) == 2 and target.shape[1] == 14:
                    # 多标签分类：使用普通BCEWithLogitsLoss进行评估
                    loss_meter += F.binary_cross_entropy_with_logits(pred, target.float(), reduction='sum').item()
                    # 使用新的准确率计算函数
                    acc_results = accuracy(pred, target)
                    label_acc = acc_results[0]  # 标签级准确率（主要指标）
                    sample_acc = acc_results[1]  # 样本级准确率
                    acc_meter += label_acc.item() * data.size(0) / 100.0  # 标签级准确率
                    sample_acc_meter += sample_acc.item() * data.size(0) / 100.0  # 样本级准确率
                    
                    # 计算sigmoid概率用于AUC计算
                    pred_prob = torch.sigmoid(pred)
                    
                    # 调试信息（仅在verbose模式下且是第一个batch时显示）
                    if self.verbose and run_count == 0:
                        pred_normal = torch.all(pred_prob < 0.5, dim=1).sum().item()
                        print(f"First batch - Label-level accuracy: {label_acc:.4f}%, Sample-level accuracy: {sample_acc:.4f}%, Predicted normal samples: {pred_normal}/{data.size(0)}")
                    
                    # 累积AUC计算数据
                    all_y_true.append(target.detach().cpu().numpy())
                    all_y_score.append(pred_prob.detach().cpu().numpy())
                else:
                    # 单标签分类：使用CrossEntropyLoss
                    loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                    pred = pred.max(1, keepdim=True)[1]
                    acc_meter += pred.eq(target.view_as(pred)).sum().item()
                    sample_acc_meter += pred.eq(target.view_as(pred)).sum().item()
                    # 累积单标签分类的AUC计算数据
                    prob = F.softmax(self.model(data), dim=1)
                    all_y_true.append(target.detach().cpu().numpy())
                    all_y_score.append(prob.detach().cpu().numpy())
                
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
                
                # 检查是否为多标签分类
                if len(y_true_all.shape) == 2 and y_true_all.shape[1] == 14:
                    # 多标签分类：宏平均AUC
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
                else:
                    # 单标签分类：多类AUC
                    unique_classes = np.unique(y_true_all)
                    if len(unique_classes) > 1:
                        auc_val = roc_auc_score(y_true_all, y_score_all, multi_class='ovr', average='macro')
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
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.random_positions = random_positions  # 加入随机位置列表
        self.args = args  # 添加参数对象
        self._key_manager = None
        self.position_dict = random_positions  # 为了兼容性，添加这个属性

    def get_loss_function(self, pred, target):
        """使用BCEWithLogitsLoss，支持类别权重"""
        if self.args is None or not self.args.class_weights:
            # 使用标准BCEWithLogitsLoss
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

    def local_update(self, dataloader, local_ep, lr, client_id):
        self.model.train()
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
                if len(y.shape) == 2 and y.shape[1] == 14:  # 多标签
                    loss = self.get_loss_function(pred, y)
                else:  # 单标签
                    loss = F.cross_entropy(pred, y)

                # 计算准确率（按样本加权）
                acc_results = accuracy(pred, y)
                acc_meter += acc_results[0].item() * x.size(0) / 100.0

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

            if epoch + 1 == local_ep:
                print(f"Client {client_id} - Epoch {epoch+1}/{local_ep}: "
                      f"Loss={loss_meter:.4f}, Acc={acc_meter:.4f}, LR={lr:.6f}")

        # 本地训练结束后，进行水印嵌入
        if self.args is not None and getattr(self.args, 'use_key_matrix', False):
            try:
                # 获取该客户端的水印位置：优先使用保存的密钥矩阵；否则回退到随机位置
                if self._key_manager is None:
                    try:
                        self._key_manager = KeyMatrixManager(self.args.key_matrix_dir)
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
                    all_params = torch.cat([param.view(-1) for param in self.model.parameters()])
                    encoder_flat = torch.cat([param.view(-1) for param in encoder.parameters()])

                    # 按顺序将编码器参数拷贝到水印位置
                    for wm_idx, (_, param_idx) in enumerate(position_dict):
                        if wm_idx < encoder_flat.numel() and param_idx < all_params.numel():
                            all_params[param_idx] = encoder_flat[wm_idx]

                    # 将修改后的参数还原回模型
                    offset = 0
                    for param in self.model.parameters():
                        numel = param.numel()
                        param.data.copy_(all_params[offset:offset + numel].view_as(param))
                        offset += numel
                        
            except Exception as e:
                print(f"[Watermark Warning] Failed to embed watermark for client {client_id}: {e}")

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def local_update_with_no_fl(self, dataloader, local_ep, lr):
        self.model.train()
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
                if len(y.shape) == 2 and y.shape[1] == 14:  # 多标签
                    loss = self.get_loss_function(pred, y)
                else:  # 单标签
                    loss = F.cross_entropy(pred, y)

                # 计算准确率（按样本加权）
                acc_results = accuracy(pred, y)
                acc_meter += acc_results[0].item() * x.size(0) / 100.0

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
