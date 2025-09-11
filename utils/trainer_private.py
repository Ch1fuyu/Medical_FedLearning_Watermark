import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
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
        acc_meter = 0
        run_count = 0

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
                    acc_meter += label_acc.item() * data.size(0) / 100.0  # 使用标签级准确率，确保转换为标量
                    
                    # 计算sigmoid概率用于AUC计算
                    pred_prob = torch.sigmoid(pred)
                    
                    # 调试信息（仅在verbose模式下且是第一个batch时显示）
                    if self.verbose and run_count == 0:
                        pred_normal = torch.all(pred_prob < 0.5, dim=1).sum().item()
                        print(f"First batch - Label-level accuracy: {label_acc:.4f}%, Sample-level accuracy: {sample_acc:.4f}%, Predicted normal samples: {pred_normal}/{data.size(0)}")
                    
                    # AUC（宏平均）：需要sklearn
                    if roc_auc_score is not None:
                        try:
                            y_true = target.detach().cpu().numpy()
                            y_score = pred_prob.detach().cpu().numpy()
                            
                            # 检查每个类别是否有足够的正负样本
                            valid_classes = []
                            for i in range(y_true.shape[1]):
                                if len(np.unique(y_true[:, i])) > 1:  # 确保有0和1两种标签
                                    valid_classes.append(i)
                            
                            if len(valid_classes) > 0:
                                # 只计算有效类别的AUC
                                auc_scores = []
                                for i in valid_classes:
                                    try:
                                        auc_i = roc_auc_score(y_true[:, i], y_score[:, i])
                                        auc_scores.append(auc_i)
                                    except Exception:
                                        continue
                                
                                if len(auc_scores) > 0:
                                    auc_val = np.mean(auc_scores)  # 宏平均
                                else:
                                    auc_val = 0.0
                            else:
                                auc_val = 0.0
                        except Exception:
                            auc_val = 0.0
                    else:
                        auc_val = 0.0
                else:
                    # 单标签分类：使用CrossEntropyLoss
                    loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                    pred = pred.max(1, keepdim=True)[1]
                    acc_meter += pred.eq(target.view_as(pred)).sum().item()
                    # AUC（单标签多类）：使用概率
                    if roc_auc_score is not None:
                        try:
                            # 将logits转为softmax概率
                            prob = F.softmax(self.model(data), dim=1)
                            y_true = target.detach().cpu().numpy()
                            y_score = prob.detach().cpu().numpy()
                            
                            # 检查是否有足够的类别
                            unique_classes = np.unique(y_true)
                            if len(unique_classes) > 1:
                                # 对于多类，使用ovr宏平均
                                auc_val = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                            else:
                                auc_val = 0.0
                        except Exception:
                            auc_val = 0.0
                    else:
                        auc_val = 0.0
                
                run_count += data.size(0)

        loss_meter /= run_count
        acc_meter /= run_count

        # 确保acc_meter是标量
        if hasattr(acc_meter, 'item'):
            acc_meter = acc_meter.item()

        # 为保持接口一致性，返回 (loss, acc, auc)
        return loss_meter, acc_meter, float(auc_val)

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
        # 使用 Adam 优化器，移除weight_decay以对齐官方策略
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 本地训练不使用学习率调度，使用全局调度

        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        epoch_acc = []
        train_ldr = dataloader

        # 获取该客户端的随机参数位置
        position_dict = self.random_positions[client_id]

        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0
            batch_count = 0

            for batch_idx, (x, y) in enumerate(train_ldr):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)

                # 获取模型预测结果
                pred = self.model(x)
                
                # 检查是否为多标签分类
                if len(y.shape) == 2 and y.shape[1] == 14:
                    # 多标签分类：根据参数选择损失函数
                    loss += self.get_loss_function(pred, y)
                else:
                    # 单标签分类：使用CrossEntropyLoss
                    loss += F.cross_entropy(pred, y)
                
                acc_results = accuracy(pred, y)
                if len(y.shape) == 2 and y.shape[1] == 14:
                    # 多标签分类：使用标签级准确率
                    acc_meter += acc_results[0].item()  # 标签级准确率
                else:
                    # 单标签分类：使用原来的准确率
                    acc_meter += acc_results[0].item()

                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                loss_meter += loss.item()
                batch_count += 1
            
            loss_meter /= batch_count
            acc_meter /= batch_count

            epoch_loss.append(loss_meter)
            epoch_acc.append(acc_meter)
            
            # 输出客户端训练进度
            if epoch + 1 == local_ep:
                print(f"Client {client_id} - Epoch {epoch+1}/{local_ep}: Loss={loss_meter:.4f}, Acc={acc_meter:.4f}, LR={lr:.6f}")

            # 将整个参数空间展开成一维张量
            # all_params = torch.cat([param.view(-1) for param in self.model.parameters()])
            # # 在每轮训练结束后，将指定位置的参数设置为 0.5
            # for _, param_idx in position_dict:
            #     all_params[param_idx] = torch.tensor(0.5, device=self.device)

            # # 将修改后的一维张量重新赋值给模型参数
            # start_idx = 0
            # for param in self.model.parameters():
            #     param.data = all_params[start_idx:start_idx + param.numel()].view(param.size())
            #     start_idx += param.numel()

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss)

    def local_update_with_no_fl(self, dataloader, local_ep, lr):
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr,
                                   momentum=0.9,
                                   weight_decay=0.0005)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(x)
                
                # 检查是否为多标签分类
                if len(y.shape) == 2 and y.shape[1] == 14:
                    # 多标签分类：根据参数选择损失函数
                    loss = self.get_loss_function(pred, y)
                else:
                    # 单标签分类：使用CrossEntropyLoss
                    loss = F.cross_entropy(pred, y)
                
                acc_results = accuracy(pred, y)
                if len(y.shape) == 2 and y.shape[1] == 14:
                    # 多标签分类：使用标签级准确率
                    acc_meter += acc_results[0].item()  # 标签级准确率
                else:
                    # 单标签分类：使用原来的准确率
                    acc_meter += acc_results[0].item()

                loss.backward()
                self.optimizer.step()

                loss_meter += loss.item()

            loss_meter /= len(dataloader)
            acc_meter /= len(dataloader)

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

    def test(self, dataloader):
        return self.tester.test(dataloader)
