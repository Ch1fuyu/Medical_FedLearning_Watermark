import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def is_softmax_task(args):
    """判断是否为 softmax 任务（多分类或二分类），否则为多标签任务"""
    if args is None:
        return True  # 默认假设为 softmax 任务
    task_type = getattr(args, 'task_type', 'multiclass')
    return task_type in ['multiclass', 'binary']


def compute_accuracy(pred, target, task_type='multiclass'):
    """计算准确率，返回 [label_acc, sample_acc]"""
    if task_type in ['multiclass', 'binary']:
        # 多分类或二分类
        preds_top1 = pred.argmax(dim=1)
        if target.dim() > 1:
            target = target.argmax(dim=1)
        label_acc = (preds_top1 == target).float().mean() * 100.0
        return [label_acc.item(), label_acc.item()]
    else:
        # 多标签分类
        pred_prob = torch.sigmoid(pred)
        pred_binary = pred_prob > 0.5
        label_correct = (pred_binary == target).float().mean()
        sample_correct = torch.all(pred_binary == target, dim=1).float().mean()
        return [label_correct * 100.0, sample_correct * 100.0]


def compute_auc(y_true_list, y_score_list, task_type='multiclass'):
    """计算 AUC"""
    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # 多标签：计算每个标签的 AUC 然后取平均
        aucs = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:  # 需要两个类别才能计算 AUC
                try:
                    auc = roc_auc_score(y_true[:, i], y_score[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass
        return np.mean(aucs) if aucs else 0.0
    else:
        # 单标签：二分类或多分类
        if task_type in ['multiclass', 'binary']:
            # 对于多分类，使用第一列（正类的概率）
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1] if y_score.shape[1] == 2 else y_score.argmax(axis=1)
        try:
            if len(np.unique(y_true)) > 1:
                return roc_auc_score(y_true, y_score)
        except ValueError:
            pass
        return 0.0


def compute_loss(pred, target, task_type='multiclass', class_weights=None):
    """计算损失"""
    import torch.nn.functional as F
    
    if task_type in ['multiclass', 'binary']:
        if target.dim() > 1:
            target = target.argmax(dim=1)
        return F.cross_entropy(pred, target)
    else:
        # 多标签
        if class_weights is not None:
            return F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=class_weights)
        return F.binary_cross_entropy_with_logits(pred, target.float())


def normalize_targets(target):
    """归一化标签"""
    return target
