import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models.light_autoencoder import LightAutoencoder
from utils.key_matrix_utils import KeyMatrixManager
from models.losses.multi_loss import MultiLoss
from utils.mask_utils import MaskManager
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
    def __init__(self, model, device, verbose=False):
        self.model = model
        self.device = device
        self.verbose = verbose

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
                
                # 使用普通BCEWithLogitsLoss进行评估
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
                
                # 宏平均AUC
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

class TrainerPrivateEnhanced(object):
    def __init__(self, model, device, dp, sigma, random_positions, args=None):
        self.optimizer = None
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.random_positions = random_positions
        self.args = args
        self._key_manager = None
        self.position_dict = random_positions
        
        # 新增：MultiLoss和掩码管理器
        self.multi_loss = MultiLoss()
        self.mask_manager = None
        self.autoencoder = None
        self.autoencoder_optimizer = None
        
        # 初始化掩码管理器
        if args and getattr(args, 'use_key_matrix', False):
            try:
                from utils.mask_utils import create_mask_manager
                self.mask_manager = create_mask_manager(model, args.key_matrix_dir)
            except Exception as e:
                print(f"初始化掩码管理器失败: {e}")
                self.mask_manager = None

    def get_loss_function(self, pred, target):
        """使用BCEWithLogitsLoss，支持类别权重"""
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

    def _initialize_autoencoder(self):
        """初始化自编码器"""
        if self.autoencoder is None:
            self.autoencoder = LightAutoencoder().to(self.device)
            self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)

    def _train_autoencoder_epoch(self, dataloader):
        """训练自编码器一个epoch"""
        self._initialize_autoencoder()
        self.autoencoder.train()
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            
            # 只使用第一个通道作为自编码器输入
            if len(x.shape) == 4 and x.shape[1] == 3:
                x = x[:, 0:1, :, :]  # 取第一个通道
            
            self.autoencoder_optimizer.zero_grad()
            
            # 前向传播
            reconstructed = self.autoencoder(x)
            
            # 计算重建损失
            loss = F.mse_loss(reconstructed, x)
            
            # 反向传播
            loss.backward()
            self.autoencoder_optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        return total_loss / batch_count if batch_count > 0 else 0.0

    def _extract_encoder_parameters(self):
        """提取编码器参数"""
        if self.autoencoder is None:
            return None
        
        encoder_params = []
        for param in self.autoencoder.encoder.parameters():
            encoder_params.append(param.data.view(-1))
        
        return torch.cat(encoder_params)

    def _embed_watermark(self, client_id, current_epoch):
        """嵌入水印到目标模型"""
        if not self.args or not getattr(self.args, 'use_key_matrix', False):
            return
        
        try:
            # 更新编码器掩码
            if self.mask_manager:
                self.mask_manager.update_encoder_mask(client_id)
            
            # 获取编码器参数
            encoder_params = self._extract_encoder_parameters()
            if encoder_params is None:
                return
            
            # 获取位置信息
            if self._key_manager is None:
                try:
                    self._key_manager = KeyMatrixManager(self.args.key_matrix_dir)
                except Exception as e:
                    print(f"加载密钥矩阵管理器失败: {e}")
                    return
            
            try:
                position_dict = self._key_manager.load_positions(client_id)
            except Exception as e:
                print(f"加载客户端 {client_id} 位置信息失败: {e}")
                return
            
            # 嵌入水印
            with torch.no_grad():
                wm_idx = 0
                for param_name, param_idx in position_dict:
                    if wm_idx < encoder_params.numel():
                        for name, param in self.model.named_parameters():
                            if name == param_name:
                                param_flat = param.view(-1)
                                if param_idx < param_flat.numel():
                                    param_flat[param_idx] = encoder_params[wm_idx]
                                wm_idx += 1
                                break
                                
        except Exception as e:
            print(f"嵌入水印失败: {e}")

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """增强的本地更新，支持MultiLoss和自编码器训练"""
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        epoch_loss, epoch_acc = [], []

        for epoch in range(local_ep):
            loss_meter = 0.0
            acc_meter = 0.0
            run_count = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                pred = self.model(x)

                # 计算主任务损失
                main_loss = self.get_loss_function(pred, y)

                # 计算总损失（第一轮只使用主任务损失）
                if current_epoch == 0:
                    total_loss = main_loss
                else:
                    # 使用MultiLoss
                    total_loss = self.multi_loss.compute_loss(main_loss, current_epoch, total_epochs)

                # 反向传播
                total_loss.backward()

                # 更新梯度统计（非第一轮）
                if current_epoch > 0 and self.mask_manager:
                    try:
                        # 获取当前梯度
                        gradients = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
                        
                        # 获取掩码
                        target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                        
                        # 更新梯度统计
                        self.multi_loss.update_gradient_stats(
                            gradients, gradients, target_mask, encoder_mask, effective_mask
                        )
                    except Exception as e:
                        print(f"更新梯度统计失败: {e}")

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 计算准确率
                acc_results = accuracy(pred, y)
                acc_meter += acc_results[0].item() * x.size(0) / 100.0

                # 记录损失
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)

            # 样本平均
            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            epoch_acc.append(acc_meter)

            if epoch + 1 == local_ep:
                print(f"Client {client_id} - Epoch {epoch+1}/{local_ep}: "
                      f"Loss={loss_meter:.4f}, Acc={acc_meter:.4f}, LR={lr:.6f}")

        # 训练自编码器
        autoencoder_loss = self._train_autoencoder_epoch(dataloader)
        print(f"Client {client_id} - Autoencoder Loss: {autoencoder_loss:.4f}")

        # 每5个epoch进行水印融合
        if (current_epoch + 1) % 5 == 0:
            self._embed_watermark(client_id, current_epoch)
            print(f"Client {client_id} - Watermark embedded at epoch {current_epoch + 1}")

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def test(self, dataloader):
        return self.tester.test(dataloader)
