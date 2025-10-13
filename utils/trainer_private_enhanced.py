import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.light_autoencoder import LightAutoencoder
from models.losses.multi_loss import MultiLoss
from utils.key_matrix_utils import KeyMatrixManager
from utils.mask_utils import MaskManager


def accuracy(output, target, top_k=(1,)):
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
    
    def __init__(self, model, device, verbose=False):
        self.model = model
        self.device = device
        self.verbose = verbose

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
        
        # MultiLoss和掩码管理器
        self.multi_loss = MultiLoss()
        self.mask_manager = None
        self.autoencoder = None
        
        # 初始化掩码管理器
        if args and getattr(args, 'use_key_matrix', False):
            try:
                from utils.mask_utils import create_mask_manager
                self.mask_manager = create_mask_manager(model, args.key_matrix_dir)
            except Exception as e:
                print(f"初始化掩码管理器失败: {e}")
                self.mask_manager = None

    def get_loss_function(self, pred, target):
        """计算损失函数，支持类别权重"""
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

    def _initialize_autoencoder(self):
        """初始化自编码器"""
        if self.autoencoder is None:
            self.autoencoder = LightAutoencoder().to(self.device)
            pretrained_path = './save/autoencoder/autoencoder.pth'
            
            if os.path.exists(pretrained_path):
                self.autoencoder.load_state_dict(torch.load(pretrained_path, map_location=self.device, weights_only=False))
                print(f"✓ 已加载预训练自编码器: {pretrained_path}")
            else:
                raise FileNotFoundError(f"预训练自编码器不存在: {pretrained_path}")


    def _extract_encoder_parameters(self):
        """提取编码器参数作为水印"""
        if self.autoencoder is None:
            self._initialize_autoencoder()
        
        encoder_params = []
        for param in self.autoencoder.encoder.parameters():
            encoder_params.append(param.data.view(-1))
        
        return torch.cat(encoder_params)

    def _embed_watermark(self, client_id, current_epoch):
        """嵌入水印到目标模型"""
        if not self.args or not getattr(self.args, 'use_key_matrix', False):
            return
        
        try:
            if self.mask_manager:
                self.mask_manager.update_encoder_mask(client_id)
            
            encoder_params = self._extract_encoder_parameters()
            if encoder_params is None:
                return
            
            if self._key_manager is None:
                try:
                    enable_scaling = getattr(self.args, 'enable_watermark_scaling', True)
                    scaling_factor = getattr(self.args, 'scaling_factor', 0.1)
                    
                    self._key_manager = KeyMatrixManager(
                        self.args.key_matrix_dir,
                        enable_scaling=enable_scaling,
                        scaling_factor=scaling_factor
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
                
                print(f"🔧 水印嵌入完成，使用KeyMatrixManager自动缩放")
                                
        except Exception as e:
            print(f"嵌入水印失败: {e}")

    def local_update(self, dataloader, local_ep, lr, client_id, current_epoch=0, total_epochs=100):
        """本地更新，支持MultiLoss和自编码器训练"""
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

                pred = self.model(x)
                main_loss = self.get_loss_function(pred, y)

                if current_epoch == 0:
                    total_loss = main_loss
                else:
                    total_loss = self.multi_loss.compute_loss(main_loss, current_epoch, total_epochs)

                total_loss.backward()

                if current_epoch > 0 and self.mask_manager:
                    try:
                        gradients = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
                        target_mask, encoder_mask, effective_mask = self.mask_manager.get_masks(self.device)
                        self.multi_loss.update_gradient_stats(
                            gradients, gradients, target_mask, encoder_mask, effective_mask
                        )
                    except Exception as e:
                        print(f"更新梯度统计失败: {e}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                acc_results = accuracy(pred, y)
                acc_meter += acc_results[0].item() * x.size(0) / 100.0
                loss_meter += main_loss.item() * x.size(0)
                run_count += x.size(0)

            loss_meter /= run_count
            acc_meter /= run_count

            epoch_loss.append(loss_meter)
            epoch_acc.append(acc_meter)

            if epoch + 1 == local_ep:
                print(f"Client {client_id} - Epoch {epoch+1}/{local_ep}: "
                      f"Loss={loss_meter:.4f}, Acc={acc_meter:.4f}, LR={lr:.6f}")

        # 每5个epoch进行水印融合
        if (current_epoch + 1) % 5 == 0:
            self._embed_watermark(client_id, current_epoch)
            print(f"Client {client_id} - Watermark embedded at epoch {current_epoch + 1}")

        # 差分隐私噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def test(self, dataloader):
        """测试模型"""
        return self.tester.test(dataloader)
