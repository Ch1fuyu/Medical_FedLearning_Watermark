import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-label classification"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = 1.0
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none', weight=self.alpha
        )
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiLoss:
    """Minimal wrapper for watermark-gradient diagnostics and param-drift tracking."""

    def __init__(self, model=None, target_ratio=0.3):
        self.model = model
        self.target_ratio = target_ratio
        self.device = None

        self._ema_gm = 0.0
        self._ema_gh = 0.0
        self._ema_ratio = 0.0
        self.smoothing = 0.9
        self._is_initialized = False

        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0

        self.current_wm_grad = 0.0
        self.current_non_wm_grad = 0.0
        self.current_wm_ratio = 1.0

        self._param_snapshot = None
        self._total_param_drift = 0.0
        self._wm_param_drift = 0.0

    @property
    def prevGM(self):
        return self._ema_gm

    @property
    def prevGH(self):
        return self._ema_gh

    @property
    def prevRatio(self):
        return self._ema_ratio

    def set_device(self, device):
        self.device = device

    def update_gradient_stats(self, target_gradients, encoder_gradients,
                              target_mask, encoder_mask, effective_mask):
        if self.device is None:
            self.device = target_gradients.device

        nonzero_threshold = 0.00001

        target_nonzero_mask = torch.where(
            torch.abs(target_gradients) <= nonzero_threshold,
            torch.zeros_like(target_gradients),
            torch.ones_like(target_gradients),
        )
        target_grad_filtered = torch.abs(target_gradients) * target_nonzero_mask
        target_nonzero_count = torch.sum(target_nonzero_mask)

        encoder_nonzero_mask = torch.where(
            torch.abs(encoder_gradients) <= nonzero_threshold,
            torch.zeros_like(encoder_gradients),
            torch.ones_like(encoder_gradients),
        )
        encoder_grad_filtered = torch.abs(encoder_gradients) * encoder_nonzero_mask
        encoder_nonzero_count = torch.sum(encoder_nonzero_mask)

        current_gm = (
            torch.sum(target_grad_filtered) / target_nonzero_count
            if target_nonzero_count > 0
            else torch.tensor(0.0, device=self.device)
        )
        current_gh = (
            torch.sum(encoder_grad_filtered) / encoder_nonzero_count
            if encoder_nonzero_count > 0
            else torch.tensor(0.0, device=self.device)
        )

        self.current_grad_M = current_gm.detach().item()
        self.current_grad_H = current_gh.detach().item()

        target_mean = torch.sum(target_grad_filtered) / target_nonzero_count if target_nonzero_count > 0 else None
        encoder_mean = torch.sum(encoder_grad_filtered) / encoder_nonzero_count if encoder_nonzero_count > 0 else None

        self.current_var_M = (
            torch.sum(torch.pow(target_grad_filtered - target_mean, 2)) / target_nonzero_count
        ).detach().item() if target_mean is not None else 0.0
        self.current_var_H = (
            torch.sum(torch.pow(encoder_grad_filtered - encoder_mean, 2)) / encoder_nonzero_count
        ).detach().item() if encoder_mean is not None else 0.0

        current_ratio = (
            (torch.sum(target_grad_filtered) / target_nonzero_count)
            / (torch.sum(encoder_grad_filtered) / encoder_nonzero_count)
            if target_nonzero_count > 0 and encoder_nonzero_count > 0
            else torch.tensor(1.0, device=self.device)
        )

        self._ema_gm = self.smoothing * self._ema_gm + (1 - self.smoothing) * current_gm.detach().item()
        self._ema_gh = self.smoothing * self._ema_gh + (1 - self.smoothing) * current_gh.detach().item()
        self._ema_ratio = self.smoothing * self._ema_ratio + (1 - self.smoothing) * current_ratio.detach().item()

        self._is_initialized = True

        return current_gm.detach(), current_gh.detach(), current_ratio.detach()

    def compute_loss_and_backward(self, main_loss, target_mask, encoder_mask, effective_mask,
                                  current_epoch, total_epochs,
                                  alpha_early=None, alpha_late=None):
        """Compute loss and perform backward pass with contrastive regularization.
        
        注意：已移除梯度缩放逻辑，水印区域参数在优化器中直接冻结。
        """
        device = main_loss.device
        if self.device is None:
            self.set_device(device)

        main_loss.backward()
        return main_loss

    def update_param_snapshot(self, model):
        if model is None:
            return

        snapshot = {}
        total_sq = 0.0
        wm_sq = 0.0

        for name, param in model.named_parameters():
            if param is None:
                continue
            tensor = param.detach().cpu().clone()
            snapshot[name] = tensor
            total_sq += tensor.float().pow(2).sum().item()
            if 'weight' in name:
                wm_sq += tensor.float().pow(2).sum().item()

        self._param_snapshot = snapshot
        self._total_param_drift = total_sq ** 0.5
        self._wm_param_drift = wm_sq ** 0.5

    def get_param_drift_stats(self, model):
        if self._param_snapshot is None or model is None:
            return {
                'total_param_drift': 0.0,
                'wm_param_drift': 0.0,
                'drift_ratio': 0.0,
            }

        total_sq = 0.0
        wm_sq = 0.0

        for name, param in model.named_parameters():
            if name not in self._param_snapshot:
                continue
            diff = (param.detach().cpu() - self._param_snapshot[name]).float()
            diff_sq = diff.pow(2).sum().item()
            total_sq += diff_sq
            if 'weight' in name:
                wm_sq += diff_sq

        total_drift = total_sq ** 0.5
        wm_drift = wm_sq ** 0.5
        drift_ratio = wm_drift / (total_drift + 1e-12)

        self._total_param_drift = total_drift
        self._wm_param_drift = wm_drift
        return {
            'total_param_drift': total_drift,
            'wm_param_drift': wm_drift,
            'drift_ratio': drift_ratio,
        }

    def get_stats(self):
        return {
            'prevGM': self._ema_gm,
            'prevGH': self._ema_gh,
            'prevRatio': self._ema_ratio,
            'current_grad_M': self.current_grad_M,
            'current_grad_H': self.current_grad_H,
            'current_var_M': self.current_var_M,
            'current_var_H': self.current_var_H,
            'is_initialized': self._is_initialized,
            'current_wm_grad': self.current_wm_grad,
            'current_non_wm_grad': self.current_non_wm_grad,
            'current_wm_ratio': self.current_wm_ratio,
        }

    def reset_stats(self):
        self._ema_gm = 0.0
        self._ema_gh = 0.0
        self._ema_ratio = 1.0
        self._is_initialized = False
        self.current_grad_M = 0.0
        self.current_grad_H = 0.0
        self.current_var_M = 0.0
        self.current_var_H = 0.0
        self.current_wm_grad = 0.0
        self.current_non_wm_grad = 0.0
        self.current_wm_ratio = 1.0
        self._param_snapshot = None
        self._total_param_drift = 0.0
        self._wm_param_drift = 0.0
