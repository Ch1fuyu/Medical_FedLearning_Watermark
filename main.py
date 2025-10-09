import copy
import os
import sys
import time
from datetime import datetime
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.globals import set_seed
from models.alexnet import AlexNet
from models.resnet import resnet18
from utils.args import parser_args
from utils.base import Experiment
from utils.dataset import get_data, DatasetSplit
from utils.trainer_private import TrainerPrivate, TesterPrivate
from utils.trainer_private_enhanced import TrainerPrivateEnhanced
import pandas as pd

set_seed()

# 配置 logging
args = parser_args()
log_file_name = args.log_file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',  # 日期格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')  # 追加模式
    ]
)


class FederatedLearningOnChestMNIST(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.random_positions = None
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma
        self.key_matrix_dir = getattr(args, 'key_matrix_dir', './save/key_matrix')
        
        logging.info('--------------------------------Start--------------------------------------')
        logging.info(args)
        logging.info('==> Preparing data...')
        logging.info('==> 使用普通水印系统')
        
        # 数据集配置
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels
            
        self.train_set, self.test_set, self.dict_users = get_data(dataset_name=self.dataset,
                                                                  data_root=self.data_root,
                                                                  iid=self.iid,
                                                                  client_num=self.client_num,
                                                                  )
        logging.info('==> Training model...')
        self.logs = {'best_train_acc': -np.inf, 'best_train_loss': -np.inf,
                     'val_acc': [], 'val_loss': [],
                     'best_model_acc': -np.inf, 'best_model_loss': -np.inf,
                     'best_model_auc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     # 独立跟踪历史最高指标
                     'highest_acc_ever': -np.inf,     # 历史最高准确率（纯准确率）
                     'highest_auc_ever': -np.inf,     # 历史最高AUC（纯AUC）
                     'acc_when_highest_auc': -np.inf, # 达到历史最高AUC时的准确率
                     'auc_when_highest_acc': -np.inf, # 达到历史最高准确率时的AUC
                     }

        self.construct_model()
        self.w_t = copy.deepcopy(self.model.state_dict())

        # 根据参数选择水印模式
        self.random_positions = {}
        # 设置密钥矩阵目录
        self.args.key_matrix_dir = self.key_matrix_dir
        self.args.use_key_matrix = True
        
        # 根据watermark_mode参数选择trainer
        if self.args.watermark_mode == 'enhanced':
            logging.info('==> 使用增强水印系统（密钥矩阵 + 自编码器）')
            self.trainer = TrainerPrivateEnhanced(self.model, self.device, self.dp, self.sigma, self.random_positions, self.args)
        else:
            logging.info('==> 使用普通水印系统')
            self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma, self.random_positions, self.args)
            
        self.tester = TesterPrivate(self.model, self.device)

    def construct_model(self):
        if self.model_name == 'resnet':
            model = resnet18(num_classes=self.num_classes, in_channels=self.in_channels, input_size=28)
        else:
            model = AlexNet(self.in_channels, self.num_classes)
        self.model = model.to(self.device)

    def training(self):
        start = time.time()
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)

        local_train_loader = []

        for i in range(self.client_num):
            local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]),
                                         batch_size=self.batch_size,
                                         shuffle=True, num_workers=2)
            local_train_loader.append(local_train_ldr)

        idxs_users = []

        # Early Stopping 配置 - 基于准确率
        patience = self.args.patience  # 从参数中获取耐心值
        early_stop_counter = 0
        best_val_acc = -np.inf
        best_val_auc = -np.inf

        # 统计记录
        stats_rows = []

        for epoch in range(self.epochs): # 均匀采样，frac 默认为 1，即每轮中全体客户端参与训练
            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.client_num), 1)
                idxs_users = np.random.choice(range(self.client_num), self.m, replace=False)

            local_ws, local_losses = [], []

            logging.info('Epoch: %d / %d, lr: %f' % (epoch + 1, self.epochs, self.lr))
            for idx in tqdm(idxs_users, desc='Progress: %d / %d' % (epoch + 1, self.epochs)):
                self.model.load_state_dict(self.w_t)

                # 统一调用：始终传入 current_epoch/total_epochs；
                # 普通 Trainer 会通过 **kwargs 忽略
                local_w, local_loss, local_acc = self.trainer.local_update(
                    dataloader=local_train_loader[idx], 
                    local_ep=self.local_ep, 
                    lr=self.lr, 
                    client_id=idx,
                    current_epoch=epoch,
                    total_epochs=self.epochs
                )

                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)

            # 学习率调度 - MultiStepLR
            milestones = [int(self.epochs * m) for m in self.args.lr_decay_milestones]
            if (epoch + 1) in milestones:
                self.lr *= self.args.lr_decay_gamma
                logging.info(f'LR decayed at epoch {epoch + 1} (milestone: {milestones}). New lr: {self.lr}')

            # 计算参与训练的客户端的权重（相对于总数据集）
            client_weights = []
            for idx in idxs_users:
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[idx])) / len(self.train_set)
                client_weights.append(client_weight)

            # 更新全局模型权重
            self._fed_avg(local_ws, client_weights, idxs_users)
            self.model.load_state_dict(self.w_t)

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                train_metrics = self.trainer.test(train_ldr)
                val_metrics = self.trainer.test(val_ldr)

                # (loss, acc_label, auc, acc_sample)
                loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean = train_metrics
                loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean = val_metrics

                self.logs['val_acc'].append(acc_val_label_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))

                # 更新历史最高值跟踪
                if self.logs['highest_acc_ever'] < acc_val_label_mean:
                    self.logs['highest_acc_ever'] = acc_val_label_mean
                    self.logs['auc_when_highest_acc'] = auc_val
                    
                if self.logs['highest_auc_ever'] < auc_val:
                    self.logs['highest_auc_ever'] = auc_val
                    self.logs['acc_when_highest_auc'] = acc_val_label_mean

                # 模型选择标准：以验证集AUC为准，只有AUC提升才保存模型
                if self.logs['best_model_auc'] < auc_val:
                    self.logs['best_model_acc'] = acc_val_label_mean
                    self.logs['best_model_loss'] = loss_val_mean
                    self.logs['best_model_auc'] = auc_val
                    self.logs['best_model'] = [copy.deepcopy(self.model.state_dict())]
                    logging.info(f'New best model saved! AUC improved to {auc_val:.4f}')

                if self.logs['best_train_acc'] < acc_train_label_mean:
                    self.logs['best_train_acc'] = acc_train_label_mean
                    self.logs['best_train_loss'] = loss_train_mean

                logging.info(
                    f"Train Loss {loss_train_mean:.4f} --- Val Loss {loss_val_mean:.4f}")
                logging.info(
                    f"Train: acc(label) {acc_train_label_mean:.4f}, acc(sample) {acc_train_sample_mean:.4f} (AUC {auc_train:.4f}) | "
                    f"Val: acc(label) {acc_val_label_mean:.4f}, acc(sample) {acc_val_sample_mean:.4f} (AUC {auc_val:.4f}) | "
                    f"Highest ACC: {self.logs['highest_acc_ever']:.4f} | Highest AUC: {self.logs['highest_auc_ever']:.4f}")
                
                # 打印增强水印系统统计信息
                if hasattr(self.trainer, 'multi_loss'):
                    stats = self.trainer.multi_loss.get_stats()
                    logging.info(f"MultiLoss统计 - prevGM: {stats['prevGM']:.6f}, prevGH: {stats['prevGH']:.6f}, prevRatio: {stats['prevRatio']:.6f}")


                # Early Stopping：基于验证AUC
                if auc_val > best_val_auc:
                    best_val_auc = auc_val
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val AUC: {best_val_auc:.4f}')
                        break

                # 记录本轮统计数据
                stats_row = {
                    'round': epoch + 1,
                    'lr': self.lr,
                    'train_loss': float(loss_train_mean),
                    'val_loss': float(loss_val_mean),
                    'train_acc_label': float(acc_train_label_mean),
                    'train_auc': float(auc_train),
                    'val_acc_label': float(acc_val_label_mean),
                    'val_auc': float(auc_val),
                    'best_val_acc_so_far': float(self.logs['highest_acc_ever']),
                    'best_val_auc_so_far': float(self.logs['highest_auc_ever']),
                    'train_acc_sample': float(acc_train_sample_mean),
                    'val_acc_sample': float(acc_val_sample_mean),
                }
                
                # 添加增强水印系统统计信息
                if hasattr(self.trainer, 'multi_loss'):
                    multi_loss_stats = self.trainer.multi_loss.get_stats()
                    stats_row.update({
                        'prevGM': float(multi_loss_stats['prevGM']),
                        'prevGH': float(multi_loss_stats['prevGH']),
                        'prevRatio': float(multi_loss_stats['prevRatio']),
                        'current_grad_M': float(multi_loss_stats['current_grad_M']),
                        'current_grad_H': float(multi_loss_stats['current_grad_H']),
                        'current_var_M': float(multi_loss_stats['current_var_M']),
                        'current_var_H': float(multi_loss_stats['current_var_H']),
                    })
                
                stats_rows.append(stats_row)

        logging.info('-------------------------------Result--------------------------------------')
        logging.info(
            f'Test loss: {self.logs["best_model_loss"]:.4f} --- Test acc: {self.logs["best_model_acc"]:.4f} --- Test auc: {self.logs["best_model_auc"]:.4f}')
        logging.info('历史最高统计:')
        logging.info(
            f'  历史最高准确率: {self.logs["highest_acc_ever"]:.4f} (对应AUC: {self.logs["auc_when_highest_acc"]:.4f})')
        logging.info(
            f'  历史最高AUC: {self.logs["highest_auc_ever"]:.4f} (对应准确率: {self.logs["acc_when_highest_auc"]:.4f})')
        end = time.time()
        logging.info('Time: {:.1f} min'.format((end - start) / 60))
        logging.info('-------------------------------Finish--------------------------------------')

        # 导出Excel
        try:
            os.makedirs(self.args.save_excel_dir, exist_ok=True)
            # 基础列 + 增强水印系统统计列
            columns = ['round', 'lr', 'train_loss', 'val_loss', 'train_acc_label', 'train_auc', 
                     'val_acc_label', 'val_auc', 'best_val_acc_so_far', 'best_val_auc_so_far', 
                     'train_acc_sample', 'val_acc_sample', 'prevGM', 'prevGH', 'prevRatio', 
                     'current_grad_M', 'current_grad_H', 'current_var_M', 'current_var_H']
            df = pd.DataFrame(stats_rows, columns=columns)
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_path = f'{self.args.save_excel_dir}/metrics_{self.model_name}_{self.dataset}_{now}.xlsx'
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logging.info(f'Excel metrics saved to: {excel_path}')
        except Exception as e:
            logging.warning(f'Failed to export Excel metrics: {e}')

        return self.logs, self.logs['best_model_auc']


    def _fed_avg(self, local_ws, client_weights, idxs_users):
        """联邦平均算法，FedAvg with exclusive watermark aggregation"""
        # 计算参与训练的客户端权重总和
        total_weight = sum(client_weights)
        
        # 归一化权重，确保权重和为1
        normalized_weights = [w / total_weight for w in client_weights]
        
        # 验证权重和是否为1
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > self.args.weight_tolerance:
            logging.warning(f"Weight sum is {weight_sum:.6f}, not 1.0. Normalizing...")
            normalized_weights = [w / weight_sum for w in normalized_weights]
        
        # 初始化平均权重
        w_avg = copy.deepcopy(local_ws[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * normalized_weights[0]

        # 累加其他客户端的权重
        for i in range(1, len(local_ws)):
            for k in w_avg.keys():
                w_avg[k] += local_ws[i][k] * normalized_weights[i]

        # 水印聚合：使用密钥矩阵的独占式聚合
        self._watermark_aggregation(local_ws, idxs_users, w_avg)

        # 更新全局模型权重
        for k in w_avg.keys():
            self.w_t[k] = w_avg[k]

    def _watermark_aggregation(self, local_ws, idxs_users, w_avg):
        """
        水印聚合：使用密钥矩阵的独占式聚合
        
        Args:
            local_ws: 本地模型权重列表
            idxs_users: 参与训练的客户端ID列表
            w_avg: 平均聚合后的权重字典
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            
            # 加载密钥矩阵管理器
            key_manager = KeyMatrixManager(self.key_matrix_dir)
            
            # 对每个客户端的水印位置进行独占式聚合
            for i, client_id in enumerate(idxs_users):
                try:
                    # 获取该客户端的水印位置
                    positions = key_manager.load_positions(client_id)
                    
                    # 对该客户端的水印位置使用独占式聚合
                    for param_name, param_idx in positions:
                        if param_name in local_ws[i] and param_name in w_avg:
                            # 获取参数张量
                            local_param = local_ws[i][param_name]
                            avg_param = w_avg[param_name]
                            
                            # 确保参数形状一致
                            if local_param.shape == avg_param.shape:
                                # 将局部索引转换为扁平化索引
                                param_flat = avg_param.view(-1)
                                local_flat = local_param.view(-1)
                                
                                # 使用局部索引直接替换
                                if param_idx < param_flat.numel():
                                    param_flat[param_idx] = local_flat[param_idx]
                                    
                except Exception as e:
                    logging.warning(f"Failed to apply watermark aggregation for client {client_id}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Failed to load key matrix manager for watermark aggregation: {e}")

def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'client_num': args.client_num,
                'console_log': os.path.basename(log_file_name),
            }
            }
    fl = FederatedLearningOnChestMNIST(args)
    logg, test_auc = fl.training()
    logs['net_info'] = logg
    logs['test_auc'] = {'value': test_auc}
    logs['bp_local'] = {'value': True if args.bp_interval == 0 else False}

    save_dir = os.path.join(args.save_model_dir, args.model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    # 构建文件名
    enhanced = "_enhanced" if args.watermark_mode == 'enhanced' else ""

    file_name = '{}_Dp_{}_iid_{}_ns_{}_wt_{}_lt_{}_ep_{}_le_{}_cn_{}_fra_{:.4f}_auc_{:.4f}{}.pkl'.format(
        formatted_now, args.sigma, args.iid, args.num_sign, args.weight_type, args.loss_type,
        args.epochs, args.local_ep, args.client_num, args.frac, test_auc, enhanced
    )
    torch.save(logs, os.path.join(save_dir, file_name))

    return

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)