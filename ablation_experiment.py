import copy
import os
import sys
import time
from datetime import datetime
import logging
import gc

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
from utils.trainer_ablation import TrainerAblation
from utils.trainer_private import TesterPrivate
import pandas as pd

set_seed()

# 配置 logging
args = parser_args()
log_file_name = args.log_file.replace('.logs', '_ablation.logs')  # 消融实验使用不同的日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
    ]
)


class AblationExperiment(Experiment):
    """
    消融实验：只使用主任务损失，不使用三个正则项
    但保持水印嵌入逻辑不变（使用KeyMatrixManager进行参数替换）
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma
        self.key_matrix_dir = getattr(args, 'key_matrix_dir', './save/key_matrix')
        
        logging.info('='*60)
        logging.info('消融实验：只使用主任务损失，不使用三个正则项')
        logging.info('水印嵌入逻辑保持不变（使用KeyMatrixManager进行参数替换）')
        logging.info('='*60)
        logging.info('--------------------------------Start--------------------------------------')
        logging.info(args)
        logging.info('==> Preparing data...')
        
        # 数据集配置
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels
            
        # 确保数据根目录存在
        os.makedirs(self.data_root, exist_ok=True)

        self.train_set, self.test_set, self.dict_users = get_data(dataset_name=self.dataset,
                                                                  data_root=self.data_root,
                                                                  )
        logging.info('==> Training model...')
        self.logs = {'best_train_acc': -np.inf, 'best_train_loss': -np.inf,
                     'val_acc': [], 'val_loss': [],
                     'best_model_acc': -np.inf, 'best_model_loss': -np.inf,
                     'best_model_auc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     # 独立跟踪历史最高指标
                     'highest_acc_ever': -np.inf,
                     'highest_auc_ever': -np.inf,
                     'acc_when_highest_auc': -np.inf,
                     'auc_when_highest_acc': -np.inf,
                     }

        self.construct_model()
        self.w_t = copy.deepcopy(self.model.state_dict())

        # 设置密钥矩阵目录
        self.args.key_matrix_dir = self.key_matrix_dir
        self.args.use_key_matrix = True
        
        # 使用消融实验训练器（只使用主任务损失）
        logging.info('==> 使用消融实验训练器（只使用主任务损失，不使用正则项）')
        self.trainer = TrainerAblation(self.model, self.device, self.dp, self.sigma, self.args)
        self.tester = TesterPrivate(self.model, self.device, args=self.args)

    def construct_model(self):
        if self.model_name == 'resnet':
            model = resnet18(num_classes=self.num_classes, in_channels=self.in_channels, input_size=self.args.input_size)
        else:
            dropout_rate = getattr(self.args, 'dropout_rate', 0.5)
            model = AlexNet(self.in_channels, self.num_classes, input_size=self.args.input_size, dropout_rate=dropout_rate)
        self.model = model.to(self.device)

    def _cleanup_memory(self):
        """清理内存和GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _optimize_model_storage(self, model_state):
        """优化模型状态存储，减少内存占用"""
        optimized_state = {}
        for key, value in model_state.items():
            optimized_state[key] = value.detach().cpu()
        return optimized_state

    def training(self):
        start = time.time()
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)

        local_train_loader = []

        num_clients = len(self.dict_users)
        for i in range(num_clients):
            local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]),
                                         batch_size=self.batch_size,
                                         shuffle=True, num_workers=0, pin_memory=False)
            local_train_loader.append(local_train_ldr)

        idxs_users = []

        # Early Stopping 配置
        patience = 10
        early_stop_counter = 0
        best_val_acc = -np.inf
        best_val_auc = -np.inf

        # 决定模型选择依据：ChestMNIST 按 AUC，其他按准确率
        dataset_key = (self.dataset or '').lower()
        select_by_auc = (dataset_key == 'chestmnist')

        # 统计记录
        stats_rows = []

        for epoch in range(self.epochs):
            # 均匀采样
            self.m = max(int(self.frac * num_clients), 1)
            idxs_users = np.random.choice(range(num_clients), self.m, replace=False)

            local_ws, local_losses = [], []

            logging.info('Epoch: %d / %d' % (epoch + 1, self.epochs))
            
            # 消融实验：不需要自编码器微调，只进行水印参数替换
            
            for i, idx in enumerate(tqdm(idxs_users, desc='Progress: %d / %d' % (epoch + 1, self.epochs))):
                self.model.load_state_dict(self.w_t)

                # 调用消融实验训练器
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
                
                # 清理临时变量
                del local_w, local_loss, local_acc

            # 计算参与训练的客户端的权重
            client_weights = []
            for idx in idxs_users:
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[idx])) / len(self.train_set)
                client_weights.append(client_weight)

            # 更新全局模型权重
            self._fed_avg(local_ws, client_weights, idxs_users)
            
            # 加载聚合后的权重
            self.model.load_state_dict(self.w_t)
            self.model.train()
            
            # 清理优化器状态
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                self.trainer.optimizer.zero_grad()

            # 评估模型
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

                # 模型选择标准：ChestMNIST 按 AUC；否则按准确率
                if select_by_auc:
                    if self.logs['best_model_auc'] < auc_val:
                        self.logs['best_model_acc'] = acc_val_label_mean
                        self.logs['best_model_loss'] = loss_val_mean
                        self.logs['best_model_auc'] = auc_val
                        optimized_state = self._optimize_model_storage(self.model.state_dict())
                        self.logs['best_model'] = [optimized_state]
                        logging.info(f'🌟 最佳模型已保存! AUC↑{auc_val:.4f}')
                else:
                    if self.logs['best_model_acc'] < acc_val_label_mean:
                        self.logs['best_model_acc'] = acc_val_label_mean
                        self.logs['best_model_loss'] = loss_val_mean
                        self.logs['best_model_auc'] = auc_val
                        optimized_state = self._optimize_model_storage(self.model.state_dict())
                        self.logs['best_model'] = [optimized_state]
                        logging.info(f'🌟 最佳模型已保存! ACC↑{acc_val_label_mean:.4f}')

                if self.logs['best_train_acc'] < acc_train_label_mean:
                    self.logs['best_train_acc'] = acc_train_label_mean
                    self.logs['best_train_loss'] = loss_train_mean

                # 合并训练和验证指标到一行
                logging.info(
                    f"轮次{epoch+1} | Train Loss:{loss_train_mean:.4f} Acc:{acc_train_label_mean:.4f} AUC:{auc_train:.4f} | "
                    f"Val Loss:{loss_val_mean:.4f} Acc:{acc_val_label_mean:.4f} AUC:{auc_val:.4f} | "
                    f"Best Acc:{self.logs['highest_acc_ever']:.4f} Best AUC:{self.logs['highest_auc_ever']:.4f}")
                
                # 记录本轮统计数据（消融实验不包含MultiLoss统计）
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
                
                # Early Stopping：ChestMNIST 基于 AUC，其它数据集基于准确率
                if select_by_auc:
                    if auc_val > best_val_auc:
                        best_val_auc = auc_val
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val AUC: {best_val_auc:.4f}')
                            break
                else:
                    if acc_val_label_mean > best_val_acc:
                        best_val_acc = acc_val_label_mean
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val ACC: {best_val_acc:.4f}')
                            break
                
                # 每轮训练后清理内存
                self._cleanup_memory()
                
                # 清理临时变量
                del train_metrics, val_metrics
                del loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean
                del loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean
                
                stats_rows.append(stats_row)

        logging.info('='*60 + ' 训练结果 ' + '='*60)
        logging.info(
            f'最佳模型 | Loss:{self.logs["best_model_loss"]:.4f} Acc:{self.logs["best_model_acc"]:.4f} AUC:{self.logs["best_model_auc"]:.4f}')
        logging.info(
            f'历史最高 | Acc:{self.logs["highest_acc_ever"]:.4f}(AUC:{self.logs["auc_when_highest_acc"]:.4f}) | '
            f'AUC:{self.logs["highest_auc_ever"]:.4f}(Acc:{self.logs["acc_when_highest_auc"]:.4f})')
        end = time.time()
        logging.info(f'训练耗时: {(end - start) / 60:.1f} 分钟')
        
        # 最终内存清理
        self._cleanup_memory()
        logging.info('🧹 训练完成，已清理内存缓存')

        # 导出Excel
        try:
            os.makedirs(self.args.save_excel_dir, exist_ok=True)
            # 消融实验只包含基础列，不包含MultiLoss统计
            columns = ['round', 'lr', 'train_loss', 'val_loss', 'train_acc_label', 'train_auc', 
                     'val_acc_label', 'val_auc', 'best_val_acc_so_far', 'best_val_auc_so_far', 
                     'train_acc_sample', 'val_acc_sample']
            df = pd.DataFrame(stats_rows, columns=columns)
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_path = f'{self.args.save_excel_dir}/metrics_ablation_{self.model_name}_{self.dataset}_{now}.xlsx'
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logging.info(f'Excel metrics saved to: {excel_path}')
        except Exception as e:
            logging.warning(f'Failed to export Excel metrics: {e}')

        # 返回用于模型选择的指标
        best_metric_value = self.logs['best_model_auc'] if select_by_auc else self.logs['best_model_acc']
        best_metric_name = 'auc' if select_by_auc else 'acc'
        return self.logs, best_metric_value, best_metric_name

    def _fed_avg(self, local_ws, client_weights, idxs_users):
        """联邦平均算法，FedAvg with exclusive watermark aggregation"""
        # 计算参与训练的客户端权重总和
        total_weight = sum(client_weights)
        
        # 归一化权重
        normalized_weights = [w / total_weight for w in client_weights]
        
        # 验证权重和是否为1
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logging.warning(f"Weight sum is {weight_sum:.6f}, not 1.0. Normalizing...")
            normalized_weights = [w / weight_sum for w in normalized_weights]
        
        # 初始化平均权重
        w_avg = {}
        for k in local_ws[0].keys():
            w_avg[k] = local_ws[0][k].clone() * normalized_weights[0]

        # 累加其他客户端的权重
        for i in range(1, len(local_ws)):
            for k in w_avg.keys():
                w_avg[k] += local_ws[i][k] * normalized_weights[i]

        # 水印聚合：使用密钥矩阵的独占式聚合（消融实验保持此逻辑不变）
        self._watermark_aggregation(local_ws, idxs_users, w_avg)

        # 更新全局模型权重
        for k in w_avg.keys():
            self.w_t[k] = w_avg[k]

    def _watermark_aggregation(self, local_ws, idxs_users, w_avg):
        """
        水印聚合：使用密钥矩阵的独占式聚合
        消融实验保持此逻辑不变
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            
            # 加载密钥矩阵管理器
            key_manager = KeyMatrixManager(self.args.key_matrix_path)
            
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
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'console_log': os.path.basename(log_file_name),
            }
            }
    fl = AblationExperiment(args)
    logg, best_metric_value, best_metric_name = fl.training()
    logs['net_info'] = logg
    # 兼容：总是记录 test_auc；当指标为 acc 时也额外记录
    logs['test_auc'] = {'value': logg.get('best_model_auc', best_metric_value if best_metric_name == 'auc' else 0.0)}
    if best_metric_name == 'acc':
        logs['test_acc'] = {'value': best_metric_value}
    logs['bp_local'] = {'value': False}

    save_dir = os.path.join(args.save_model_dir, args.model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    # 构建文件名
    enhanced = "_enhanced" if args.watermark_mode == 'enhanced' else ""

    # 根据选择指标命名文件
    watermark_suffix = f"wm_{args.watermark_mode}" if hasattr(args, 'watermark_mode') and args.watermark_mode else "wm_basic"
    file_name = '{}_Dp_{}_iid_{}_{}_ep_{}_le_{}_fra_{:.4f}_{}_{{:.4f}}{}.pkl'.format(
        formatted_now, args.sigma, args.iid, watermark_suffix,
        args.epochs, args.local_ep, args.frac, best_metric_name, enhanced
    )
    file_name = file_name.format(best_metric_value)
    torch.save(logs, os.path.join(save_dir, file_name))
    logging.info(f"训练日志已保存: {file_name}")
    logging.info('-------------------------------Finish--------------------------------------')

    return


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)

