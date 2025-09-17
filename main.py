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
from utils.dataset import get_data, DatasetSplit, construct_random_wm_position
from utils.trainer_private import TrainerPrivate, TesterPrivate
import pandas as pd

set_seed()

# 配置 logging
log_file_name = './logs/console.logs'
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
        logging.info('--------------------------------Start--------------------------------------')
        logging.info(args)
        logging.info('==> Preparing data...')
        
        # ChestMNIST多标签分类设置
        self.num_classes = 14  # ChestMNIST是多标签分类，14个病理标签
        self.in_channels = 3  # RGB图像
            
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
                     'highest_acc_ever': -np.inf,      # 历史最高准确率（纯准确率）
                     'highest_auc_ever': -np.inf,     # 历史最高AUC（纯AUC）
                     'acc_when_highest_auc': -np.inf, # 达到历史最高AUC时的准确率
                     'auc_when_highest_acc': -np.inf, # 达到历史最高准确率时的AUC
                     }

        self.construct_model()
        self.w_t = copy.deepcopy(self.model.state_dict())

        # 将随机参数位置分配给每个客户端
        self.random_positions = construct_random_wm_position(self.model, self.client_num)
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

        # 统计记录：轮次，学习率，训练损失，验证损失，训练准确率，训练AUC，验证准确率，验证AUC，
        # 最高验证准确率，最高验证AUC，标签级准确率，样本级准确率
        stats_rows = []

        for epoch in range(self.epochs): # 均匀采样，frac 默认为 1，即每轮中全体客户端参与训练
            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.client_num), 1)
                idxs_users = np.random.choice(range(self.client_num), self.m, replace=False)

            local_ws, local_losses = [], []

            logging.info('Epoch: %d / %d, lr: %f' % (epoch + 1, self.epochs, self.lr))
            for idx in tqdm(idxs_users, desc='Progress: %d / %d' % (epoch + 1, self.epochs)):
                self.model.load_state_dict(self.w_t)

                # 传递客户端的ID给 TrainerPrivate
                local_w, local_loss = self.trainer.local_update(local_train_loader[idx], self.local_ep, self.lr, idx)

                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)

            # 学习率调度 - MultiStepLR (在50%和75%epoch时衰减)
            milestones = [int(self.epochs * 0.5), int(self.epochs * 0.75)]
            if (epoch + 1) in milestones:
                self.lr *= 0.1  # 使用MedMNIST2D的gamma=0.1
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
                    "Train Loss {:.4f} --- Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean))
                logging.info("Train: acc(label) {:.4f}, acc(sample) {:.4f} (AUC {:.4f}) | Val: acc(label) {:.4f}, acc(sample) {:.4f} (AUC {:.4f}) | Highest ACC: {:.4f} | Highest AUC: {:.4f}"
                             .format(acc_train_label_mean, acc_train_sample_mean, auc_train, acc_val_label_mean, acc_val_sample_mean, auc_val,
                                     self.logs['highest_acc_ever'], self.logs['highest_auc_ever']))

                # 添加调试信息：检查模型预测分布
                self._debug_model_predictions(val_ldr, epoch + 1)

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
                stats_rows.append({
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
                })

        logging.info('-------------------------------Result--------------------------------------')
        logging.info('Test loss: {:.4f} --- Test acc: {:.4f} --- Test auc: {:.4f}'.format(self.logs['best_model_loss'],
                                                                                          self.logs['best_model_acc'],
                                                                                          self.logs['best_model_auc']))
        logging.info('历史最高统计:')
        logging.info('  历史最高准确率: {:.4f} (对应AUC: {:.4f})'.format(self.logs['highest_acc_ever'], self.logs['auc_when_highest_acc']))
        logging.info('  历史最高AUC: {:.4f} (对应准确率: {:.4f})'.format(self.logs['highest_auc_ever'], self.logs['acc_when_highest_auc']))
        end = time.time()
        logging.info('Time: {:.1f} min'.format((end - start) / 60))
        logging.info('-------------------------------Finish--------------------------------------')

        # 导出Excel
        try:
            os.makedirs('save/excel', exist_ok=True)
            df = pd.DataFrame(stats_rows,
                              columns=['round','lr','train_loss','val_loss','train_acc_label','train_auc','val_acc_label','val_auc','best_val_acc_so_far','best_val_auc_so_far','train_acc_sample','val_acc_sample'])
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_path = f'save/excel/metrics_{self.model_name}_{self.dataset}_{now}.xlsx'
            df.to_excel(excel_path, index=False)
            logging.info(f'Excel metrics saved to: {excel_path}')
        except Exception as e:
            logging.warning(f'Failed to export Excel metrics: {e}')

        return self.logs, self.logs['best_model_auc']

    def _debug_model_predictions(self, dataloader, epoch):
        """调试模型预测分布"""
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                pred = self.model(data)
                pred_prob = torch.sigmoid(pred)
                pred_binary = pred_prob > 0.5
                
                all_preds.append(pred_binary.cpu())
                all_targets.append(target.cpu())
            
            # 合并所有预测
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 统计预测分布
            pred_normal = torch.all(all_preds == 0, dim=1).sum().item()
            pred_pathological = all_preds.shape[0] - pred_normal
            
            # 统计真实分布
            true_normal = torch.all(all_targets == 0, dim=1).sum().item()
            true_pathological = all_targets.shape[0] - true_normal
            
            # 计算各类别的预测准确率
            label_acc = (all_preds == all_targets).float().mean()
            sample_acc = torch.all(all_preds == all_targets, dim=1).float().mean()
            
            logging.info(f"Epoch {epoch} 调试信息:")
            logging.info(f"  真实分布: 正常{true_normal}个 ({true_normal/len(all_targets)*100:.1f}%), 病理{true_pathological}个 ({true_pathological/len(all_targets)*100:.1f}%)")
            logging.info(f"  预测分布: 正常{pred_normal}个 ({pred_normal/len(all_preds)*100:.1f}%), 病理{pred_pathological}个 ({pred_pathological/len(all_preds)*100:.1f}%)")
            logging.info(f"  标签级准确率: {label_acc*100:.2f}%, 样本级准确率: {sample_acc*100:.2f}%")
            
            # 检查是否总是预测正常
            if (epoch % 10 == 0) & (pred_normal / len(all_preds) > 0.95):
                logging.warning(f"  警告: 模型倾向于总是预测正常 ({pred_normal/len(all_preds)*100:.1f}%)！")

    def _fed_avg(self, local_ws, client_weights, idxs_users):
        """联邦平均算法，正确实现FedAvg"""
        # 计算参与训练的客户端权重总和
        total_weight = sum(client_weights)
        
        # 归一化权重，确保权重和为1
        normalized_weights = [w / total_weight for w in client_weights]
        
        # 验证权重和是否为1
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > 1e-6:
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

        # 更新全局模型权重
        for k in w_avg.keys():
            self.w_t[k] = w_avg[k]
            
        # 记录详细的聚合信息
        # logging.info(f"FedAvg: {len(local_ws)} clients, weights={[f'{w:.4f}' for w in normalized_weights]}")
        # logging.info(f"Participating clients: {idxs_users}")
        
        # 验证聚合算法的正确性
        self._verify_fedavg_correctness(local_ws, normalized_weights)

    def _verify_fedavg_correctness(self, local_ws, normalized_weights):
        """验证联邦平均算法的正确性"""
        # 检查权重和是否为1
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logging.error(f"FedAvg verification failed: weight sum = {weight_sum:.6f}")
            return False
        
        # 检查参数维度一致性
        for i, local_w in enumerate(local_ws):
            for key in local_w.keys():
                if key not in self.w_t:
                    logging.error(f"FedAvg verification failed: key {key} missing in global model")
                    return False
                if local_w[key].shape != self.w_t[key].shape:
                    logging.error(f"FedAvg verification failed: shape mismatch for {key}")
                    return False
        
        return True

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

    save_dir = './save/'

    if not os.path.exists(save_dir + args.model_name + '/' + args.dataset):
        os.makedirs(save_dir + args.model_name + '/' + args.dataset)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    torch.save(logs,
               save_dir + args.model_name + '/' + args.dataset + '/{}_Dp_{}_sig_{}_iid_{}_ns_{}_wt_{}_lt_{}_bit_{}_'
                                                                 'alp_{}_nb_{}_type_{}_tri_{}_ep_{}_le_{}_cn_{}_'
                                                                 'fra_{:.4f}_acc_{:.4f}.pkl'.format(
                   formatted_now, args.dp, args.sigma, args.iid, args.num_sign, args.weight_type, args.loss_type, args.num_bit,
                   args.loss_alpha, args.num_back, args.backdoor_indis, args.num_trigger, args.epochs, args.local_ep,
                   args.client_num, args.frac, test_auc
               ))

    return

if __name__ == '__main__':
    args = parser_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)