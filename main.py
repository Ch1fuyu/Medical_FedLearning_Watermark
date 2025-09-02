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

set_seed()

# 配置 logging
log_file_name = './logs/console.logs'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',  # 日期格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_file_name, mode='a')  # 追加模式
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
        logging.info(f'Using ChestMNIST dataset with {self.num_classes} classes (multi-label)')
        self.in_channels = 3  # RGB图像
            
        self.train_set, self.test_set, self.dict_users = get_data(dataset_name=self.dataset,
                                                                  data_root=self.data_root,
                                                                  iid=self.iid,
                                                                  client_num=self.client_num,
                                                                  )
        logging.info('==> Training model...')
        self.logs = {'best_train_acc': -np.inf, 'best_train_loss': -np.inf,
                     'val_acc': [], 'val_loss': [],
                     'best_test_acc': -np.inf, 'best_test_loss': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.construct_model()
        self.w_t = copy.deepcopy(self.model.state_dict())

        # 将随机参数位置分配给每个客户端
        self.random_positions = construct_random_wm_position(self.model, self.client_num)
        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma, self.random_positions)
        self.tester = TesterPrivate(self.model, self.device)

    def construct_model(self):
        if self.model_name == 'resnet':
            model = resnet18(num_classes=self.num_classes, in_channels=self.in_channels, input_size=28)
            logging.info('Using ResNet-18 model')
        else:
            model = AlexNet(self.in_channels, self.num_classes)
            logging.info('Using AlexNet model')
        self.model = model.to(self.device)
        logging.info(f'Model created with {self.in_channels} input channels and {self.num_classes} output classes')

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

        # Early Stopping 配置
        patience = 10
        early_stop_counter = 0
        best_val_auc = -np.inf

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

            # 学习率调度（在第50和75轮时衰减0.1倍）
            if (epoch + 1) in [50, 75]:
                self.lr *= 0.1
                logging.info(f'LR decayed at epoch {epoch + 1}. New lr: {self.lr}')

            client_weights = []
            for i in range(self.client_num):
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[i])) / len(self.train_set)
                client_weights.append(client_weight)

            self._fed_avg(local_ws, client_weights)
            self.model.load_state_dict(self.w_t)

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                train_metrics = self.trainer.test(train_ldr)
                val_metrics = self.trainer.test(val_ldr)

                loss_train_mean, acc_train_mean, auc_train = train_metrics
                loss_val_mean, acc_val_mean, auc_val = val_metrics

                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))

                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = [copy.deepcopy(self.model.state_dict())]

                if self.logs['best_train_acc'] < acc_train_mean:
                    self.logs['best_train_acc'] = acc_train_mean
                    self.logs['best_train_loss'] = loss_train_mean

                logging.info(
                    "Train Loss {:.4f} --- Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean))
                logging.info("Train acc {:.4f} (AUC {:.4f}) --- Val acc {:.4f} (AUC {:.4f}) --Best acc {:.4f}"
                             .format(acc_train_mean, auc_train, acc_val_mean, auc_val, self.logs['best_test_acc']))

                # Early Stopping：基于验证AUC
                if auc_val > best_val_auc:
                    best_val_auc = auc_val
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val AUC: {best_val_auc:.4f}')
                        break

        logging.info('-------------------------------Result--------------------------------------')
        logging.info('Test loss: {:.4f} --- Test acc: {:.4f}'.format(self.logs['best_test_loss'],
                                                                     self.logs['best_test_acc']))
        end = time.time()
        logging.info('Time: {:.1f} min'.format((end - start) / 60))
        logging.info('-------------------------------Finish--------------------------------------')

        return self.logs, self.logs['best_test_acc']

    def _fed_avg(self, local_ws, client_weights):
        w_avg = copy.deepcopy(local_ws[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i]

            self.w_t[k] = w_avg[k]

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
    logg, test_acc = fl.training()
    logs['net_info'] = logg
    logs['test_acc'] = {'value': test_acc}
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
                   args.client_num, args.frac, test_acc
               ))

    return

if __name__ == '__main__':
    args = parser_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)