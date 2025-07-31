import copy
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.globals import set_seed
from models.alexnet import AlexNet
from utils.args import parser_args
from utils.base import Experiment
from utils.dataset import get_data_no_fl
from utils.trainer_private import TrainerPrivate, TesterPrivate

set_seed()

class AlexNetTrainingOnCifar10(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma

        print('==> Preparing data...')
        self.train_set, self.test_set = get_data_no_fl(dataset_name=self.dataset,
                                                    data_root=self.data_root
                                                    )

        print('==> Training model...')
        self.logs = {'train_acc': [], 'train_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.construct_model()
        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma)
        self.tester = TesterPrivate(self.model, self.device)

    def construct_model(self):
        model = AlexNet(self.in_channels, self.num_classes)
        self.model = model.to(self.device)

    def training(self):
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)

        for _ in tqdm(range(self.epochs), desc="Training Progress: "):
            self.trainer.local_update_with_no_fl(train_ldr, self.local_ep, self.lr)
            self.lr = self.lr * 0.99

            loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
            loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)

            self.logs['train_acc'].append(acc_train_mean)
            self.logs['train_loss'].append(loss_train_mean)
            self.logs['val_acc'].append(acc_val_mean)
            self.logs['val_loss'].append(loss_val_mean)

            if self.logs['best_test_acc'] < acc_val_mean:
                self.logs['best_test_acc'] = acc_val_mean
                self.logs['best_test_loss'] = loss_val_mean
                self.logs['best_model'] = [copy.deepcopy(self.model.state_dict())] # TODO

            print(
                "Train Loss {:.4f} --- Val Loss {:.4f}"
                .format(loss_train_mean, loss_val_mean))
            print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(acc_train_mean, acc_val_mean,
                                                                                 self.logs[
                                                                                     'best_test_acc']
                                                                                 )
                  )

        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}'.format(self.logs['best_test_loss'],
                                                              self.logs['best_test_acc']))

        return self.logs, self.logs['best_test_acc']


def main(args):
    print(args)
    logs = {'net_info': None,
            'arguments': {
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr': args.lr,
                'model_name': args.model_name,
                'dataset': args.dataset,
                'epochs': args.epochs,
            }
            }
    trainer = AlexNetTrainingOnCifar10(args)
    logg, test_acc = trainer.training()
    logs['net_info'] = logg
    logs['test_acc'] = {'value': test_acc} # TODO

    save_dir = 'D:/CODE/FL_Cifar/save/no_fl/'

    if not os.path.exists(save_dir + args.model_name + '/' + args.dataset):
        os.makedirs(save_dir + args.model_name + '/' + args.dataset)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    torch.save(logs,
               save_dir + args.model_name + '/' + args.dataset + '/{}_Dp_{}_sig_{}_ep_{}_acc_{:.4f}.pkl'.format(
                   formatted_now, args.dp, args.sigma, args.epochs,  test_acc
               ))
    return

if __name__ == '__main__':
    args = parser_args()
    args.epochs = 100
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)