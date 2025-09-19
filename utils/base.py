import csv
import os

import torch

import config.globals


class Experiment(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        self.prefix = ''
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.experiment_id = args.exp_id
        self.buffer = []
        self.save_history_interval = 1
        self.device = torch.device('cuda')

        self.client_num = args.client_num
        self.num_back = args.num_back
        self.num_sign = args.num_sign
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.iid = args.iid
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.save_interval = args.save_interval
        self.loss_type = args.loss_type
        self.weight_type = args.weight_type
        self.in_channels = 3
        self.sampling_type = args.sampling_type

        if args.dataset == 'cifar10':
            self.num_classes = 10
        if args.dataset == 'cifar100':
            self.num_classes = 100
        ## federated learning args
        self.frac = args.frac
        self.data_root = config.globals.data_root
        self.local_ep = args.local_ep

        self.sl_ratio = args.loss_alpha
        self.logdir = f'logs/{self.model_name}_{self.dataset}'