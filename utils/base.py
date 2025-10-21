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
        self.buffer = []
        self.save_history_interval = 1
        self.device = torch.device('cuda')

        self.client_num = args.client_num
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.iid = args.iid
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.loss_type = args.loss_type
        self.in_channels = 3

        if args.dataset == 'cifar10':
            self.num_classes = 10
        if args.dataset == 'cifar100':
            self.num_classes = 100
        ## federated learning args
        self.frac = args.frac
        # 优先使用命令行传入的数据根目录，其次回退到全局配置
        self.data_root = getattr(args, 'data_root', None) or config.globals.data_root
        self.local_ep = args.local_ep

        self.sl_ratio = args.loss_alpha
        self.logdir = f'logs/{self.model_name}_{self.dataset}'