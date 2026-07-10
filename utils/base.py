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

        self.model_name = args.model_name
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.in_channels = args.in_channels
        
        # 优先使用命令行传入的数据根目录，其次回退到全局配置
        self.data_root = getattr(args, 'data_root', None) or config.globals.data_root

        self.logdir = f'logs/{self.model_name}_{self.dataset}'
