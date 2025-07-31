import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config.globals import set_seed
from models.light_autoencoder import LightAutoencoder
from utils.sampling import *

set_seed()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_data(dataset_name, data_root, iid, client_num):
    train_set = []
    test_set = []
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        # 检查本地是否已经下载好数据
        data_exists = os.path.exists(os.path.join(data_root, 'cifar-10-batches-py'))

        train_set = torchvision.datasets.CIFAR10(data_root,
                                                 train=True,
                                                 download=not data_exists,
                                                 transform=transform_train
                                                 )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=not data_exists,
                                                transform=transform_test
                                                )

    if iid:
        dict_users = cifar_iid(train_set, client_num)
    else:
        dict_users = cifar_beta(train_set, 0.1, client_num)

    return train_set, test_set, dict_users

def get_data_no_fl(dataset_name, data_root):
    train_set = []
    test_set = []
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        # 检查本地是否已经下载好数据
        data_exists = os.path.exists(os.path.join(data_root, 'cifar-10-batches-py'))

        train_set = torchvision.datasets.CIFAR10(data_root,
                                                 train=True,
                                                 download=not data_exists,
                                                 transform=transform_train
                                                 )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=not data_exists,
                                                transform=transform_test
                                                )

    return train_set, test_set

def construct_random_wm_position(model, client_num):
    # 编码器参数数量（作为水印大小）
    encoder = LightAutoencoder().encoder
    encoder_total_params = sum(param.numel() for param in encoder.parameters())
    if encoder_total_params == 0:
        raise ValueError("编码器参数数量为 0，请检查编码器模型是否正确。")

    print(f"编码器总参数数量（作为水印的大小）: {encoder_total_params}")

    # 获取主任务模型所有参数扁平索引
    param_indices = []
    for name, param in model.named_parameters():
        param_indices.extend([(name, i) for i in range(param.numel())])

    if encoder_total_params > len(param_indices):
        raise ValueError(f"编码器参数数量 ({encoder_total_params}) 超过主任务模型总参数数量 ({len(param_indices)})，无法分配位置。")

    # 打乱所有参数索引
    np.random.shuffle(param_indices)

    # 只取编码器大小数量的索引
    selected_param_indices = param_indices[:encoder_total_params]

    # 平均划分给所有客户端
    chunk_size = encoder_total_params // client_num

    position_dict = {}
    for client_id in range(client_num):
        start = client_id * chunk_size
        # 最后一个客户端拿剩余所有
        end = (client_id + 1) * chunk_size if client_id != client_num - 1 else encoder_total_params
        position_dict[client_id] = selected_param_indices[start:end]

    return position_dict