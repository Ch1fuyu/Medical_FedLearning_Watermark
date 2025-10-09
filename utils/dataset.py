import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 本地数据集类
class LocalChestMNISTDataset(Dataset):
    """本地ChestMNIST数据集类，用于加载.npz文件"""
    def __init__(self, data_path, split='train', transform=None):
        self.transform = transform
        self.split = split
        
        # 加载.npz文件
        data = np.load(data_path)
        
        if split == 'train':
            self.images = data['train_images']
            self.labels = data['train_labels']
        elif split == 'test':
            self.images = data['test_images']
            self.labels = data['test_labels']
        else:
            raise ValueError(f"Unsupported split: {split}")
        
        # 保持标签的2D格式 (样本数, 标签数)
        # 确保标签是1D数组
        if len(self.labels.shape) > 1:
            if self.labels.shape[1] > 1:
                # 保持2D格式
                pass
            else:
                # squeeze为1D
                self.labels = self.labels.squeeze()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 处理图像数据 - 根据错误信息修复
        if len(image.shape) == 3:
            # 检查图像格式
            if image.shape[0] == 1:  # 单通道，形状为 (1, H, W)
                image = image.squeeze(0)  # 转换为 (H, W)
            elif image.shape[0] == 3:  # RGB，形状为 (3, H, W)
                image = image.transpose(1, 2, 0)  # 转换为 (H, W, 3)
            elif image.shape[2] == 1:  # 单通道，形状为 (H, W, 1)
                image = image.squeeze(2)  # 转换为 (H, W)
        
        # 确保数据类型正确
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 转换为PIL图像
        from PIL import Image
        if len(image.shape) == 2:
            # 灰度图像，转换为RGB
            image = Image.fromarray(image, mode='L').convert('RGB')
        else:
            # RGB图像
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # 确保标签保持正确的形状
        # 对于ChestMNIST分类，标签应该是 (14,) 形状
        if len(label.shape) == 2 and label.shape[1] == 14:
            # 如果是 (1, 14) 形状，squeeze为 (14,)
            label = label.squeeze()
        
        return image, label

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

    dataset_path = os.path.join(data_root, dataset_name + '.npz')
    if dataset_name == 'chestmnist':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"ChestMNIST dataset file not found: {dataset_path}")
        
        # ChestMNIST参数
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        num_classes = 14  # 14个病理标签
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalChestMNISTDataset(dataset_path, split='train', transform=transform_train)
        test_set = LocalChestMNISTDataset(dataset_path, split='test', transform=transform_test)
        
        print(f"ChestMNIST dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if iid:
        dict_users = chestmnist_iid(train_set, client_num)
    else:
        dict_users = chestmnist_beta(train_set, 0.1, client_num)

    return train_set, test_set, dict_users

def get_data_no_fl(dataset_name, data_root, dataset_file=None):
    train_set = []
    test_set = []

    dataset_path = os.path.join(data_root, dataset_name + '.npz')
    if dataset_name == 'chestmnist':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"ChestMNIST dataset file not found: {dataset_path}")
        
        print(f"Using ChestMNIST dataset file: {dataset_name}")
        
        # ChestMNIST参数
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        num_classes = 14  # 14个病理标签
        print("Using ChestMNIST multi-label classification setup (14 pathology labels)")
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalChestMNISTDataset(dataset_path, split='train', transform=transform_train)
        test_set = LocalChestMNISTDataset(dataset_path, split='test', transform=transform_test)
        
        print(f"ChestMNIST dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set

def construct_random_wm_position(model, client_num):
    # 编码器参数数量（作为水印大小）
    encoder = LightAutoencoder().encoder
    encoder_total_params = sum(param.numel() for param in encoder.parameters())
    if encoder_total_params == 0:
        raise ValueError("编码器参数数量为 0，请检查编码器模型是否正确。")

    print(f"Encoder total parameter count (as watermark size): {encoder_total_params}")

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