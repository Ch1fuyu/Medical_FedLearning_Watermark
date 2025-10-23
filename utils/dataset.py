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


class LocalCIFAR10Dataset(Dataset):
    """本地CIFAR-10数据集类，用于联邦学习"""
    def __init__(self, data_root, split='train', transform=None):
        self.transform = transform
        self.split = split
        
        # 使用torchvision加载CIFAR-10数据集
        if split == 'train':
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=None
            )
        elif split == 'test':
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=None
            )
        else:
            raise ValueError(f"Unsupported split: {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LocalCIFAR100Dataset(Dataset):
    """本地CIFAR-100数据集类，用于联邦学习"""
    def __init__(self, data_root, split='train', transform=None):
        self.transform = transform
        self.split = split
        
        # 使用torchvision加载CIFAR-100数据集
        if split == 'train':
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_root, train=True, download=True, transform=None
            )
        elif split == 'test':
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_root, train=False, download=True, transform=None
            )
        else:
            raise ValueError(f"Unsupported split: {split}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)
        
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

    if dataset_name == 'chestmnist':
        dataset_path = os.path.join(data_root, dataset_name + '.npz')
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
        
        if iid:
            dict_users = chestmnist_iid(train_set, client_num)
        else:
            dict_users = chestmnist_beta(train_set, 0.1, client_num)
            
    elif dataset_name == 'cifar10':
        # CIFAR-10参数
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        num_classes = 10
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalCIFAR10Dataset(data_root, split='train', transform=transform_train)
        test_set = LocalCIFAR10Dataset(data_root, split='test', transform=transform_test)
        
        print(f"CIFAR-10 dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
        
        if iid:
            dict_users = cifar10_iid(train_set, client_num)
        else:
            dict_users = cifar10_beta(train_set, 0.1, client_num)
    elif dataset_name == 'cifar100':
        # CIFAR-100参数
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        num_classes = 100
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalCIFAR100Dataset(data_root, split='train', transform=transform_train)
        test_set = LocalCIFAR100Dataset(data_root, split='test', transform=transform_test)
        
        print(f"CIFAR-100 dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
        
        if iid:
            dict_users = cifar100_iid(train_set, client_num)
        else:
            dict_users = cifar100_beta(train_set, 0.1, client_num)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, dict_users

def get_data_no_fl(dataset_name, data_root, dataset_file=None):
    train_set = []
    test_set = []

    if dataset_name == 'chestmnist':
        dataset_path = os.path.join(data_root, dataset_name + '.npz')
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
        
    elif dataset_name == 'cifar10':
        print(f"Using CIFAR-10 dataset")
        
        # CIFAR-10参数
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        num_classes = 10
        print("Using CIFAR-10 multi-class classification setup (10 classes)")
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalCIFAR10Dataset(data_root, split='train', transform=transform_train)
        test_set = LocalCIFAR10Dataset(data_root, split='test', transform=transform_test)
        
        print(f"CIFAR-10 dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
    elif dataset_name == 'cifar100':
        print(f"Using CIFAR-100 dataset")
        
        # CIFAR-100参数
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        num_classes = 100
        print("Using CIFAR-100 multi-class classification setup (100 classes)")
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LocalCIFAR100Dataset(data_root, split='train', transform=transform_train)
        test_set = LocalCIFAR100Dataset(data_root, split='test', transform=transform_test)
        
        print(f"CIFAR-100 dataset - Training set size: {len(train_set)}, Test set size: {len(test_set)}")
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

    # 只获取卷积层参数索引
    conv_param_indices = []
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:  # 只处理卷积层权重
            conv_param_indices.extend([(name, i) for i in range(param.numel())])

    print(f"Convolutional layer parameter count: {len(conv_param_indices)}")

    if encoder_total_params > len(conv_param_indices):
        raise ValueError(f"编码器参数数量 ({encoder_total_params}) 超过卷积层参数数量 ({len(conv_param_indices)})，无法分配位置。")

    # 打乱卷积层参数索引
    np.random.shuffle(conv_param_indices)

    # 只取编码器大小数量的索引
    selected_param_indices = conv_param_indices[:encoder_total_params]

    # 平均划分给所有客户端
    chunk_size = encoder_total_params // client_num

    position_dict = {}
    for client_id in range(client_num):
        start = client_id * chunk_size
        # 最后一个客户端拿剩余所有
        end = (client_id + 1) * chunk_size if client_id != client_num - 1 else encoder_total_params
        position_dict[client_id] = selected_param_indices[start:end]

    return position_dict