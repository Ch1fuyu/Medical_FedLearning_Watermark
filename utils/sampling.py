import numpy as np
from torch.utils.data import Subset

# ChestMNIST数据分割函数
def chestmnist_iid(dataset, client_num):
    """IID数据分割 for ChestMNIST"""
    sample_num_per_client = int(len(dataset)/client_num)
    dict_per_client, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(client_num):
        dict_per_client[i] = set(np.random.choice(all_idxs, sample_num_per_client, replace=False))
        all_idxs = list(set(all_idxs) - dict_per_client[i])
    return dict_per_client

def chestmnist_beta(dataset, beta, n_clients):
    """Non-IID数据分割 for ChestMNIST - 多标签分类处理"""
    # ChestMNIST是多标签分类，有14个病理标签
    num_classes = 14
    
    # 获取每个样本的标签向量
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        # 确保标签是numpy数组格式
        if hasattr(label, 'numpy'):
            label = label.numpy()
        elif isinstance(label, np.ndarray):
            pass
        else:
            label = np.array(label)
        
        # 确保标签是1D数组，形状为(14,)
        if len(label.shape) > 1:
            label = label.squeeze()
        
        labels.append(label)
    labels = np.array(labels)  # 形状为 (样本数, 14)
    
    # 计算每个类别的分布
    label_distributions = []
    for y in range(num_classes):  # 修复：应该是14个类别，不是数据集大小
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    # 为每个类别分配样本
    for y in range(num_classes):  # 修复：应该是14个类别
        # 找到包含第y个标签的样本
        label_y_idx = np.where(labels[:, y] == 1)[0]  # 多标签：标签为1表示包含该类别
        label_y_size = len(label_y_idx)

        if label_y_size == 0:
            # 如果该类别没有样本，跳过
            for i in range(n_clients):
                client_size_map[i][y] = 0
                client_idx_map[i][y] = []
            continue

        sample_size = (label_distributions[y] * label_y_size).astype(int)
        sample_size[n_clients - 1] += len(label_y_idx) - np.sum(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]

    # 为每个客户端合并所有类别的样本，返回索引字典（与chestmnist_iid保持一致）
    dict_per_client = {}
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        # 去重，因为一个样本可能属于多个类别
        client_i_idx = np.unique(client_i_idx)
        np.random.shuffle(client_i_idx)
        dict_per_client[i] = set(client_i_idx)

    return dict_per_client


# CIFAR-10数据分割函数
def cifar10_iid(dataset, client_num):
    """IID数据分割 for CIFAR-10"""
    sample_num_per_client = int(len(dataset)/client_num)
    dict_per_client, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(client_num):
        dict_per_client[i] = set(np.random.choice(all_idxs, sample_num_per_client, replace=False))
        all_idxs = list(set(all_idxs) - dict_per_client[i])
    return dict_per_client

def cifar10_beta(dataset, beta, n_clients):
    """Non-IID数据分割 for CIFAR-10"""
    # 获取标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        # 保证label为int类型
        if hasattr(label, 'item'):
            labels.append(int(label.item()))
        elif isinstance(label, np.ndarray):
            labels.append(int(label.squeeze()))
        else:
            labels.append(int(label))
    labels = np.array(labels).astype(int)
    
    # 计算每个类别的分布
    label_distributions = []
    for y in range(10):  # CIFAR-10有10个类别
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    for y in range(10):  # CIFAR-10有10个类别
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)

        sample_size = (label_distributions[y] * label_y_size).astype(int)
        sample_size[n_clients - 1] += len(label_y_idx) - np.sum(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]

    # 返回索引字典
    dict_per_client = {}
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        dict_per_client[i] = set(client_i_idx)

    return dict_per_client


# CIFAR-100数据分割函数
def cifar100_iid(dataset, client_num):
    """IID数据分割 for CIFAR-100"""
    sample_num_per_client = int(len(dataset)/client_num)
    dict_per_client, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(client_num):
        dict_per_client[i] = set(np.random.choice(all_idxs, sample_num_per_client, replace=False))
        all_idxs = list(set(all_idxs) - dict_per_client[i])
    return dict_per_client

def cifar100_beta(dataset, beta, n_clients):
    """Non-IID数据分割 for CIFAR-100"""
    # 获取标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        # 保证label为int类型
        if hasattr(label, 'item'):
            labels.append(int(label.item()))
        elif isinstance(label, np.ndarray):
            labels.append(int(label.squeeze()))
        else:
            labels.append(int(label))
    labels = np.array(labels).astype(int)
    
    # 计算每个类别的分布
    label_distributions = []
    for y in range(100):  # CIFAR-100有100个类别
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    for y in range(100):  # CIFAR-100有100个类别
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)

        sample_size = (label_distributions[y] * label_y_size).astype(int)
        sample_size[n_clients - 1] += len(label_y_idx) - np.sum(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]

    # 返回索引字典
    dict_per_client = {}
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        dict_per_client[i] = set(client_i_idx)

    return dict_per_client