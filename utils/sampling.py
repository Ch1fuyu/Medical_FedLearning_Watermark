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
    """Non-IID数据分割 for ChestMNIST"""
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
    for y in range(len(dataset)):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    for y in range(len(dataset)):
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

    client_datasets = []
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset, client_i_idx)
        client_datasets.append(subset)

    return client_datasets