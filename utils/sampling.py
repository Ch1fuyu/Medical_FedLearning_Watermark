import numpy as np
from torch.utils.data import Subset

def cifar_iid(dataset, client_num):
    sample_num_per_client = int(len(dataset)/client_num)
    dict_per_client, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(client_num):
        dict_per_client[i] = set(np.random.choice(all_idxs, sample_num_per_client, replace=False))
        all_idxs = list(set(all_idxs) - dict_per_client[i])
    return dict_per_client

def cifar_beta(dataset, beta, n_clients):
    # beta = 0.1, n_clients = 10
    label_distributions = []
    for y in range(len(dataset)):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))


    labels = np.array(dataset.targets).astype(int)
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