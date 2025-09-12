# Medical Federated Learning with Watermarking

基于联邦学习的医学图像分类项目，支持水印技术保护模型知识产权。

## 项目概述

本项目实现了在医学图像数据集（ChestMNIST）上的联邦学习训练，并集成了基于密钥矩阵的水印技术，用于保护模型的知识产权。

## 主要特性

- **联邦学习**：支持多客户端分布式训练
- **水印技术**：基于密钥矩阵的模型水印嵌入
- **医学图像分类**：支持ChestMNIST多标签分类
- **模型支持**：ResNet18、AlexNet等主流架构
- **差分隐私**：可选的隐私保护机制

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

### 2. 生成自编码器水印

```bash
python train_autoencoder.py
```

### 3. 生成密钥矩阵

```bash
python train_key_matrix.py --model_type resnet --client_num 10 --dataset chestmnist
```

### 4. 运行联邦学习训练

```bash
python main.py --model resnet --dataset chestmnist --client_num 10 --epochs 100
```

## 项目结构

```
├── main.py                 # 主程序入口
├── train_autoencoder.py   # 自编码器训练
├── train_key_matrix.py    # 密钥矩阵生成
├── models/                # 模型定义
│   ├── resnet.py         # ResNet模型
│   ├── alexnet.py        # AlexNet模型
│   └── light_autoencoder.py  # 轻量自编码器
├── utils/                 # 工具模块
│   ├── dataset.py        # 数据处理
│   ├── trainer_private.py # 训练器
│   └── key_matrix_utils.py # 密钥矩阵工具
├── config/               # 配置文件
└── save/                # 保存目录
    ├── autoencoder/     # 自编码器权重
    └── key_matrix/      # 密钥矩阵
```

## 核心功能

### 水印系统

- **密钥矩阵生成**：为每个客户端生成唯一的水印位置
- **水印嵌入**：在训练过程中将水印嵌入模型参数
- **水印验证**：支持水印的提取和验证

### 联邦学习

- **客户端训练**：支持本地模型训练
- **模型聚合**：FedAvg算法聚合客户端模型
- **隐私保护**：可选的差分隐私机制

## 参数配置

主要参数可通过命令行或配置文件设置：

- `--model`: 模型类型 (resnet/alexnet)
- `--dataset`: 数据集 (chestmnist)
- `--client_num`: 客户端数量
- `--epochs`: 训练轮数
- `--dp`: 是否启用差分隐私
- `--sigma`: 噪声强度

## 数据集

项目使用ChestMNIST数据集：
- **类型**：胸部X光片多标签分类
- **类别数**：14个疾病类别
- **图像尺寸**：28×28像素
- **样本数**：训练集78,468，测试集11,219

## 注意事项

- 确保有足够的GPU内存进行训练
- 数据文件需要放在`data/`目录下
- 训练结果会保存在`save/`目录下

## 许可证

本项目仅供学术研究使用。