# Medical Federated Learning with Watermarking

基于联邦学习的医学图像分类项目，集成水印技术保护模型知识产权，支持多种攻击实验。

## 🚀 快速开始

### 环境准备
```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm pandas
```

### 1. 训练自编码器
```bash
python train_autoencoder.py
```

### 2. 生成密钥矩阵
```bash
python train_key_matrix.py --model_type resnet --client_num 10 --dataset chestmnist
```

### 3. 联邦学习训练
```bash
python main.py --model_name resnet --dataset chestmnist --client_num 10 --epochs 100
```

### 4. 攻击实验
```bash
# 微调攻击
python finetune_attack.py --model_path ./save/resnet/chestmnist/model.pkl

# 剪枝攻击  
python pruning_attack.py --model_path ./save/resnet/chestmnist/model.pkl
```

## 📊 支持的数据集

| 数据集 | 类型 | 类别数 | 任务 |
|--------|------|--------|------|
| ChestMNIST | 医学图像 | 14 | 多标签分类 |
| CIFAR-10 | 自然图像 | 10 | 多分类 |
| CIFAR-100 | 自然图像 | 100 | 多分类 |

## 🏗️ 项目结构

```
├── main.py                 # 联邦学习主程序
├── finetune_attack.py     # 微调攻击实验
├── pruning_attack.py      # 剪枝攻击实验
├── train_autoencoder.py   # 自编码器训练
├── train_key_matrix.py    # 密钥矩阵生成
├── models/                # 模型定义
│   ├── resnet.py         # ResNet18
│   ├── alexnet.py        # AlexNet
│   └── light_autoencoder.py
├── utils/                 # 工具模块
│   ├── dataset.py        # 数据处理
│   ├── trainer_private.py # 训练器
│   ├── watermark_reconstruction.py # 水印重建
│   └── delta_pcc_utils.py # ΔPCC评估
└── save/                 # 结果保存
    ├── autoencoder/      # 自编码器权重
    ├── key_matrix/       # 密钥矩阵
    └── excel/           # 实验结果
```

## 🔧 核心功能

### 水印系统
- **密钥矩阵生成**：为每个客户端生成唯一水印位置
- **水印嵌入**：训练过程中嵌入水印到模型参数
- **水印验证**：支持水印提取和完整性验证
- **ΔPCC评估**：量化水印完整性变化

### 联邦学习
- **多客户端训练**：支持IID/非IID数据分布
- **模型聚合**：FedAvg算法
- **隐私保护**：可选差分隐私机制
- **水印缩放**：自适应水印参数调整

### 攻击实验
- **微调攻击**：模拟模型微调对水印的影响
- **剪枝攻击**：测试模型剪枝对水印的破坏性
- **完整性评估**：量化攻击对水印的破坏程度

## ⚙️ 主要参数

```bash
# 基础配置
--model_name resnet              # 模型类型 (resnet/alexnet)
--dataset chestmnist        # 数据集
--client_num 10            # 客户端数量
--epochs 100               # 训练轮数

# 水印配置
--watermark_mode enhanced   # 水印模式
--use_key_matrix           # 使用密钥矩阵
--enable_watermark_scaling # 启用水印缩放

# 隐私保护
--dp                       # 启用差分隐私
--sigma 0.1                # 噪声强度
```

## 📈 实验结果

项目支持多种评估指标：
- **准确率**：模型分类性能
- **AUC**：多标签分类性能
- **水印完整性**：水印提取成功率
- **ΔPCC值**：水印参数相关性变化

## 📝 注意事项

- 确保GPU内存充足（建议8GB+）
- 数据文件需放在`data/`目录
- 训练结果保存在`save/`目录
- 支持Windows多进程兼容

## 📄 许可证

本项目仅供学术研究使用。