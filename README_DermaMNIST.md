# DermaMNIST联邦学习集成

本项目已成功集成MedMNIST数据集中的DermaMNIST，用于皮肤病图像分类的联邦学习实验。

## 数据集介绍

**DermaMNIST** 是一个皮肤病图像分类数据集：
- **图像尺寸**: 28×28像素（RGB彩色图像）
- **类别数**: 7个皮肤病类别
- **训练集**: 约10,015张图像
- **测试集**: 约1,505张图像
- **任务类型**: 多分类

## 安装依赖

### 方法1：使用安装脚本
```bash
python install_dependencies.py
```

### 方法2：手动安装
```bash
pip install medmnist torch torchvision numpy matplotlib tqdm
```

## 测试集成

运行测试脚本验证DermaMNIST集成是否正常：

```bash
python test_dermamnist.py
```

测试内容包括：
- 数据加载测试
- 数据采样测试
- 模型兼容性测试
- 数据可视化测试

## 运行实验

### 联邦学习实验
```bash
python run_dermamnist.py --mode fl
```

### 使用命令行参数
```bash
python main.py --dataset dermamnist --num_classes 7 --client_num 5 --batch_size 8 --epochs 50
```

## 主要修改

### 1. 数据加载模块 (`utils/dataset.py`)
- 添加了MedMNIST支持
- 实现了DermaMNIST数据预处理
- 支持IID和Non-IID数据分割

### 2. 参数配置 (`utils/args.py`)
- 添加了`dermamnist`数据集选项
- 更新了类别数参数说明

### 3. 主程序 (`main.py`)
- 自动检测数据集类型并调整参数
- 支持DermaMNIST的模型配置

### 4. 数据采样 (`utils/sampling.py`)
- 添加了DermaMNIST专用的IID和Non-IID采样函数

## 实验配置

### 推荐参数设置
```python
# 数据集参数
dataset = 'dermamnist'
num_classes = 7
in_channels = 3

# 联邦学习参数
client_num = 5      # 客户端数量
batch_size = 8      # 批次大小
local_ep = 3        # 本地训练轮数
epochs = 50         # 通信轮数
lr = 0.01          # 学习率

# 水印参数
num_bit = 20        # 水印位数
weight_type = 'gamma'  # 水印类型
loss_alpha = 0.2    # 损失权重
```

## 数据预处理

DermaMNIST使用以下预处理策略：
- **归一化**: ImageNet标准参数
- **数据增强**: 水平翻转、旋转、颜色抖动
- **图像尺寸**: 28×28×3 (RGB)

## 模型架构

使用修改后的AlexNet：
- **输入层**: 3通道 (RGB)
- **输出层**: 7类别 (皮肤病分类)
- **适配**: 28×28像素输入

## 实验结果

实验将保存在 `./save/alexnet/dermamnist/` 目录下，文件名包含实验参数和最终准确率。

## 注意事项

1. **数据量较小**: DermaMNIST数据量相对较小，建议减少客户端数量
2. **医学数据敏感性**: 注意医学数据的隐私保护
3. **模型性能**: 可能需要调整网络结构以获得更好的性能
4. **水印强度**: 医学图像对水印嵌入更敏感，需要谨慎调整参数

## 故障排除

### 常见问题

1. **MedMNIST导入失败**
   ```bash
   pip install medmnist
   ```

2. **内存不足**
   - 减少batch_size
   - 减少client_num

3. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理

## 扩展其他MedMNIST数据集

可以轻松扩展到其他MedMNIST数据集：

1. 修改`dataset_name`参数
2. 调整`num_classes`和`in_channels`
3. 更新数据预处理参数
4. 添加相应的采样函数

支持的数据集：
- PathMNIST (病理图像)
- ChestMNIST (胸部X光)
- OCTMNIST (眼科OCT)
- PneumoniaMNIST (肺炎X光)
- RetinaMNIST (视网膜图像)
- 等等...

## 联系方式

如有问题，请检查：
1. 依赖是否正确安装
2. 数据集是否正确下载
3. 参数设置是否合理 