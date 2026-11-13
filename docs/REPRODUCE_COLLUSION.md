# 在新机器上复现抗合谋实验

本文档说明如何在一台**全新的计算机**上复现抗合谋（collusion-resistant）
白盒水印实验。以 **AlexNet + CIFAR10 + client_num=10 + K=3** 为例。

---

## 快速开始（三步）

```bash
# 0) 环境就绪后，一键铺好所有配置下的密钥矩阵
python generate_all_key_matrices.py

# 1) 改 utils/args.py 顶部的 EXPERIMENT_CONFIG, 然后:
python main.py
```

`generate_all_key_matrices.py` 会扫描抗合谋配置表自动枚举所需的
`(数据集, 模型, 客户端数)` 组合并全部生成，最后自检是否就绪。
详见 [第 4 节](#4-生成密钥矩阵抗合谋分支)。

---

## 0. 前置条件

| 项 | 要求 |
|---|---|
| OS | Windows / Linux / macOS 均可 |
| Python | 3.9（本项目在 3.9.21 上验证） |
| GPU | 建议 CUDA 可用（CPU 也能跑，但慢很多） |
| 磁盘 | ≥ 5 GB（数据集 + 密钥矩阵 + 模型存档） |

已验证的依赖版本（`conda` 环境）：

```
python      3.9.21
torch       2.5.1
torchvision 0.20.1
numpy       1.26.4
pandas      2.3.2
scikit-learn 1.6.1
tqdm
```

> **注意**：项目默认 `python`（anaconda base）缺少 `mkl_rt.dll`，
> 任何 `@` 矩阵乘法都会直接退出进程。请务必使用带完整 MKL 的环境。

---

## 1. 拉取代码

```bash
git clone https://github.com/Ch1fuyu/Medical_FedLearning_Watermark.git
cd Medical_FedLearning_Watermark
git checkout feature/collusion-attack-resistance
```

## 2. 准备环境

```bash
conda create -n Med_FL_WM python=3.9 -y
conda activate Med_FL_WM

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pandas==2.3.2 scikit-learn==1.6.1 tqdm
```

## 3. 准备数据

CIFAR10 会由 `torchvision` 自动下载到 `./data`：

```bash
python -c "
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', train=True, download=True)
CIFAR10(root='./data', train=False, download=True)
print('CIFAR10 ready')
"
```

> ChestMNIST 需手动下载 `chestmnist.npz` 放到 `./data/`。

## 4. 生成密钥矩阵（抗合谋分支）

**这一步是抗合谋分支的核心，务必执行。**

### 4.1 一键批量生成（推荐）

```bash
python generate_all_key_matrices.py
```

这条命令会扫描 `config/collusion_presets*.json`，把**所有**配置表里
用到的 `(数据集, 模型, 客户端数, design_K)` 组合一次性生成完，
最后做一次完整性自检：

```
抗合谋分支（profile=collusion）共 6 个规格:
  · chestmnist  alexnet  client_num=10   design_K=3
  · chestmnist  resnet   client_num=10   design_K=3
  · cifar10     alexnet  client_num=10   design_K=3
  · cifar10     resnet   client_num=10   design_K=3
  · cifar100    alexnet  client_num=10   design_K=3
  · cifar100    resnet   client_num=10   design_K=3

...
生成完毕: 成功 6 个, 失败 0 个, 耗时 20.9s

=== 完整性自检 ===
  ✓ chestmnist  alexnet  client_num=10   就绪
  ...
  共 6 项, 就绪 6 项, 缺失 0 项

✅ 全部就绪。现在可以直接运行: python main.py
```

常用选项：

```bash
# 同时生成默认分支（传统水印基线）的密钥矩阵
python generate_all_key_matrices.py --include-default

# 只生成指定数据集 / 模型 / 客户端数
python generate_all_key_matrices.py --datasets chestmnist --models resnet

# 覆盖已存在的
python generate_all_key_matrices.py --force

# 只检查不生成（返回码 0 = 全部就绪，2 = 有缺失）
python generate_all_key_matrices.py --check-only

# 在新机器上复现与另一台机器完全一致的水印
python generate_all_key_matrices.py --witness_secret_file my.key
```

> **关于密钥**：脚本默认使用 `save/keys/witness_secret_collusion.key`
> （不存在时自动创建）。要跨机器复现**完全一致**的水印，把那边的
> 密钥文件拷过来用 `--witness_secret_file` 指定。
>
> 运行时**不需要**密钥文件——见证点权威值已经固化在
> `witness_layout.json` 的 `point_values` 字段里。

### 4.2 手动生成单个

```bash
python train_key_matrix.py \
    --model_type alexnet \
    --client_num 10 \
    --profile collusion \
    --collusion_K 3 \
    --dataset cifar10
```

> `--save_dir` **不用手填**，会按 `profile + dataset` 自动推导：
> - `chestmnist` → `./save/key_matrix_collusion`
> - `cifar10` → `./save/key_matrix_collusion_cifar10`
> - `cifar100` → `./save/key_matrix_collusion_cifar100`
>
> `main.py` 侧用同一套规则（`utils/args.py: default_key_matrix_dir`），
> 因此两边目录**必然一致**，不需要手工对齐。

产出：

```
save/key_matrix_collusion_cifar10/alexnet/client10/
├── key_matrix_info.json      # 含 model_signature（模型结构指纹）+ dataset
├── witness_layout.json       # 见证点布局（v=21, G=45）
├── client_0/
│   ├── key_matrix.pth
│   └── positions.json
└── ... (client_1 ~ client_9)
```

### ⚠️ `--dataset` 必须与评测用的数据集一致

卷积层形状随 `in_channels` 变化（ChestMNIST=1，CIFAR10/CIFAR100=3），
因此：

| 评测数据集 | `in_channels` | 生成命令必须用 |
|---|---|---|
| chestmnist | 1 | `--dataset chestmnist` |
| cifar10 | 3 | `--dataset cifar10` |
| cifar100 | 3 | `--dataset cifar100` |

用错会导致密钥矩阵与模型结构不匹配。程序会在启动时**主动检出并退出**：

```
❌ 配置错误：密钥矩阵与本次实验的模型结构不匹配:
   密钥矩阵目录: save/key_matrix_collusion/alexnet/client10
   该目录的卷积层形状是按另一个 in_channels 生成的
   本次实验: alexnet + in_channels=3 (dataset=cifar10)
   形状差异: [('features.0.conv.weight', (64, 1, 5, 5), (64, 3, 5, 5))]
   ...
```

### ⚠️ 不同数据集不能共用同一个密钥矩阵目录

密钥矩阵目录结构是 `<base>/<model>/client<N>`，**不含数据集维度**。
因此 `--save_dir` 会自动按数据集加后缀，避免互相覆盖：

| 数据集 | 自动推导的 `--save_dir` | 配置表 |
|---|---|---|
| chestmnist | `./save/key_matrix_collusion` | `config/collusion_presets.json` |
| cifar10 | `./save/key_matrix_collusion_cifar10` | `config/collusion_presets_cifar10.json` |
| cifar100 | `./save/key_matrix_collusion_cifar100` | `config/collusion_presets_cifar100.json` |

`main.py` 侧的 `key_matrix_dir` / `collusion_size_configs` 也用同一套规则
自动推导（见 `utils/args.py` 的 `default_key_matrix_dir` /
`default_collusion_presets_path`），因此**两边天然对齐**。

若强行用同一个目录，`train_key_matrix.py` 会给出覆盖告警：

```
⚠️  警告：密钥矩阵目录已被另一个数据集占用:
     原有数据集: chestmnist -> 本次: cifar10
```

### 关于密钥

首次运行会自动创建长期密钥 `save/keys/witness_secret_collusion.key`
（32 字节随机数）。**该文件已被 `.gitignore` 排除**，不会进入版本库。

- 同一密钥生成的所有布局互相兼容；
- 换密钥则新旧布局的水印互不兼容（解码必然失败）；
- 如需迁移到新机器，请**单独拷贝**该文件，或用 `--witness_secret_file` 指定。

## 5. 检查抗合谋配置表

`config/collusion_presets.json` 声明了"K → 密钥矩阵规格"的映射：

```json
"3": {
  "resnet":  {"client_num": 10, "design_K": 3},
  "alexnet": {"client_num": 10, "design_K": 3}
}
```

运行 `main.py` 时会按 `(--collusion_size, --model_name)` 自动查表定位目录，
**不需要手工对齐 `--client_num`**。配置缺失或不合法时程序会直接退出并说明。

新增一个 (K, 模型) 组合的步骤：

```bash
# 1) 编辑 config/collusion_presets.json 添加条目
# 2) 生成对应密钥矩阵
python train_key_matrix.py --model_type <model> --client_num <n> \
    --profile collusion --collusion_K <design_K> --dataset <dataset>
# 3) 直接运行 main.py 即可
```

## 6. 运行实验

### 6.1 ChestMNIST（默认目录 + 默认配置表）

```bash
python main.py \
    --collusion_only \
    --dataset chestmnist \
    --model_name alexnet \
    --epochs 3 \
    --local_ep 1 \
    --batch_size 128 \
    --collusion_size 3 \
    --collusion_leak_stage post_training \
    --dp --sigma 0.05
```

### 6.2 CIFAR10（目录与配置表自动推导）

```bash
python main.py \
    --collusion_only \
    --dataset cifar10 \
    --model_name alexnet \
    --epochs 3 \
    --local_ep 1 \
    --batch_size 128 \
    --collusion_size 3 \
    --collusion_leak_stage post_training \
    --dp --sigma 0.05
```

> 更推荐的做法：直接在 `utils/args.py` 的 `EXPERIMENT_CONFIG` 里改，
> 然后 `python main.py` 即可（不用敲任何命令行参数）。
> 见 `docs/REPRODUCE_COLLUSION.md` 第 7 节。

`--collusion_only` 一键完成四件事：

1. 关闭单客户端泄漏仿真；
2. 关闭泄漏源追踪；
3. 开启合谋检测与归因；
4. 切到 `collusion` profile + 开启水印区域冻结。

### 预期输出

```
🔒 抗合谋评测模式：已关闭单客户端泄漏仿真与追踪，开启合谋检测，已自动开启水印区域冻结
🔑 见证点密钥: save\keys\witness_secret_collusion.key (已存在, fingerprint=518ab5a1bd75e65d)
📋 抗合谋配置: K=3, model=alexnet -> client_num=10, dataset=cifar10, design_K=3 (归因搜索上限 5)
✅ 找到密钥矩阵目录: save/key_matrix_collusion_cifar10/alexnet/client10
✅ 水印冻结已启用, 见证点区域 945 个位置
==> 泄漏评测模式: 仅合谋泄漏（单客户端泄漏已关闭）
    抗合谋规格: client_num=10 | model_name=alexnet | design_K=3 | 搜索上限=5
[见证点] 第 1 轮已为 10 个客户端 嵌入见证点水印 (v=21, G=45)
[合谋仿真] 第 1 轮 - 客户端 [2, 4, 9] (K=3, post_training) 合谋发布参数平均值
[合谋检测] 识别出合谋者 = [2, 4, 9] (计数解码, 平均残差=0.0000, margin=0.5185, 有效见证点=21, 规模=3)
[合谋归因] 轮次1 | 实际[2, 4, 9] | 检出[2, 4, 9] (估计规模=3) ✓
...
合谋归因准确率: 100.00% (3/3)
```

## 7. 实验配置中心（推荐用法）

`utils/args.py` 顶部有一个 **实验配置中心**，改完直接 `python main.py`
即可，不用敲任何命令行参数：

```python
EXPERIMENT_CONFIG: Dict[str, Any] = {
    # ---------------- 数据集与模型 ----------------
    'dataset': 'cifar10',          # chestmnist / cifar10 / cifar100
    'model_name': 'alexnet',       # alexnet / resnet

    # ---------------- 联邦学习 ----------------
    'epochs': 3,                   # 通信轮数 R
    'local_ep': 1,                 # 每客户端本地轮数 E
    'batch_size': 128,             # 本地批大小 B
    'client_num': 10,              # 客户端数

    # ---------------- 抗合谋评测（见证点）----------------
    'collusion_only': True,        # 一键切到抗合谋评测模式
    'collusion_size': 3,           # 仿真的合谋规模 K
    'collusion_leak_stage': 'post_training',
    # None = 按 profile + dataset 自动推导
    'collusion_size_configs': None,
    'key_matrix_dir': None,
    'key_matrix_profile': 'collusion',
    'enable_witness': True,

    # ---------------- 差分隐私 ----------------
    'dp': True,
    'sigma': 0.05,
    ...
}
```

**优先级**：命令行参数 > 配置中心 > argparse 默认值。

也就是说，命令行仍然可以临时覆盖任何一项：

```bash
# 配置中心用的是 epochs=3，这里临时改成 5
python main.py --epochs 5
```

**生效范围**：只有 `main.py` 会应用配置中心。攻击脚本（`pruning_attack.py` /
`finetune_attack.py`）各自有独立的命令行参数，不受影响。

### 换数据集只需改一个字段

把配置中心的 `'dataset'` 改成 `'chestmnist'`，密钥矩阵目录与配置表
都会自动切到对应的一套：

```
chestmnist -> save/key_matrix_collusion            + config/collusion_presets.json
cifar10    -> save/key_matrix_collusion_cifar10    + config/collusion_presets_cifar10.json
cifar100   -> save/key_matrix_collusion_cifar100   + config/collusion_presets_cifar100.json
```

（规则见 `utils/args.py: default_key_matrix_dir` /
`default_collusion_presets_path`，`train_key_matrix.py` 复用同一套逻辑。）

## 8. 输出产物

| 文件 | 内容 |
|---|---|
| `save/collusion/collusion_attribution_{model}_{dataset}_{ts}.json` | 逐轮归因结果、准确率、按 K 分组统计 |
| `save/excel/metrics_{model}_{dataset}_{ts}.xlsx` | 含 `collusion_actual_clients` / `collusion_detected_clients` / `collusion_estimated_size` / `collusion_attribution_correct` 列 |
| `save/{model}/{dataset}/*_enhanced.pkl` | 训练日志 + 最优模型权重 |
| `logs/console.logs` | 完整运行日志 |

## 9. 常见问题

### Q: 提示 `抗合谋配置表中没有 (合谋规模 K=..., 模型=...) 的条目`

在 `config/collusion_presets.json` 添加对应条目，或改用已配置的 (K, 模型)。
错误信息会列出所有可用组合。

### Q: 提示 `见证点布局的 design.K=3 与配置表声明的 design_K=7 不一致`

配置表与磁盘上的布局不同步。二选一：
- 用布局的实际 `design.K` 更新配置表；
- 或重新生成密钥矩阵（`--collusion_K 7`）。

### Q: 归因输出 `检出=[]`

搜索上界不足。程序会自动按实际合谋规模放宽并打日志：

```
[合谋检测] 布局 design.K=3 推出的搜索上界 5 < 合谋规模 K=6，已自动放宽到 6。
```

若仍失败，检查见证点区域是否被正确冻结（`--collusion_only` 会自动开启）。

### Q: 想跑"泄漏追踪"实验（带噪声攻击）

那是另一条实验线，与抗合谋互斥，**不要**加 `--collusion_only`：

```bash
python main.py --dataset cifar10 --model_name alexnet \
    --enable_leakage_tracking --leak_attack_mode gaussian_noise ...
```

## 10. 附：目录结构速查

```
Medical_FedLearning_Watermark/
├── config/
│   ├── globals.py
│   ├── collusion_presets.json           # 抗合谋配置表 (chestmnist)
│   ├── collusion_presets_cifar10.json   # 抗合谋配置表 (cifar10)
│   └── collusion_presets_cifar100.json  # 抗合谋配置表 (cifar100)
├── data/                            # 数据集（自动下载）
├── docs/REPRODUCE_COLLUSION.md       # 本文档
├── logs/console.logs                # 运行日志
├── save/
│   ├── keys/witness_secret_collusion.key   # 长期密钥（gitignore）
│   ├── key_matrix/                   # 默认分支（传统水印基线）
│   ├── key_matrix_collusion/         # 抗合谋分支 (chestmnist)
│   │   └── <model>/client<N>/
│   │       ├── key_matrix_info.json
│   │       ├── witness_layout.json
│   │       └── client_<i>/{key_matrix.pth, positions.json}
│   ├── key_matrix_collusion_cifar10/ # 抗合谋分支 (cifar10, in_ch=3)
│   ├── key_matrix_collusion_cifar100/# 抗合谋分支 (cifar100, in_ch=3)
│   ├── collusion/                    # 抗合谋归因结果 JSON
│   └── excel/                        # 指标 Excel
├── main.py                           # 主实验入口
├── train_key_matrix.py               # 密钥矩阵 + 见证点布局生成（单个）
├── generate_all_key_matrices.py      # 批量生成（按配置表自动枚举）★ 新机器首选
└── utils/
    ├── args.py                       # 参数中心 + 实验配置中心
    ├── collusion_presets.py          # 配置表解析与校验
    ├── collusion_resistant.py        # 见证点设计 / 嵌入 / 解码
    ├── witness_secret.py             # 密钥持久化
    └── trainer_private_enhanced.py   # 训练器（训练后硬覆盖水印 + 梯度缩放）
```
