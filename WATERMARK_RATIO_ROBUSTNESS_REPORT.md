# watermark_grad_target_ratio 对水印鲁棒性的影响 —— 排查报告

> **排查目标**: 定位 `utils/args.py` 中 `--watermark_grad_target_ratio` 对水印鲁棒性（追踪/检测准确率）的影响机制
> **分支**: `feature/watermark-grad-ratio-robustness`
> **环境**: `Med_FL_WM` (torch 2.5.1 + CUDA)

---

## 0. 结论速览

排查过程中发现 `watermark_grad_target_ratio` 的**保护机制本身存在两个严重 bug**，
使得该参数在修复前**几乎完全不起作用**：

| # | Bug | 位置 | 后果 |
|---|-----|------|------|
| 1 | 水印区**负梯度被翻转为正梯度** | `models/losses/multi_loss.py` | 方向反转，等价于在主任务梯度上叠加随机噪声，水印信号被破坏 |
| 2 | `scale_factor` 下限 `clamp(0.1)` | `models/losses/multi_loss.py` | 实际达成比例停在 `0.1 × (wm/non_wm)`，**永远压不到目标值** |

> **关键推论**: 在此之前所有关于「ratio 取多少合适」的结论都建立在**未真正生效**的
> capping 之上，需要在修复后重新评估。

### 修复后的实测结论（chestmnist / alexnet / K=5 / 30 轮 × 6 个 ratio）

| 结论 | 说明 |
|------|------|
| capping **精确生效** | 触发 capping 时的实际比例与目标值吻合到 0.001 以内（0.100/0.200/0.299/0.399/0.499） |
| capping **确有必要** | 未缩放时水印区梯度约为非水印区的 1.0 倍，30 轮中触发 25 次 |
| 对精度 **影响极小** | 最佳精度 94.7880% ~ 94.7940%（极差 0.006 个百分点） |
| 检测指标 **已饱和** | 所有 ratio 下检测准确率均为 100%，**本配置无法区分 ratio 优劣** |

> 因此：**不能**据此调整 `--watermark_grad_target_ratio` 的默认值。
> 详见第 6.1 / 6.2 节。

### 附带发现（独立回归，与 ratio 无关）

扫描过程中还发现**今天刚合入**的 `4877a2c`（服务器端水印权威覆盖）
导致「训练前泄漏追踪」退化到随机水平（3.33%，K=5），
历史（该提交之前）为 100%。详见第 7 节。
该退化与本次 ratio 修复**无关**（`ratio=0` 关闭 capping 时依然存在）。

> **已于 2026-10-26 修复**：在分支 `fix/pre-training-tracing-regression` 中把
> `_watermark_aggregation` 回退为 no-op（聚合阶段保持纯 FedAvg），
> 并验证「训练前追踪」与「训练后检测」**两个指标同时恢复/保持 100%**。
> 详见第 10 节。

### 相关分支的实现对比

排查过程中顺带比对了仓库中其他分支的 capping 实现，
可以确认当前分支的实现是**回归（regression）**而非历史遗留：

| 分支 | 符号处理 | `scale_factor` 下限 | 说明 |
|------|---------|-------------------|------|
| `grad_redu_exact_detect`（本分支基线） | ❌ 翻转 | ❌ `0.1` | 两个 bug 都在 |
| `grad_redu` | ❌ 翻转 | ❌ `0.1` | 与基线一致 |
| `only_margin` | ✅ 正确（`-torch.sign(...)`） | ❌ `0.1` | 符号处理是对的，回归发生在之后的 `grad_redu` |
| `grad_2_zero` / `with_3_regs` / `pure3regs` | — | — | 无 capping 代码，走了别的实现路线 |

`only_margin` 分支中的写法证明**正确的符号处理方式在本仓库中曾被实现过**，
只是没有被正确地迁移到当前分支。这也印证了 Bug 1 是无意的实现错误而非有意设计。

---

## 1. 参数链路

`--watermark_grad_target_ratio`（`utils/args.py:52`，默认 `0.3`）的完整生效链路：

```
utils/args.py:52            解析命令行参数，默认 0.3
        ↓
utils/trainer_private_enhanced.py:173
                           MultiLoss(model=..., target_ratio=args.watermark_grad_target_ratio)
        ↓
models/losses/multi_loss.py:198-220
                           若 wm_ratio > target_ratio，则对水印区梯度做缩放
                           scale_factor = (target_wm_grad / current_wm_grad_val).clamp(0.1, 1.0)
```

作用机理：把**水印区域梯度均值**压到 **非水印区域梯度均值 × target_ratio**，
从而在参数更新中让水印信号不被主任务梯度"淹没"。

---

## 2. Bug 1：水印区负梯度被翻转

### 2.1 问题代码

```python
wm_sign = torch.sign(gradients * watermark_mask).clamp_min(0.0)        # 正部: 1
wm_negative_sign = torch.sign(gradients * watermark_mask).clamp_max(0.0)  # 负部: -1

scaled_wm_grad = torch.abs(gradients * watermark_mask) * scale_factor

corrected_gradients = (
    gradients * non_watermark_mask
    + wm_sign * scaled_wm_grad        # 正部加回
    - wm_negative_sign * scaled_wm_grad  # 负部: -(-1)*scaled = +scaled  ← 符号翻转
)
```

对负梯度 `g = -2.0`：
- `wm_sign = 0`，`wm_negative_sign = -1`
- 贡献 = `-(-1) × |g| × scale` = `+ |g| × scale` → **梯度方向被反转**

### 2.2 实测验证

构造水印区梯度 `[+2, -2, ...]`、非水印区 `[+1, ...]`（`target_ratio=0.3`）：

| 位置 | 原始 | capping 后 | 判定 |
|------|------|-----------|------|
| 水印位 0 | `+2.000` | `+0.300` | OK（等比缩小） |
| 水印位 1 | `-2.000` | `+0.300` | **符号翻转** |

即负梯度被变成了正梯度。梯度下降时这等价于**朝错误方向更新**水印参数。

### 2.3 修复

改为对水印区梯度**整体等比缩放**，天然保留符号：

```python
watermark_grad_only = gradients * watermark_mask
corrected_gradients = (
    gradients * non_watermark_mask
    + watermark_grad_only * scale_factor
)
```

---

## 3. Bug 2：`clamp(0.1)` 下限导致 capping 永远达不到目标

### 3.1 问题代码

```python
scale_factor = (target_wm_grad / current_wm_grad_val).clamp(0.1, 1.0)
```

当原始 `wm/non_wm` 比例远大于 `target_ratio` 时，理论所需的
`scale_factor` 远小于 `0.1`，但被下限截断在 `0.1`。

于是实际达成的比例为：

```
实际比例 = 0.1 × (原始 wm/non_wm 比例)
```

**永远不可能等于 `target_ratio`**。

### 3.2 实测验证

水印区梯度 `[10.0]×108`、非水印区 `[1.0]×16`（原始比例 = 10.0）：

| `target_ratio` | 期望达成 | 实际达成（修复前） | 判定 |
|---------------|---------|------------------|------|
| 0.5 | 0.5 | **1.0000** | 未达成 |
| 0.3 | 0.3 | **1.0000** | 未达成 |
| 0.1 | 0.1 | **1.0000** | 未达成 |
| 0.05 | 0.05 | **1.0000** | 未达成 |

注意实际达成比例恒为 `1.0`——因为 `0.1 × 10.0 = 1.0`，
即**缩放被完全抵消**，梯度几乎原样保留，capping 等于没做。

### 3.3 修复

```python
scale_factor = (target_wm_grad / current_wm_grad_val).clamp(min=0.0, max=1.0)
```

下限改为 `0.0`（数学上所需缩放系数必然 ≥ 0，截断到 0 只在梯度为 0 时发生）。

---

## 4. 附加发现：日志打印的是 capping **之前**的比例

`multi_loss.py` 原先打印的 `wm/non_wm ratio` 是在缩放**之前**统计的
（`self.current_wm_ratio` 在 capping 之前赋值），
因此**无法用它验证 capping 是否生效**——这正是问题长期未被发现的直接原因。

修复后新增统计字段：

| 字段 | 含义 |
|------|------|
| `current_wm_ratio_after_cap` | capping **之后**水印区梯度的实际比例 |
| `cap_applied` | 本次是否真的执行了缩放 |
| `cap_scale_factor` | 本次使用的缩放系数 |

日志输出同步增加：

```
[MultiLoss] E1 | ... | wm/non_wm ratio:0.975 | wm/non_wm ratio(after cap):0.298 | cap_applied:True | ...
```

---

## 5. 回归测试

新增 `tests/test_watermark_grad_capping.py`，**不依赖 pytest**，可直接运行：

```bash
python tests/test_watermark_grad_capping.py
```

覆盖 12 项：

| 用例 | 验证内容 |
|------|---------|
| `test_watermark_gradient_sign_is_preserved` | 符号保持（Bug 1） |
| `test_negative_gradient_stays_negative` | 负梯度不为正（Bug 1） |
| `test_scaling_is_proportional` | 缩放等比（Bug 1） |
| `test_target_ratio_is_actually_achieved` | 目标达成（Bug 2） |
| `test_achieved_ratio_tracks_target_across_range` | 多目标扫描（Bug 2） |
| `test_after_cap_ratio_not_exceed_target` | 不超目标 |
| `test_after_cap_ratio_field_is_tracked` | 新统计字段 |
| `test_after_cap_ratio_when_cap_disabled` | 关闭时记录一致 |
| `test_non_watermark_region_is_untouched` | 非水印区不受影响 |
| `test_no_scaling_when_below_threshold` | 未超阈值不缩放 |
| `test_target_ratio_zero_disables_capping` | `0` 关闭 capping |
| `test_malicious_client_skips_scaling` | 恶意客户端跳过 |

**验证测试有效性**：在修复前的代码上运行，**5 个用例会失败**
（符号相关 3 个 + 目标达成相关 2 个）；修复后 **12/12 通过**。

---

## 6. 敏感性扫描工具

新增 `ratio_sweep_experiment.py`，用于系统评估 ratio 对鲁棒性的影响。

```bash
# 扫描一组 ratio
python ratio_sweep_experiment.py --dataset chestmnist --model_name alexnet \
    --client_num 5 --epochs 30 --ratios 0.0 0.1 0.2 0.3 0.4 0.5

# 仅从已有结果汇总，不重训
python ratio_sweep_experiment.py --dataset chestmnist --model_name alexnet --analyze-only
```

特点：
- **可复现**：各 ratio 使用相同随机种子，数据划分/客户端采样/泄漏客户端一致，可直接对比
- **产物隔离**：结果写入 `save/ratio_sweep/<tag>/ratio<pXX>/`
- **自动汇总**：检测准确率、追踪准确率、最佳精度、capping 前后比例，一屏呈现
- **生效性校验**：自动检查「capping 后实际比例」是否贴合目标值
- **未完成标记**：训练未跑完的实验标注 `*未完成`，不参与最优 ratio 评选

> 注意：`main.py` 中 `save/trace` 与 `save/detection` 路径是**硬编码**的
> （`os.path.join('save', 'trace')`），无法通过参数重定向。
> 脚本采用「运行前快照 → 运行后取差集」定位本次实验的新产物，并复制一份到隔离目录。

### 6.1 最终扫描结果

配置：`chestmnist / alexnet / K=5 / 30 轮`，全部 6 个 ratio 跑完。

| ratio | 检测准确率% | 追踪准确率% | 最佳精度% | capping前比例 | capping后均值 | **达成比例** | 触发次数 |
|------:|-------------:|------------:|----------:|--------------:|-------------:|------------:|--------:|
| 0.0 | 100.00 | 3.33\* | 94.7880 | 1.037 | 1.037 | —（已关闭） | 0 |
| 0.1 | 100.00 | 3.33\* | 94.7899 | 1.027 | 0.217 | **0.100** | 25 |
| 0.2 | 100.00 | 3.33\* | 94.7921 | 1.027 | 0.303 | **0.200** | 25 |
| 0.3 | 100.00 | 3.33\* | 94.7940 | 1.012 | 0.393 | **0.299** | 25 |
| 0.4 | 100.00 | 3.33\* | 94.7899 | 1.022 | 0.478 | **0.399** | 25 |
| 0.5 | 100.00 | 3.33\* | 94.7912 | 1.016 | 0.557 | **0.499** | 25 |

\* 追踪准确率一列受第 7 节的 `4877a2c` 回归影响（本次扫描在修复前运行），
与 ratio 无关；合并第 10 节的修复后重跑即可得到有效值。

### 6.2 扫描得到的三个结论

**结论 1：修复后 capping 精确达成目标值。**
「达成比例」= 仅在**真正触发 capping** 的那些行上测得的 after 值，
全部与目标值吻合到 0.001 以内（0.100/0.200/0.299/0.399/0.499）。
修复前该值恒为 1.0（见第 3 节），即 capping 形同虚设。

**结论 2：capping 前比例约 1.0，说明主任务梯度确实会淹没水印梯度。**
未缩放时水印区梯度均值约为非水印区的 **1.0 倍**，
因此 capping 几乎每轮都会触发（30 轮中触发 25 次）。
这从数据上印证了「需要 capping」的必要性。

**结论 3：在本配置下，ratio 对主任务精度几乎无影响，但 capping 后均值会偏离目标。**
最佳精度在 94.7880% ~ 94.7940% 之间波动（极差 0.006 个百分点），
且**非单调**（0.3 最高，0.4 回落）。这说明：

- 修复后的 capping 是**温和**的干预，不会明显损害主任务精度；
- 「capping 后均值」随 ratio 增大而增大（0.217 → 0.557），
  但「达成比例」始终贴合目标——说明少数**未触发** capping 的轮次
  （原始比例本就低于目标）拉高了均值，这是预期行为而非 bug；
- 由于检测准确率在所有 ratio 下都是 100%，**本次扫描无法区分
  ratio 的优劣**，无法据此调整默认值。

> **重要提醒**：检测准确率在本配置下已经**饱和**（100%），
> 因此该指标无法反映 ratio 变化带来的差异。
> 若要真正评估 ratio 的影响，需要：
> 1. 合并第 10 节的追踪修复，让「追踪准确率」恢复区分力；
> 2. 在**噪声攻击更强**或**客户端数更多**（如 K=20）的配置下重跑，
>    使检测准确率脱离饱和区。

---

## 7. 附带发现（与 ratio 无关，但影响鲁棒性指标）：训练前泄漏追踪已退化

### 7.1 现象

扫描时 `ratio=0.0` 组的指标出现异常：

| 指标 | 2026-10-21 的 150 轮历史结果 | 本次 30 轮结果 |
|------|---------------------------|--------------|
| 水印偏差**检测**准确率 | 100% | 100% |
| 训练前泄漏**追踪**准确率 | **100%** | **3.33%**（≈ 随机，K=5） |

追踪准确率退化为随机水平（1/K），且从逐轮明细看，
**泄漏者的相似度排名反而排在最后**：

```
round 2  actual C4  detected C2   rankings: [(4, 0.99934), ... ]  <- C4 排最后
round 3  actual C0  detected C4
...
```

### 7.2 根因定位

对比 git 历史，定位到**今天（2026-10-26 15:43）刚合入**的提交：

```
4877a2c feat: 服务器端水印权威覆盖
```

该提交把 `_watermark_aggregation` 从 no-op 改成：
在服务器端聚合后，将 `w_avg` 中**所有**客户端的水印区域 `P_k (k=0..N-1)`
统一覆盖为 `encoder_params` 权威值。

### 7.3 机理

追踪算法（`_track_model_leakage`）的原理是：
比较泄漏模型的水印分片与各客户端「权威水印」的余弦相似度，
**相似度最高者即为泄漏者**。

| | 4877a2c 之前 | 4877a2c 之后 |
|---|---|---|
| `w_t` 的 `P_k` | FedAvg 结果（每轮变化） | **全部 = 权威值** |
| 定制模型 `M_k` 的 `P_k` | 权威值 | 权威值 |
| 定制模型 `M_k` 的 `P_j (j≠k)` | FedAvg 值（非权威） | **权威值** |
| `M_k` 的水印区域内容 | 随 k 变化 | **对所有 k 完全相同** |
| 客户端可区分性 | 有 | **无** |

叠加噪声攻击的效果更糟：噪声只加在**泄漏者自己的** `P_k` 上，
于是泄漏者的相似度反而**最低**，检测退化为「排除法」式的随机选择。

### 7.4 数值验证

用真实密钥矩阵 + 真实 encoder 参数直接复现：

```
场景 A（4877a2c 之前，M_k 只有 P_k 是权威值）
  泄漏者 C0 -> 检测 C0 OK   sims=[(0, 1.0), (1, 0.9545), (3, 0.9543), ...]
  泄漏者 C1 -> 检测 C1 OK   sims=[(1, 1.0), (0, 0.9581), ...]
  ...全部正确

场景 B（4877a2c 之后，所有水印区域都是权威值）
  M0 ~ M4 的水印区域完全相同: True   (全部)
  泄漏者 C0 -> 检测 C1 错误(退化)  泄漏者相似度排名=5 (最后)
  泄漏者 C1 -> 检测 C0 错误(退化)  泄漏者相似度排名=5
  ...全部退化
```

### 7.5 与本次修复的关系

**该退化与 `watermark_grad_target_ratio` 的两个 bug 无关**：

- `ratio=0.0` 时 capping 完全关闭，梯度原样保留，
  我的修复在该配置下**不产生任何行为差异**；
- 而该配置下追踪准确率依然只有 3.33%，证明退化来自 `4877a2c`；
- 历史 100% 的追踪结果产生于 `4877a2c` 之前（2026-10-21），
  当时 `_watermark_aggregation` 仍是 no-op。

**因此：扫描出的「追踪准确率」一列暂时不可用于评估 ratio 的影响**，
需要先处理 `4877a2c` 引入的追踪退化，再重跑扫描。

### 7.6 建议

`4877a2c` 的意图是让服务器端持有权威水印（这对「训练后水印偏差检测」有益，
该指标目前仍是 100%）。但它同时抹掉了客户端专属指纹，
破坏了「训练前泄漏追踪」赖以工作的区分性。

> **已于 2026-10-26 在分支 `fix/pre-training-tracing-regression` 修复并验证**，
> 详见第 10 节。

可选方案：

1. **恢复 no-op**：`_watermark_aggregation` 回到不做覆盖，
   保留 `4877a2c` 之前 100% 的追踪能力（但检测准确率需重新验证）。**← 已采用**
2. **保留覆盖、但为每个客户端引入差异化**：例如在 `M_k` 的 `P_j (j≠k)`
   上叠加与 k 相关的扰动，使客户端之间仍可区分。
3. **改用其它可区分特征追踪**：放弃「水印区域数值比对」，
   改用参数范数/梯度指纹等客户端特有信号。

> 该问题属于 `4877a2c` 引入的独立回归，已在**独立分支**处理，
> 避免与 ratio 排查工作混在一起。

---

## 8. Windows 编码问题（环境相关，非本次改动引入）

`main.py` / `utils/args.py` 中大量 `print` 含 emoji（如 `✅`、`⚠️`），
在 Windows 默认 GBK 控制台下会抛 `UnicodeEncodeError`：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'
```

扫描脚本已通过为子进程设置 `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` 规避。
**直接手工运行 `main.py` 时同样会遇到该问题**，建议同样设置环境变量：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
```

---

## 9. 后续建议

1. **重跑 ratio 扫描**：在修复后的代码上重新扫描，才能得到有效的
   「ratio ↔ 鲁棒性」权衡曲线，进而决定默认值。
   **注意**：当前扫描的检测准确率已饱和在 100%，本身没有区分力（见 6.2 结论 3）。

2. ~~**先处理 `4877a2c` 的追踪退化**~~：**已处理**（见第 10 节，
   分支 `fix/pre-training-tracing-regression`）。合并该修复后重跑扫描，
   「追踪准确率」一列才会恢复参考价值。

3. **重新评估既有结论**：此前「AlexNet + CIFAR 追踪准确率仅 17%~31%」的归因
   （主任务梯度挤压 / 模型容量竞争）需要重新验证——在 capping 实际未生效的前提下，
   该现象可能主要由 Bug 1（梯度方向反转）造成，而非 ratio 设置不当。

4. **考虑把符号断言写进训练循环**：在 `compute_loss_and_backward` 中
   断言缩放前后梯度符号一致，作为运行期保护。

5. **让检测指标脱离饱和区**：当前检测准确率恒为 100%，无法评估 ratio 影响。
   可尝试更强噪声（`--leak_noise_strong` 调大）、更多客户端（K=20）、
   或增加攻击轮数占比（`--leak_interval` 调大）以拉低基线。

6. **合并两个分支后跑一次完整对照**：`feature/watermark-grad-ratio-robustness`
   （capping 修复）+ `fix/pre-training-tracing-regression`（追踪修复），
   在 `cifar10`/`cifar100`/`chestmnist` × `alexnet`/`resnet` 上重跑，
   重新评估第 3 点的历史结论。

---

## 10. 训练前泄漏追踪回归的修复与验证

**分支**: `fix/pre-training-tracing-regression`（基于 `feature/watermark-grad-ratio-robustness`）
**提交**: `b8ee893`

### 10.1 修复内容

把 `_watermark_aggregation` 回退为 **no-op**（聚合阶段保持纯 FedAvg）。

关键论证：**水印权威值本来就在下发阶段被正确写入**，服务器端聚合阶段
的覆盖是多余的：

| 阶段 | 动作 | 是否写入权威值 |
|---|---|---|
| `_generate_customized_model` | 客户端 k 的 `P_k` 覆盖为权威值 | ✅ 是 |
| `_embed_watermark` | 本地训练前把权威值写回模型 | ✅ 是 |
| `_watermark_aggregation`（4877a2c 新增） | 把**所有** `P_k` 覆盖为权威值 | ⚠️ 冗余且有害 |

因此「训练后水印偏差检测」（`_watermark_deviation_detection`）**不依赖**聚合阶段的覆盖——
它比较的是「泄漏模型在 `P_leaked` 的分片」与「各客户端定制模型在 `P_client` 的分片」，
而定制模型的 `P_client` 恒为权威值（由下发阶段保证）。

### 10.2 确定性回归测试

新增 `tests/test_pre_training_tracing.py`（不依赖 pytest），用**真实密钥矩阵 +
真实 encoder 参数**构造定制模型，直接对比两种聚合行为：

```
场景: 4877a2c(覆盖)
  第1轮: 实际泄漏者=C0 -> 检测=C0 | top sim=0.999057, 次高=C2 sim=0.073742 | 区分度=0.925315
  [症状检查] 区域 P0 在各客户端定制模型中是否完全相同: True
  第2轮: 实际泄漏者=C1 -> 检测=C4 | top sim=1.000000, 次高=C2 sim=1.000000 | 区分度=0.000000
  第3轮: 实际泄漏者=C2 -> 检测=C4 | top sim=1.000000, 次高=C2 sim=1.000000 | 区分度=0.000000
  => 追踪准确率: 1/3 = 33.33%

场景: 修复后(纯FedAvg)
  第1轮: 实际泄漏者=C0 -> 检测=C0 | top sim=0.999057, 次高=C2 sim=0.073742 | 区分度=0.925315
  [症状检查] 区域 P0 在各客户端定制模型中是否完全相同: False
  第2轮: 实际泄漏者=C1 -> 检测=C1 | top sim=0.998797, 次高=C2 sim=0.838115 | 区分度=0.160683
  第3轮: 实际泄漏者=C2 -> 检测=C2 | top sim=0.999960, 次高=C0 sim=0.959490 | 区分度=0.040470
  => 追踪准确率: 3/3 = 100.00%
```

该结果与真实训练中观察到的现象**完全一致**（第 1 轮 0.9987 正确，
第 2 轮起全部并列 1.0000，区分度归零）。

### 10.3 端到端验证

在修复后的代码上跑真实训练（chestmnist / alexnet / K=5 / 4 轮）：

| 指标 | 4877a2c 之后（修复前） | 修复后 |
|---|---|---|
| 训练前泄漏**追踪**准确率 | 3.33%（≈ 随机） | **100%**（4/4） |
| 训练后水印偏差**检测**准确率 | 100% | **100%**（4/4） |

逐轮追踪相似度（修复后）：

```
round 1: actual=C3 detected=C3 | top=0.998651  2nd=0.031056
round 2: actual=C4 detected=C4 | top=0.999041  2nd=0.834522
round 3: actual=C0 detected=C0 | top=0.998437  2nd=0.949508
round 4: actual=C4 detected=C4 | top=0.999074  2nd=0.957937
```

**结论**：回退服务器端覆盖后，「训练前追踪」与「训练后检测」
**两个指标同时保持 100%**，实现了双赢，无需引入方案 2 的差异化扰动。

---

## 11. 服务器端权威模型与下发模型的分离（最终方案）

> **2026-10-27 与作者确认后的最终方案**。作者澄清了设计意图：
> 服务器端最终保存的权威模型带有**所有客户端的完整水印**，
> 该模型**仅存服务器端**，用于后续的水印鲁棒性检测实验；
> 而**下发的定制模型**是另一个对象——客户端 k 负责的水印区域 `P_k` 每轮保持不变，
> 但**其它客户端的水印区域参数始终在变动**，因此客户端观测不到别人的水印。

### 11.1 原实现的问题

`4877a2c` 把这两件事混为一谈：它在 `_watermark_aggregation` 中把 `w_t` 的
**所有** `P_k` 覆盖为权威值，而下发的 `M_k` 正是从 `w_t` 派生的：

```
w_t 的 P_j = 权威值（固定）  →  M_k 的 P_j (j≠k) = 权威值（固定）
                              →  泄漏模型各分片相似度全部并列 1.0
                              →  训练前追踪退化为随机（3.33%）
```

### 11.2 分离方案

引入**独立对象** `self.server_authoritative_model`，与下发模型彻底分离：

| 对象 | 水印区域内容 | 用途 | 是否下发 |
|------|-------------|------|---------|
| `w_t`（下发用全局模型） | `P_k` 保持纯 FedAvg（每轮变动） | 联邦训练 + 生成定制模型 | ✅ 下发 |
| `M_k`（定制模型） | `P_k` = 权威值；`P_j (j≠k)` = FedAvg 值 | 客户端 k 本地训练 | ✅ 下发给 k |
| `server_authoritative_model` | **所有** `P_k` = 权威值 | 水印鲁棒性检测 / 存活分析 | ❌ **仅存服务器端** |

实现要点：

- `_build_server_authoritative_model(base_model)`：以 FedAvg 结果为基础，
  把所有客户端的水印区域覆盖为权威值，构成服务器端权威模型。
- `_fed_avg()`：聚合后 `w_t` 保持纯 FedAvg（**不下发权威覆盖**），
  另外独立构建 `server_authoritative_model`。
- 模型保存：保存的是**服务器端权威模型**（`_optimize_model_storage(save_state)`），
  确保落盘产物带完整水印；构建失败时回退到 FedAvg 模型。

### 11.3 端到端验证（chestmnist / alexnet / K=5 / 6 轮）

| 指标 | `4877a2c`（修复前） | 分离方案 |
|---|---|---|
| 训练前泄漏**追踪**准确率 | 3.33%（≈ 随机） | **100%**（6/6） |
| 训练后水印偏差**检测**准确率 | 100% | **100%**（6/6） |
| 保存模型是否含完整权威水印 | 部分（FedAvg 残留） | **是**（逐客户端 cos-sim = 0.99999998） |

保存模型的逐客户端校验：

```
client0: cos=0.99999998 maxdiff=0.000470
client1: cos=0.99999998 maxdiff=0.000278
client2: cos=0.99999998 maxdiff=0.000276
client3: cos=0.99999998 maxdiff=0.000423
client4: cos=0.99999998 maxdiff=0.000337
```

（maxdiff 来自 fp16 存储压缩，非算法误差。）

---

*报告生成时间: 2026-10-26（2026-10-27 更新）*
*排查分支: `feature/watermark-grad-ratio-robustness`*
*追踪回归修复分支: `fix/pre-training-tracing-regression`*
