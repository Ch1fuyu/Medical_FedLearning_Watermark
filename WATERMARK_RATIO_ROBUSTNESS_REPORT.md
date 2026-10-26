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

### 附带发现（独立回归，与 ratio 无关）

扫描过程中还发现**今天刚合入**的 `4877a2c`（服务器端水印权威覆盖）
导致「训练前泄漏追踪」退化到随机水平（3.33%，K=5），
历史（该提交之前）为 100%。详见第 7 节。
该退化与本次 ratio 修复**无关**（`ratio=0` 关闭 capping 时依然存在），
建议另开分支处理。

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

> 注意：`main.py` 中 `save/trace` 与 `save/detection` 路径是**硬编码**的
> （`os.path.join('save', 'trace')`），无法通过参数重定向。
> 脚本采用「运行前快照 → 运行后取差集」定位本次实验的新产物，并复制一份到隔离目录。

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

可选方案：

1. **恢复 no-op**：`_watermark_aggregation` 回到不做覆盖，
   保留 `4877a2c` 之前 100% 的追踪能力（但检测准确率需重新验证）。
2. **保留覆盖、但为每个客户端引入差异化**：例如在 `M_k` 的 `P_j (j≠k)`
   上叠加与 k 相关的扰动，使客户端之间仍可区分。
3. **改用其它可区分特征追踪**：放弃「水印区域数值比对」，
   改用参数范数/梯度指纹等客户端特有信号。

> 该问题属于 `4877a2c` 引入的独立回归，建议**另开分支**处理，
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

2. **先处理 `4877a2c` 的追踪退化**（见第 7 节）：在追踪恢复之前，
   扫描结果中的「追踪准确率」一列不具备参考价值。

3. **重新评估既有结论**：此前「AlexNet + CIFAR 追踪准确率仅 17%~31%」的归因
   （主任务梯度挤压 / 模型容量竞争）需要重新验证——在 capping 实际未生效的前提下，
   该现象可能主要由 Bug 1（梯度方向反转）造成，而非 ratio 设置不当。

4. **考虑把符号断言写进训练循环**：在 `compute_loss_and_backward` 中
   断言缩放前后梯度符号一致，作为运行期保护。

---

*报告生成时间: 2026-10-26*
*排查分支: `feature/watermark-grad-ratio-robustness`*
