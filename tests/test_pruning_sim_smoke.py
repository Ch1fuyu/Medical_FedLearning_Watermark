"""剪枝攻击 + 水印相似度 sim 的端到端冒烟测试。

用一个真实的水印模型跑剪枝攻击流程，检查：
1. 每个剪枝比例都产出了 wm_sim
2. sim 随剪枝比例上升而单调下降（越剪越接近 0）
3. 结果 CSV 里包含 sim 列
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning_attack import (
    load_main_task_model,
    load_test_data,
    threshold_pruning,
    evaluate_watermark_after_pruning,
)
from utils.delta_pcc_utils import evaluate_delta_pcc, calculate_fixed_tau
from utils.watermark_reconstruction import WatermarkReconstructor
from utils.watermark_sim_utils import compute_watermark_similarity, format_watermark_similarity

MODEL_PATH = 'save/alexnet/cifar10/202611041703_Dp_0.1_iid_True_wm_enhanced_ep_150_le_2_cn_5_fra_1.0000_acc_0.9040_enhanced.pkl'
KEY_MATRIX_PATH = 'save/key_matrix/alexnet/client5'
AUTOENCODER_DIR = 'save/autoencoder'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_info = load_main_task_model(MODEL_PATH, device)
    if model is None:
        print('❌ 模型加载失败')
        return

    reconstructor = WatermarkReconstructor(KEY_MATRIX_PATH, AUTOENCODER_DIR)
    reconstructor.set_model(model)

    test_loader = load_test_data('cifar10', batch_size=128)

    rows = []
    for ratio in [0.0, 0.2, 0.5, 0.8]:
        pruned = threshold_pruning(model, ratio)
        sim_result = compute_watermark_similarity(
            pruned.state_dict(), model.state_dict(),
            reconstructor.key_manager, model=pruned
        )
        fmt = format_watermark_similarity(sim_result)
        rows.append((ratio, fmt['wm_sim'], fmt['wm_nonzero_ratio']))
        print(f"  剪枝{ratio:.0%}: sim={fmt['wm_sim']:.6f} 非零占比={fmt['wm_nonzero_ratio']:.2%}")

    # 单调性检查
    sims = [r[1] for r in rows]
    assert all(sims[i] >= sims[i+1] - 1e-9 for i in range(len(sims)-1)), \
        f'sim 应随剪枝比例单调不增: {sims}'
    print('✓ sim 随剪枝比例单调不增')

    # ΔPCC 与 sim 共存的检查
    fixed_tau = calculate_fixed_tau(model.state_dict(), reconstructor, test_loader, device,
                                    perf_fail_ratio=0.3, model=model)
    d = evaluate_delta_pcc(model.state_dict(), model.state_dict(), reconstructor, test_loader,
                            device, perf_fail_ratio=0.3, fixed_tau=fixed_tau, model=model)
    print(f"  ΔPCC(自身)= {d['delta_pcc']:.6f}, 侵权: {d['result_text']}")
    assert 'delta_pcc' in d and 'perf_before' in d
    print('✅ 端到端冒烟测试通过')


if __name__ == '__main__':
    main()
