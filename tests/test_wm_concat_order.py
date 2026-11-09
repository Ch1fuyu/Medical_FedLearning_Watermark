"""校验：水印区域参数按 client_id 升序拼接 == 服务器端权威 encoder 水印。

依据 train_key_matrix.py：
  - 水印总量 total_watermark_size = encoder_params 总数（9472）
  - equal 策略下各客户端水印大小 = 总量 // client_num，余数分给前几个客户端
  - 客户端 k 拿到的水印值 = encoder_params[offset_k : offset_k + len(P_k)]
    （见 utils/trainer_private_enhanced.py::_embed_watermark）

因此：若从模型中按 client_id 升序提取各客户端水印位置上的参数值并拼接，
结果应当与 encoder_params 完全一致。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import AlexNet
from models.light_autoencoder import LightAutoencoder
from utils.key_matrix_utils import KeyMatrixManager
from utils.watermark_sim_utils import extract_watermark_vector, cosine_similarity

KEY_MATRIX_PATH = 'save/key_matrix/alexnet/client5'
AUTOENCODER_DIR = 'save/autoencoder'


def main():
    km = KeyMatrixManager(KEY_MATRIX_PATH)

    # 1) 权威水印：encoder 参数（按 encoder.parameters() 顺序展平）
    ae = LightAutoencoder()
    enc_sd = torch.load(os.path.join(AUTOENCODER_DIR, 'encoder.pth'),
                        map_location='cpu', weights_only=False)
    ae.encoder.load_state_dict(enc_sd)
    authority = torch.cat([p.data.view(-1) for p in ae.encoder.parameters()])
    print(f'权威 encoder 水印长度: {authority.numel()}')

    # 2) 各客户端水印大小（equal 策略，余数分给前几个）
    total = authority.numel()
    n = km.client_num
    base, rem = total // n, total % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    print(f'客户端水印大小: {sizes}, 合计 {sum(sizes)}')
    assert sum(sizes) == total, '客户端水印大小之和应等于 encoder 参数量'

    # 3) offsets
    offsets = []
    acc = 0
    for s in sizes:
        offsets.append(acc)
        acc += s
    print(f'offsets: {offsets}')

    # 4) 用一个带水印的模型做验证：直接把权威水印按客户端切片嵌进模型
    model = AlexNet(3, 10, input_size=32, dropout_rate=0.5)
    state = model.state_dict()
    for cid in range(n):
        wm_slice = authority[offsets[cid]:offsets[cid] + sizes[cid]]
        state = km.embed_watermark(state, cid, wm_slice, model=model)

    # 5) 按 client_id 升序提取并拼接
    vec = extract_watermark_vector(state, km, model=model)
    print(f'提取的水印向量长度: {vec.numel()}')
    assert vec.numel() == total, '提取长度应等于权威水印长度'

    # 6) 逐客户端切片比对
    acc = 0
    for cid in range(n):
        seg = vec[acc:acc + sizes[cid]]
        ref = authority[offsets[cid]:offsets[cid] + sizes[cid]]
        maxdiff = (seg - ref).abs().max().item()
        print(f'  客户端 {cid}: 切片长度 {seg.numel()}, 与权威水印最大偏差 {maxdiff:.3e}')
        assert maxdiff < 1e-6, f'客户端 {cid} 水印切片不匹配'
        acc += sizes[cid]

    # 7) 整体余弦相似度
    sim = cosine_similarity(vec, authority)
    print(f'整体余弦相似度: {sim:.10f}')
    assert sim > 0.999999, f'整体相似度过低: {sim}'

    print('\n✅ 验证通过：按 client_id 升序拼接 == 权威 encoder 水印')


if __name__ == '__main__':
    main()
