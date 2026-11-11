"""为抗合谋分支生成 v2 见证点布局（**不改动** key_matrix.pth / positions.json）。

背景
----
抗合谋分支的 6 个密钥矩阵里有 5 个的 witness_layout.json 还是 **v1 旧格式**
（见证点值取自 encoder 参数切片），计数解码不可用；只有 resnet/client10
是 v2。

v1 -> v2 的变化只涉及 **见证点取值来源**（改为密钥单向派生）与
**副本分散方式**（spread_perm），**不涉及水印位置**：

- 见证点位置始终取自"未被任何客户端分配的 conv 位置"（空闲池），
  与独占切片严格不相交；
- 空闲池的顺序由 conv 层参数顺序决定，与密钥无关。

因此本脚本**只重新生成 witness_layout.json**：
  - key_matrix.pth 与 positions.json 一律不动
    （那决定的是传统独占切片水印，改了就等于换了水印）；
  - 见证点位置从现有 positions.json 反推出的空闲池中取；
  - 见证点取值改用固定密钥文件（save/keys/witness_secret_collusion.key）派生。

conv 层形状只取决于 in_channels（num_classes 只影响 FC 层，
input_size 对这些架构的 conv 形状无影响），因此只要 conv 参数量对得上，
位置池就与原布局完全一致。

用法::

    python migrate_collusion_layouts_v2.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.alexnet import AlexNet
from models.resnet import resnet18
from utils.collusion_resistant import build_witness_layout, save_witness_layout
from utils.witness_secret import (DEFAULT_SECRET_PATH, load_or_create_secret,
                                  secret_fingerprint)

ROOT = os.path.dirname(os.path.abspath(__file__))
COLLUSION_DIR = os.path.join(ROOT, 'save', 'key_matrix_collusion')
BACKUP_DIR = os.path.join(ROOT, 'save', 'key_matrix_collusion_backup_v1')

#: (num_classes, in_channels, input_size)
DATASET_PRESETS = {
    'chestmnist': (14, 1, 28),
    'dermamnist': (7, 3, 28),
    'octmnist': (4, 1, 28),
    'pneumoniamnist': (2, 1, 28),
    'retinamnist': (5, 3, 28),
    'cifar10': (10, 3, 32),
    'cifar100': (100, 3, 32),
    'imagenet': (1000, 3, 224),
}


def build(model_class, nc, ic, size):
    if model_class == 'ResNet':
        return resnet18(num_classes=nc, in_channels=ic, input_size=size)
    return AlexNet(ic, nc, input_size=size, dropout_rate=0.5)


def conv_param_info(model):
    """复刻 train_key_matrix.KeyMatrixGenerator._get_model_param_info。"""
    out = []
    for name, param in model.named_parameters():
        is_conv = (('conv' in name.lower() and 'weight' in name.lower())
                   or 'downsample.0.weight' in name.lower())
        if is_conv and len(param.shape) == 4:
            out.append({'name': name, 'numel': param.numel()})
    return out


def all_conv_positions(model):
    """全部可用 conv 位置（与生成器同序：named_parameters 顺序 + 局部索引）。

    注意：生成器内部虽然把这份列表 shuffle 过用于分配独占切片，
    但**空闲池**是用未 shuffle 的顺序重建的，本函数与之保持一致。
    """
    out = []
    for info in conv_param_info(model):
        for i in range(info['numel']):
            out.append((info['name'], i))
    return out


def pick_model(model_class, conv_total):
    """选出 conv 参数量匹配的模型（conv 形状一致 => 位置池一致）。"""
    for dname, (nc, ic, size) in DATASET_PRESETS.items():
        model = build(model_class, nc, ic, size)
        infos = conv_param_info(model)
        if sum(i['numel'] for i in infos) == conv_total:
            return dname, model
    return None, None


def main():
    # ---- 密钥（长期不变，固定路径）----
    secret, created = load_or_create_secret(DEFAULT_SECRET_PATH, create=True)
    fp = secret_fingerprint(secret)
    print(f'见证点密钥: {DEFAULT_SECRET_PATH} '
          f'({"新建" if created else "已存在"}, fingerprint={fp})')
    print()

    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    for model_dir in sorted(os.listdir(COLLUSION_DIR)):
        mpath = os.path.join(COLLUSION_DIR, model_dir)
        if not os.path.isdir(mpath):
            continue
        for client_dir in sorted(os.listdir(mpath)):
            cpath = os.path.join(mpath, client_dir)
            info_p = os.path.join(cpath, 'key_matrix_info.json')
            if not os.path.exists(info_p):
                continue
            info = json.load(open(info_p))
            model_class = info.get('model_type')        # 'ResNet' / 'AlexNet'
            client_num = info.get('client_num')
            conv_total = info.get('total_model_params')  # 实为 conv 参数量
            encoder_params = info.get('encoder_params')

            # 读现有位置（**只读**，绝不修改）
            positions = {}
            for cid in range(client_num):
                pp = os.path.join(cpath, f'client_{cid}', 'positions.json')
                positions[cid] = [tuple(x) for x in json.load(open(pp))]

            dname, model = pick_model(model_class, conv_total)
            if model is None:
                print(f'!! {model_dir}/{client_dir}: '
                      f'找不到 conv 参数量 {conv_total} 匹配的模型，跳过')
                continue

            # 空闲位置池 = 全部 conv 位置 - 所有客户端已占用的位置
            used = set()
            for cid in range(client_num):
                used.update(positions[cid])
            all_pos = all_conv_positions(model)
            free_pool = [p for p in all_pos if p not in used]
            slice_size = len(positions[0])

            # 备份旧布局
            layout_p = os.path.join(cpath, 'witness_layout.json')
            old_v = old_g = None
            old_pool_match = None
            if os.path.exists(layout_p):
                shutil.copy2(layout_p, os.path.join(
                    BACKUP_DIR, f'{model_dir}_{client_dir}_witness_layout.json'))
                old = json.load(open(layout_p))
                d = old.get('design', {})
                old_v, old_g = d.get('v'), d.get('G')

            # 生成 v2 布局
            layout = build_witness_layout(
                n_clients=client_num,
                K=info.get('collusion_K', 3),
                encoder_params=encoder_params,
                positions=free_pool,
                slice_size=slice_size,
                secret=secret,
                amplitude=0.05,
                verbose=False,
            )
            save_witness_layout(layout, layout_p)

            # 校验：新旧布局的见证点位置集合是否一致
            if old_v:
                old_pool = set(free_pool[:old_v * old_g])
                new_pool = set(layout.positions)
                old_pool_match = (old_pool == new_pool)

            print(f'✅ {model_dir}/{client_dir}  dataset={dname}  '
                  f'v={layout.v} G={layout.G} K={layout.design.K}  '
                  f'切片={slice_size}  空闲池={len(free_pool)}  '
                  f'旧(v={old_v},G={old_g}) -> 位置集合一致={old_pool_match}')

    print(f'\n旧布局备份于: {BACKUP_DIR}')


if __name__ == '__main__':
    main()
