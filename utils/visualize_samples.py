"""
随机从 PneumoniaMNIST / PathMNIST / ChestMNIST 三个数据集中各抽取若干张样本，
拼成一张「三组×每组若干张」的网格图（无文字标注）。

布局：
  - 默认每组 4 张图像，三组共 12 张
  - 拼成两行六列：三组图像横向并排成一组，共 6 列；两行重复（同一组 4 张占 4 列，
    实际为「3 数据集 × 4 张 = 12 张」以 6 列布局，两行 × 6 列）
  - 每个数据集的图像放在一起

用法：
  python utils/visualize_samples.py --data_root D:/CODE/Medical_FedLearning_Watermark/data --out sample_grid.png
  python utils/visualize_samples.py --data_root D:/CODE/Medical_FedLearning_Watermark/data --per_dataset 6 --tile_size 384 --out samples_hd.png
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# 脚本所在目录作为默认根路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset_samples(npz_path, n_samples, seed=0):
    """从 medmnist 风格的 .npz 中随机抽取 n_samples 张样本图像。

    返回一个 list[Image.Image]（RGB）。
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    data = np.load(npz_path)
    # medmnist 数据集通常提供 train_images / val_images / test_images
    for key in ("train_images", "val_images", "test_images"):
        if key in data.files:
            images = data[key]
            break
    else:
        raise KeyError(f"No *_images array found in {npz_path}, keys={data.files}")

    rng = np.random.RandomState(seed)
    n = images.shape[0]
    indices = rng.choice(n, size=min(n_samples, n), replace=False)

    pil_images = []
    for idx in indices:
        img = images[int(idx)]
        # medmnist 图像可能是 (28,28)、(28,28,1)、(64,64,3) 等
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim == 2:
            # 灰度 -> RGB 复制为 3 通道，方便统一处理
            img = np.stack([img, img, img], axis=-1)

        # 数据可能是 uint8 或 float；统一到 uint8
        if img.dtype != np.uint8:
            arr = img.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 0:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            img = (arr * 255.0).clip(0, 255).astype(np.uint8)

        pil_images.append(Image.fromarray(img))
    return pil_images


def build_grid(groups, per_group, cols=6, tile_size=224, padding=4, bg=(255, 255, 255)):
    """把多组 PIL 图像拼成 cols 列的网格图。

    每个数据集独占一个连续的子块（rsub x csub，尽量接近正方形），
    子块在网格中从左到右、从上到下依次摆放——同数据集的图像一定彼此相邻。
    整张图行数等于单个子块的行数。
    """
    assert len(groups) >= 1
    assert all(len(g) == per_group for g in groups)

    # 每个子块的列数 / 行数：尽量接近正方形，且 csub 整除 per_group 以避免子块内空洞
    csub = min(cols, max(1, int(np.ceil(np.sqrt(per_group)))))
    while per_group % csub != 0 and csub > 1:
        csub -= 1
    rsub = (per_group + csub - 1) // csub

    blocks_per_row = max(1, cols // csub)

    cell_w = tile_size
    cell_h = tile_size

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rsub * cell_h + (rsub + 1) * padding

    canvas = Image.new("RGB", (grid_w, grid_h), bg)

    for gi, group in enumerate(groups):
        block_row = gi // blocks_per_row
        block_col = gi % blocks_per_row
        origin_x = padding + block_col * csub * (cell_w + padding) + block_col * padding
        origin_y = padding + block_row * rsub * (cell_h + padding) + block_row * padding

        for i, img in enumerate(group):
            r = i // csub
            c = i % csub
            x = origin_x + c * (cell_w + padding)
            y = origin_y + r * (cell_h + padding)
            resized = img.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(resized, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="拼接展示 medmnist 样本图像")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据集 .npz 所在目录")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["pneumoniamnist", "pathmnist", "chestmnist"])
    parser.add_argument("--per_dataset", type=int, default=4,
                        help="每个数据集抽取的图像数（默认 4，整组共 12 张；3×4 张可以拼成 2 行 6 列）")
    parser.add_argument("--cols", type=int, default=6,
                        help="网格列数（默认 6，正好两行装下 12 张）")
    parser.add_argument("--tile_size", type=int, default=256,
                        help="单张图像在网格中的像素尺寸，越大越清晰")
    parser.add_argument("--padding", type=int, default=4,
                        help="图像之间的留白像素数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="输出图像路径")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(_SCRIPT_DIR, "..", "data")

    groups = []
    for name in args.datasets:
        npz_path = os.path.join(args.data_root, f"{name}.npz")
        print(f"[info] loading {npz_path} ...")
        group_imgs = load_dataset_samples(npz_path, args.per_dataset, seed=args.seed)
        print(f"        -> got {len(group_imgs)} images, "
              f"sample size={group_imgs[0].size}")
        groups.append(group_imgs)

    canvas = build_grid(
        groups,
        per_group=args.per_dataset,
        cols=args.cols,
        tile_size=args.tile_size,
        padding=args.padding,
    )
    if args.out is None:
        out_path = os.path.join(_SCRIPT_DIR, "..", "save", "image", "sample_grid.png")
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=False)
    print(f"[done] saved grid image to: {out_path} "
          f"({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()