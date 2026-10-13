"""
MNIST 2x2 Grid Image Creator
从MNIST数据集中随机选取4张图片，拼接成一张2x2的高清图片
使用PIL创建高清拼接图
"""

import numpy as np
from PIL import Image
import struct
import os

def read_mnist_images(filepath):
    """读取MNIST图片文件"""
    print(f"正在读取: {filepath}")
    with open(filepath, 'rb') as f:
        # 读取魔数、图片数量、行数、列数
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"Magic: {magic}, 图片数: {num_images}, 尺寸: {rows}x{cols}")
        # 读取图片数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

# MNIST数据路径
data_path = 'data/MNIST/raw'
train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte')

# 加载训练图片
images = read_mnist_images(train_images_path)
print(f"成功加载 {len(images)} 张MNIST图片")

# 随机选取4个图片
indices = np.random.choice(len(images), size=4, replace=False)
selected_images = [images[i] for i in indices]

print(f"随机选取的图片索引: {list(indices)}")
print(f"图片标签: {[int(images[i][0, 0]) if False else '?' for i in indices]}")  # MNIST图片没有标签在像素里

# 每张图片放大的尺寸（原始28x28，放大到280x280 = 10倍）
scale = 10
img_size = 28 * scale

# 每个格子的尺寸
grid_cell = img_size + 20  # 加上间距

# 创建白色背景的大图 (2x2格子 + 边距)
total_size = 2 * grid_cell + 40  # 2个格子 + 边距
canvas = np.ones((total_size, total_size), dtype=np.uint8) * 255

# 放置4张图片到2x2格子中
positions = [(40, 40), (40 + grid_cell, 40), (40, 40 + grid_cell), (40 + grid_cell, 40 + grid_cell)]

for idx, (img, (x, y)) in enumerate(zip(selected_images, positions)):
    # 放大图片 (nearest neighbor插值保持清晰)
    img_enlarged = Image.fromarray(img).resize((img_size, img_size), Image.Resampling.NEAREST)
    img_enlarged = np.array(img_enlarged)
    
    # 放置到画布上
    canvas[y:y+img_size, x:x+img_size] = img_enlarged

# 保存为高清PNG
output_path = 'mnist_2x2_grid.png'
result = Image.fromarray(canvas)
result.save(output_path, dpi=(200, 200))

print(f"图片已保存到: {output_path}")
print(f"图片尺寸: {result.size}")
print("完成!")
