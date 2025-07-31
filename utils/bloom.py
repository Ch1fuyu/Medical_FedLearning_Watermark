import hashlib

import numpy as np

class BloomFilter:
    def __init__(self, watermark_length, error_rate):
        """初始化布隆过滤器，动态计算布隆过滤器的大小和哈希函数数量。"""
        # 计算布隆过滤器的大小
        self.size = int(-watermark_length * np.log(error_rate) / (np.log(2) ** 2))

        # 计算哈希函数的数量
        self.num_hashes = int((self.size / watermark_length) * np.log(2))

        self.bit_array = np.zeros(self.size, dtype=bool)

    def _hash(self, item, seed):
        """哈希函数，使用不同的种子来生成多个哈希函数"""
        hash_value = hashlib.md5((str(item) + str(seed)).encode()).hexdigest()
        return int(hash_value, 16) % self.size  # 将哈希值映射到布隆过滤器大小的范围内

    def add(self, item):
        """添加元素到布隆过滤器"""
        for seed in range(self.num_hashes):
            index = self._hash(item, seed)
            self.bit_array[index] = True

    def check(self, item):
        """检查元素是否存在于布隆过滤器中"""
        for seed in range(self.num_hashes):
            index = self._hash(item, seed)
            if not self.bit_array[index]:
                return False
        return True

    def reset(self):
        """重置布隆过滤器"""
        self.bit_array = np.zeros(self.size, dtype=bool)


class BloomFilterManager:
    def __init__(self, watermark_length, error_rate):
        """初始化布隆过滤器管理器，动态计算哈希函数数量"""
        self.watermark_length = watermark_length
        self.error_rate = error_rate

        # 计算布隆过滤器的大小和哈希函数数量
        self.bloom_filter = BloomFilter(watermark_length, error_rate)

    def add_parameter_to_filter(self, param_index):
        """将模型的参数索引添加到布隆过滤器中"""
        self.bloom_filter.add(param_index)

    def check_parameter(self, param_index):
        """检查该模型的参数索引是否应该嵌入水印"""
        return self.bloom_filter.check(param_index)

    def reset_filter(self):
        """重置布隆过滤器"""
        self.bloom_filter.reset()
