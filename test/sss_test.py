import hmac
import hashlib
import numpy as np

def generate_watermark(public_string, private_string, model_params):
    """生成模型水印（身份标签）"""
    # 1. 将模型参数展平并拼接成字节流
    params_bytes = np.concatenate([p.flatten() for p in model_params]).tobytes()
    
    # 2. 使用HMAC生成水印
    h = hmac.new(
        private_string.encode('utf-8'),
        params_bytes + public_string.encode('utf-8'),
        hashlib.sha256
    )
    return h.digest()  # 返回字节形式的水印

def embed_watermark(model_params, watermark, strength=0.001):
    """将水印嵌入到模型参数中"""
    # 1. 将水印转换为与模型参数相同形状的数组
    flat_params = np.concatenate([p.flatten() for p in model_params])
    target_shape = flat_params.shape
    
    # 2. 将水印哈希转换为浮点数数组
    watermark_array = np.frombuffer(watermark, dtype=np.uint8).astype(np.float32)
    # 扩展水印数组至与参数长度匹配
    watermark_array = np.resize(watermark_array, target_shape)
    # 归一化到[-1, 1]范围
    watermark_array = (watermark_array / 127.5) - 1.0
    
    # 3. 将水印嵌入模型参数（添加微小扰动）
    perturbed_params = flat_params + (watermark_array * strength)
    
    # 4. 恢复原始参数形状
    new_params = []
    start_idx = 0
    for param in model_params:
        size = param.size
        new_param = perturbed_params[start_idx:start_idx+size].reshape(param.shape)
        new_params.append(new_param)
        start_idx += size
    
    return new_params

def extract_watermark(model_params, public_string, private_string):
    """从模型参数中提取水印并验证"""
    # 1. 提取可能包含水印的参数
    params_bytes = np.concatenate([p.flatten() for p in model_params]).tobytes()
    
    # 2. 重新计算预期水印
    expected_watermark = hmac.new(
        private_string.encode('utf-8'),
        params_bytes + public_string.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    return expected_watermark

def verify_client(model_params, public_string, private_string, claimed_watermark):
    """验证客户端身份"""
    extracted_watermark = extract_watermark(model_params, public_string, private_string)
    return hmac.compare_digest(extracted_watermark, claimed_watermark)