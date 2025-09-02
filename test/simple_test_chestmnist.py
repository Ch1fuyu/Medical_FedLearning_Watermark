import numpy as np
import os

def simple_test_chestmnist():
    """简单测试ChestMNIST数据文件"""
    print("=== 简单ChestMNIST测试 ===")
    
    # 检查文件是否存在
    data_path = "../data/chestmnist.npz"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    print(f"✅ 数据文件存在: {data_path}")
    
    # 加载数据
    try:
        data = np.load(data_path)
        print(f"✅ 数据加载成功")
        print(f"数据键: {list(data.keys())}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 检查标签
    if 'train_labels' in data and 'test_labels' in data:
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        
        print(f"\n训练集标签形状: {train_labels.shape}")
        print(f"测试集标签形状: {test_labels.shape}")
        
        # 检查是否为14维
        if train_labels.shape[1] == 14 and test_labels.shape[1] == 14:
            print("✅ 标签维度正确 (14维)")
        else:
            print(f"❌ 标签维度错误: 训练集{train_labels.shape[1]}维, 测试集{test_labels.shape[1]}维")
            return
        
        # 检查标签值
        print(f"训练集标签值范围: [{train_labels.min()}, {train_labels.max()}]")
        print(f"测试集标签值范围: [{test_labels.min()}, {test_labels.max()}]")
        
        # 检查样本标签
        print(f"\n第一个训练样本标签: {train_labels[0]}")
        print(f"第一个测试样本标签: {test_labels[0]}")
        
        # 统计每个类别的正样本
        print(f"\n=== 各类别正样本统计 ===")
        pathology_names = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax"
        ]
        
        for i, name in enumerate(pathology_names):
            train_pos = np.sum(train_labels[:, i])
            test_pos = np.sum(test_labels[:, i])
            train_ratio = train_pos / len(train_labels) * 100
            test_ratio = test_pos / len(test_labels) * 100
            
            print(f"{name:<15}: 训练集{train_pos:>4} ({train_ratio:>5.1f}%), 测试集{test_pos:>4} ({test_ratio:>5.1f}%)")
        
        # 检查多标签情况
        train_multi = np.sum(train_labels, axis=1)
        test_multi = np.sum(test_labels, axis=1)
        
        print(f"\n多标签统计:")
        print(f"训练集平均每样本标签数: {np.mean(train_multi):.2f}")
        print(f"测试集平均每样本标签数: {np.mean(test_multi):.2f}")
        print(f"训练集最多标签数: {np.max(train_multi)}")
        print(f"测试集最多标签数: {np.max(test_multi)}")
        
    else:
        print("❌ 数据文件中缺少标签键")

if __name__ == "__main__":
    simple_test_chestmnist()
