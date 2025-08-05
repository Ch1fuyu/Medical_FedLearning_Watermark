#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装MedMNIST依赖脚本
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        print(f"✓ {package} 已安装")
        return True
    except ImportError:
        print(f"✗ {package} 未安装")
        return False

def main():
    """主函数"""
    print("=== 安装MedMNIST依赖 ===")
    
    # 需要安装的包
    packages = [
        "medmnist",
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "tqdm"
    ]
    
    # 检查已安装的包
    print("\n检查已安装的包:")
    installed_packages = []
    for package in packages:
        if check_package(package.replace("-", "_")):
            installed_packages.append(package)
    
    # 安装缺失的包
    missing_packages = [pkg for pkg in packages if pkg not in installed_packages]
    
    if missing_packages:
        print(f"\n需要安装的包: {missing_packages}")
        for package in missing_packages:
            install_package(package)
    else:
        print("\n所有依赖包已安装！")
    
    # 验证MedMNIST安装
    print("\n验证MedMNIST安装:")
    try:
        from medmnist import INFO
        print("✓ MedMNIST导入成功")
        print(f"可用数据集: {list(INFO.keys())}")
        
        # 测试DermaMNIST
        from medmnist import INFO
        if 'dermamnist' in INFO:
            print("✓ DermaMNIST数据集可用")
            derma_info = INFO['dermamnist']
            print(f"  类别数: {derma_info['n_classes']}")
            print(f"  图像尺寸: {derma_info['image_size']}")
            print(f"  任务类型: {derma_info['task']}")
        else:
            print("✗ DermaMNIST数据集不可用")
            
    except ImportError as e:
        print(f"✗ MedMNIST导入失败: {e}")
        return False
    
    print("\n🎉 依赖安装完成！")
    return True

if __name__ == '__main__':
    main() 