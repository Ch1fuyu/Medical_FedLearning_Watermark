"""
模型存档检查工具

用于手动检查保存的模型存档，进行后验泄露检测
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def list_archived_models(archive_dir):
    """列出存档目录中的所有模型"""
    model_list = []
    try:
        if not os.path.exists(archive_dir):
            print(f"存档目录不存在: {archive_dir}")
            return model_list

        print(f"\n{'='*80}")
        print(f"存档目录: {archive_dir}")
        print(f"{'='*80}\n")

        for round_dir in sorted(os.listdir(archive_dir), key=lambda x: int(x.split('_')[1]) if x.startswith('round_') else 0):
            round_path = os.path.join(archive_dir, round_dir)
            if not os.path.isdir(round_path):
                continue

            print(f"📁 {round_dir}")

            for filename in sorted(os.listdir(round_path)):
                if filename.endswith('.pth'):
                    filepath = os.path.join(round_path, filename)
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"   ├── {filename} ({size_kb:.1f} KB)")
                    model_list.append(filepath)

        print(f"\n总计: {len(model_list)} 个模型文件\n")
        return model_list

    except Exception as e:
        print(f"列出模型失败: {e}")
        return model_list


def load_model(filepath):
    """加载模型文件"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def compare_watermark_params(model_a, model_b):
    """
    比较两个模型的水印区域参数

    Returns:
        similarity: 余弦相似度
        diff: 参数差异统计
    """
    try:
        # 提取所有卷积层权重
        params_a = []
        params_b = []

        for key in model_a.keys():
            if 'conv' in key.lower() and 'weight' in key.lower():
                if key in model_b:
                    a_flat = model_a[key].view(-1).numpy()
                    b_flat = model_b[key].view(-1).numpy()
                    params_a.extend(a_flat[:10].tolist())  # 只取前10个元素作为示例
                    params_b.extend(b_flat[:10].tolist())

        if len(params_a) == 0:
            return 0.0, {}

        params_a = np.array(params_a)
        params_b = np.array(params_b)

        # 计算余弦相似度
        dot = np.dot(params_a, params_b)
        norm_a = np.linalg.norm(params_a)
        norm_b = np.linalg.norm(params_b)
        similarity = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        # 计算差异统计
        diff = np.abs(params_a - params_b)
        diff_stats = {
            'mean_diff': float(np.mean(diff)),
            'max_diff': float(np.max(diff)),
            'std_diff': float(np.std(diff))
        }

        return similarity, diff_stats

    except Exception as e:
        print(f"比较失败: {e}")
        return 0.0, {}


def manual_detection(archive_dir):
    """
    手动后验泄露检测

    用户可以：
    1. 查看所有存档的模型
    2. 选择一个模型作为"泄漏者"
    3. 与其他模型进行比较，找出最相似的
    """
    model_list = list_archived_models(archive_dir)
    if not model_list:
        return

    print("\n" + "="*80)
    print("手动后验泄露检测")
    print("="*80)

    # 显示模型列表供选择
    print("\n可用的模型文件:")
    for i, filepath in enumerate(model_list):
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        print(f"  {i+1}. [{filename}]")

    # 获取用户输入
    try:
        choice = input("\n请选择要检测的模型编号 (1-{}) 或输入 'q' 退出: ".format(len(model_list)))
        if choice.lower() == 'q':
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_list):
            print("无效的选择")
            return

        selected_path = model_list[idx]
        print(f"\n已选择: {os.path.basename(selected_path)}")

        # 加载选中的模型
        selected_model = load_model(selected_path)
        if selected_model is None:
            return

        selected_state = selected_model['state_dict']
        selected_info = f"Round {selected_model.get('round', '?')}, Client {selected_model.get('client_id', '?')}, Type {selected_model.get('model_type', '?')}"
        print(f"模型信息: {selected_info}")

        # 与所有其他模型比较
        print("\n正在与其他模型进行比较...")
        print("-" * 80)

        similarities = []

        for filepath in model_list:
            if filepath == selected_path:
                continue

            other_model = load_model(filepath)
            if other_model is None:
                continue

            other_state = other_model['state_dict']
            similarity, diff_stats = compare_watermark_params(selected_state, other_state)

            other_info = f"Round {other_model.get('round', '?')}, Client {other_model.get('client_id', '?')}, Type {other_model.get('model_type', '?')}"

            similarities.append({
                'filepath': filepath,
                'info': other_info,
                'similarity': similarity,
                'diff_stats': diff_stats
            })

        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # 显示结果
        print(f"\n相似度排名 (与 {os.path.basename(selected_path)} 比较):")
        print("-" * 80)
        print(f"{'排名':<6} {'轮次/客户端/类型':<40} {'相似度':<12} {'平均差异':<12}")
        print("-" * 80)

        for i, item in enumerate(similarities[:10], 1):
            marker = " ← 最相似" if i == 1 else ""
            print(f"{i:<6} {item['info']:<40} {item['similarity']:.6f}   {item['diff_stats'].get('mean_diff', 0):.6f}{marker}")

        print("-" * 80)

        # 分析结果
        if similarities:
            top = similarities[0]
            if top['similarity'] > 0.99:
                print(f"\n⚠️  警告: 发现高度相似的模型!")
                print(f"   最相似的模型: {top['info']}")
                print(f"   相似度: {top['similarity']:.6f}")
                print(f"   这可能表示模型泄露或来源关系。")
            else:
                print(f"\n✓ 未发现高度相似的模型")
                print(f"   最相似模型: {top['info']}")
                print(f"   相似度: {top['similarity']:.6f}")

    except ValueError:
        print("无效输入")
    except Exception as e:
        print(f"检测失败: {e}")


def find_latest_archive():
    """查找最新的存档目录"""
    base_dir = os.path.join(project_root, 'save', 'model_archive')

    if not os.path.exists(base_dir):
        return None

    # 获取所有子目录，按修改时间排序
    dirs = []
    for d in os.listdir(base_dir):
        dpath = os.path.join(base_dir, d)
        if os.path.isdir(dpath):
            dirs.append((dpath, os.path.getmtime(dpath)))

    dirs.sort(key=lambda x: x[1], reverse=True)

    return dirs[0][0] if dirs else None


def main():
    print("\n" + "="*80)
    print("                   模型存档检查工具")
    print("="*80)

    # 查找存档目录
    if len(sys.argv) > 1:
        archive_dir = sys.argv[1]
    else:
        archive_dir = find_latest_archive()

    if archive_dir is None or not os.path.exists(archive_dir):
        print("\n未找到模型存档目录!")
        print(f"请先运行 main.py 进行训练，模型存档将保存在:")
        print(f"  save/model_archive/")
        return

    print(f"\n找到存档目录: {archive_dir}")

    # 进入手动检测模式
    while True:
        print("\n" + "="*80)
        print("选项:")
        print("  1. 列出所有存档模型")
        print("  2. 手动后验泄露检测")
        print("  3. 切换存档目录")
        print("  q. 退出")
        print("="*80)

        choice = input("\n请选择: ").strip()

        if choice == '1':
            list_archived_models(archive_dir)

        elif choice == '2':
            manual_detection(archive_dir)

        elif choice == '3':
            new_dir = input("请输入新的存档目录路径: ").strip()
            if os.path.exists(new_dir):
                archive_dir = new_dir
                print(f"已切换到: {archive_dir}")
            else:
                print("目录不存在")

        elif choice.lower() == 'q':
            print("再见!")
            break

        else:
            print("无效选择")


if __name__ == '__main__':
    main()
