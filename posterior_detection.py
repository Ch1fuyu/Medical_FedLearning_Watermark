"""
独立的后验泄露检测脚本
========================

功能：
1. 加载已有的模型存档（无需重新训练）
2. 对存档中的每个模型进行"后验检测全覆盖"
3. 生成检测报告（保存到 save/leak_detection/ 目录）

使用方法：
    python posterior_detection.py                              # 自动加载最新的存档
    python posterior_detection.py <archive_dir>                # 指定存档目录
    python posterior_detection.py <archive_dir> --threshold 0.99  # 指定相似度阈值

依赖：
    需要先通过 main.py 进行训练，生成了模型存档
"""

import os
import sys
from utils.args import parser_args
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(project_root, 'save', 'posterior_detection.log'),
            mode='a',
            encoding='utf-8'
        )
    ]
)


def setup_directories():
    """创建必要的目录"""
    dirs = [
        os.path.join(project_root, 'save'),
        os.path.join(project_root, 'save', 'leak_detection'),
        os.path.join(project_root, 'save', 'model_archive'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_latest_archive(base_dir):
    """查找最新的存档目录"""
    if not os.path.exists(base_dir):
        return None

    dirs = []
    for d in os.listdir(base_dir):
        dpath = os.path.join(base_dir, d)
        if os.path.isdir(dpath):
            # 检查是否包含round子目录
            has_rounds = any(item.startswith('round_') for item in os.listdir(dpath))
            if has_rounds:
                dirs.append((dpath, os.path.getmtime(dpath)))

    dirs.sort(key=lambda x: x[1], reverse=True)
    return dirs[0][0] if dirs else None


def list_archived_models(archive_dir):
    """
    列出存档目录中的所有模型

    Returns:
        models: {
            'customized': {(round, client_id): filepath},
            'trained': {(round, client_id): filepath}
        }
    """
    models = {
        'customized': {},
        'trained': {}
    }

    if not os.path.exists(archive_dir):
        return models

    for round_dir in sorted(os.listdir(archive_dir)):
        round_path = os.path.join(archive_dir, round_dir)
        if not os.path.isdir(round_path):
            continue

        if not round_dir.startswith('round_'):
            continue

        try:
            round_num = int(round_dir.split('_')[1])
        except (IndexError, ValueError):
            continue

        for filename in os.listdir(round_path):
            if not filename.endswith('.pth'):
                continue

            # 解析文件名: client_X_customized.pth 或 client_X_trained.pth
            parts = filename[:-4].split('_')
            if len(parts) < 3:
                continue

            try:
                client_id = int(parts[1])
            except ValueError:
                continue

            model_type = parts[2]
            if model_type not in models:
                continue

            filepath = os.path.join(round_path, filename)
            models[model_type][(round_num, client_id)] = filepath

    return models


def load_model_state(filepath):
    """加载模型状态字典"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        return checkpoint.get('state_dict', checkpoint)
    except Exception as e:
        logging.warning(f"加载模型失败 {filepath}: {e}")
        return None


class PosteriorDetector:
    """独立的后验泄露检测器"""

    def __init__(self, archive_dir, args=None, gap_threshold=0.01):
        """
        Args:
            archive_dir: 存档目录
            args: 命令行参数（用于 KeyMatrixManager）
            gap_threshold: 相对差距阈值
        """
        self.archive_dir = archive_dir
        self.gap_threshold = gap_threshold
        self.args = args

        # 加载所有模型
        logging.info(f"加载存档: {archive_dir}")
        self.models = list_archived_models(archive_dir)
        self.customized_count = len(self.models['customized'])
        self.trained_count = len(self.models['trained'])
        logging.info(f"  定制模型: {self.customized_count} 个")
        logging.info(f"  训练后模型: {self.trained_count} 个")

        if self.customized_count == 0 and self.trained_count == 0:
            raise ValueError(f"存档目录为空或格式不正确: {archive_dir}")

    def compute_model_similarity(self, state_a, state_b):
        """
        计算两个模型状态的相似度
        使用全连接层和卷积层的权重计算余弦相似度

        Args:
            state_a: 模型A的state_dict
            state_b: 模型B的state_dict

        Returns:
            similarity: 余弦相似度
        """
        try:
            vectors_a = []
            vectors_b = []

            # 收集所有权重参数
            for key in state_a:
                if 'weight' not in key.lower():
                    continue
                if key not in state_b:
                    continue

                # 跳过bias和norm层
                if any(skip in key.lower() for skip in ['bias', 'bn', 'batchnorm', 'norm', 'running']):
                    continue

                try:
                    a_flat = state_a[key].view(-1).cpu().numpy()
                    b_flat = state_b[key].view(-1).cpu().numpy()

                    # 跳过全零的参数（可能是mask）
                    if np.all(a_flat == 0) or np.all(b_flat == 0):
                        continue

                    # 只取前100个元素以加速计算
                    max_len = min(100, len(a_flat), len(b_flat))
                    vectors_a.extend(a_flat[:max_len].tolist())
                    vectors_b.extend(b_flat[:max_len].tolist())

                except Exception:
                    continue

            if len(vectors_a) == 0:
                return 0.0

            vec_a = np.array(vectors_a)
            vec_b = np.array(vectors_b)

            # 计算余弦相似度
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0

            similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            return similarity

        except Exception as e:
            logging.warning(f"计算相似度失败: {e}")
            return 0.0

    def detect_single(self, suspected_round, suspected_client, model_type='trained'):
        """
        模拟泄漏检测：将可疑模型（训练后模型）的水印区域与存档中所有模型的水印区域对比，
        找出水印相似度最高的那个（用于溯源泄露者身份）。

        Args:
            suspected_round: 可疑模型的真实轮次（用于评估准确率）
            suspected_client: 可疑模型的真实客户端（用于评估准确率）
            model_type: 可疑模型的类型（目前主要用 'trained'）

        Returns:
            result: {
                'suspected': (round, client),       # 真实身份
                'predicted': (round, client) | None, # Top-1 预测的来源
                'top_matches': [(round, client, similarity), ...],
                'is_correct': bool,                  # Top-1 是否就是真实身份
                'top1_similarity': float,
                'top2_similarity': float,
                'similarity_gap': float,
                'relative_gap': float
            }
        """
        # 加载可疑模型（训练后模型）
        trained_archive = self.models.get('trained', {})
        key = (suspected_round, suspected_client)

        if key not in trained_archive:
            logging.warning(f"可疑模型不存在: 轮次 {suspected_round}, 客户端 {suspected_client}")
            return None

        # 加载可疑模型（训练后模型）
        suspected_filepath = trained_archive[key]
        suspected_state = load_model_state(suspected_filepath)
        if suspected_state is None:
            return None

        # 提取可疑模型的水印区域向量（使用其真实客户端的水印位置）
        suspected_wm = self._extract_watermark_params(suspected_state, suspected_client)
        if len(suspected_wm) == 0:
            logging.warning(f"无法提取可疑模型的水印区域")
            return None

        # 与存档中所有模型的定制模型水印区域对比
        rankings = []
        customized_archive = self.models.get('customized', {})

        for (r, c), filepath in customized_archive.items():
            other_state = load_model_state(filepath)
            if other_state is None:
                continue

            # 提取存档模型的水印区域
            other_wm = self._extract_watermark_params(other_state, c)
            if len(other_wm) == 0:
                continue

            # 计算余弦相似度（与 main.py 一致：用 min_len 截断）
            min_len = min(len(suspected_wm), len(other_wm))
            if min_len == 0:
                continue
            similarity = self._cosine_similarity(
                suspected_wm[:min_len],
                other_wm[:min_len]
            )
            rankings.append((r, c, similarity))

        # 按相似度降序排序
        rankings.sort(key=lambda x: x[2], reverse=True)

        if len(rankings) >= 2:
            top1 = rankings[0]
            top2 = rankings[1]
            similarity_gap = top1[2] - top2[2]
            relative_gap = similarity_gap / max(top2[2], 1e-8)
        elif len(rankings) == 1:
            top1 = rankings[0]
            top2 = None
            similarity_gap = top1[2]
            relative_gap = float('inf')
        else:
            top1 = None
            top2 = None
            similarity_gap = 0.0
            relative_gap = 0.0

        predicted = (top1[0], top1[1]) if top1 else None
        is_correct_round = (predicted == key) if predicted else False
        is_correct_client = (predicted[1] == key[1]) if predicted else False

        return {
            'suspected': key,
            'predicted': predicted,
            'model_type': model_type,
            'top_matches': rankings[:20],
            'all_count': len(rankings),
            'is_correct_round': is_correct_round,
            'is_correct_client': is_correct_client,
            'top1_similarity': top1[2] if top1 else 0.0,
            'top2_similarity': top2[2] if top2 else 0.0,
            'similarity_gap': similarity_gap,
            'relative_gap': relative_gap
        }

    def _build_param_offset_map(self, model_params):
        """
        构建参数全局偏移映射（与 main.py 中完全一致）
        只对卷积层权重进行计数
        """
        offset_map = {}
        param_order = []
        current_offset = 0

        for name, param in model_params.items():
            is_conv_weight = (
                'conv' in name.lower() and 'weight' in name.lower()
            ) or (
                'downsample.0.weight' in name.lower()
            )

            if is_conv_weight and isinstance(param, torch.Tensor) and len(param.shape) == 4:
                offset_map[name] = current_offset
                param_order.append(name)
                current_offset += param.numel()

        return offset_map, param_order

    def _extract_watermark_params(self, model_state, client_id):
        """
        提取指定客户端的水印参数（与 main.py._extract_single_client_watermark_params 一致）
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 构建参数偏移映射
            offset_map, param_order = self._build_param_offset_map(model_state)

            # 获取该客户端的水印位置
            positions = key_manager.load_positions(client_id)

            watermark_values = []
            for param_name, global_idx in positions:
                local_idx = None
                actual_param_name = None

                # 检查是否可以直接使用
                if param_name in model_state:
                    param_flat = model_state[param_name].view(-1)
                    if global_idx < param_flat.numel():
                        local_idx = global_idx
                        actual_param_name = param_name

                # 全局索引转换
                if actual_param_name is None and param_name in offset_map:
                    param_offset = offset_map[param_name]
                    param_flat = model_state[param_name].view(-1)
                    if param_offset <= global_idx < param_offset + param_flat.numel():
                        local_idx = global_idx - param_offset
                        actual_param_name = param_name

                # 遍历查找
                if actual_param_name is None:
                    for name, offset in offset_map.items():
                        param_size = model_state[name].numel()
                        if offset <= global_idx < offset + param_size:
                            actual_param_name = name
                            local_idx = global_idx - offset
                            break

                if actual_param_name and local_idx is not None:
                    if actual_param_name in model_state:
                        param_flat = model_state[actual_param_name].view(-1)
                        watermark_values.append(param_flat[local_idx].item())

            return np.array(watermark_values) if watermark_values else np.array([])

        except Exception as e:
            logging.warning(f"Failed to extract watermark params for client {client_id}: {e}")
            return np.array([])

    def _cosine_similarity(self, vec_a, vec_b):
        """计算两个向量的余弦相似度（自动处理不同长度）"""
        if vec_a is None or vec_b is None or len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0

        # 如果长度不同，取较短的长度（只比较共同部分）
        common_len = min(len(vec_a), len(vec_b))
        if common_len < len(vec_a) or common_len < len(vec_b):
            vec_a = vec_a[:common_len]
            vec_b = vec_b[:common_len]

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def full_coverage_detection(self, model_type='trained', sample=None):
        """
        后验检测全覆盖：对每个模型都找出它的"最相似来源"

        Args:
            model_type: 模型类型
            sample: 采样数量（None=全部，int=随机采样n个）

        Returns:
            report: 检测报告
        """
        archive = self.models.get(model_type, {})
        if not archive:
            logging.warning(f"没有 {model_type} 类型的存档模型")
            return None

        # 确定要检测的模型列表
        items = sorted(archive.keys())
        if sample is not None and sample < len(items):
            import random
            items = random.sample(items, sample)

        logging.info("=" * 80)
        logging.info(f"开始后验检测全覆盖: {model_type} 模型")
        logging.info(f"待检测数量: {len(items)}")
        logging.info("=" * 80)

        results = []
        correct_round_count = 0  # Top-1 完全正确（轮次+客户端）
        correct_client_count = 0  # Top-1 客户端正确

        for i, (round_num, client_id) in enumerate(items, 1):
            if i % 20 == 0 or i == 1:
                logging.info(f"进度: {i}/{len(items)}")

            result = self.detect_single(round_num, client_id, model_type)
            if result is None:
                continue

            results.append(result)

            if result.get('is_correct_round', False):
                correct_round_count += 1
            if result.get('is_correct_client', False):
                correct_client_count += 1

            # 报告 Top-1 预测和真实身份
            top_matches = result['top_matches']
            if top_matches:
                top1 = top_matches[0]
                top1_sim = top1[2]
                is_ok = "[OK]" if result.get('is_correct_round', False) else "[X]"
                predicted_str = f"轮次 {top1[0]}, 客户端 {top1[1]}"
                logging.info(f"  {is_ok} 真实=(轮次 {round_num:2d}, 客户端 {client_id}) "
                           f"-> Top-1={predicted_str} (相似度={top1_sim:.6f})")

        # 生成报告
        report = {
            'archive_dir': self.archive_dir,
            'model_type': model_type,
            'total_checked': len(results),
            'correct_round_count': correct_round_count,
            'correct_client_count': correct_client_count,
            'accuracy_round': correct_round_count / len(results) if results else 0.0,
            'accuracy_client': correct_client_count / len(results) if results else 0.0,
            'results': results
        }

        return report

    def save_report(self, report):
        """保存检测报告"""
        if report is None:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = report['model_type']
        report_dir = os.path.join(project_root, 'save', 'leak_detection')
        os.makedirs(report_dir, exist_ok=True)

        # 1. 保存文本报告
        report_file = os.path.join(
            report_dir,
            f'report_{model_type}_{timestamp}.txt'
        )

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("后验泄露检测全覆盖报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"存档目录: {report['archive_dir']}\n")
            f.write(f"模型类型: {report['model_type']}\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"总检测数: {report['total_checked']}\n")
            f.write(f"Top-1 完全正确（轮次+客户端）: {report['correct_round_count']} ({report['accuracy_round']*100:.2f}%)\n")
            f.write(f"Top-1 客户端正确: {report['correct_client_count']} ({report['accuracy_client']*100:.2f}%)\n\n")
            f.write("-" * 80 + "\n")
            f.write("Top-1 检测详情:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'真实身份':<22} {'Top-1 预测':<22} {'完全正确':<10} {'客户端正确':<10} {'Top-1 相似度':<15}\n")
            f.write("-" * 80 + "\n")

            for r in report['results']:
                suspected = f"轮次{r['suspected'][0]}, 客户端{r['suspected'][1]}"
                top_matches = r.get('top_matches', [])
                if top_matches:
                    top1 = top_matches[0]
                    top1_sim = top1[2]
                    round_ok = "[OK]" if r.get('is_correct_round', False) else "[X]"
                    client_ok = "[OK]" if r.get('is_correct_client', False) else "[X]"
                    source = f"轮次{top1[0]}, 客户端{top1[1]}"
                    f.write(f"{suspected:<22} {source:<22} {round_ok:<10} {client_ok:<10} {top1_sim:<15.6f}\n")
                else:
                    f.write(f"{suspected:<22} {'(无对比对象)':<22} {'-':<10} {'-':<10} {'-':<15}\n")

        # 2. 保存详细数据（torch格式）
        data_file = os.path.join(
            report_dir,
            f'data_{model_type}_{timestamp}.pt'
        )
        torch.save(report, data_file)

        logging.info(f"报告已保存:")
        logging.info(f"  文本: {report_file}")
        logging.info(f"  数据: {data_file}")

        return report_file

    def print_summary(self, report):
        """打印检测摘要"""
        if report is None:
            return

        print("\n" + "=" * 80)
        print("后验泄漏检测结果摘要（水印区域相似度对比）")
        print("=" * 80)
        print(f"存档目录: {report['archive_dir']}")
        print(f"模型类型: {report['model_type']}")
        print("-" * 80)
        print(f"总检测数: {report['total_checked']}")
        print(f"Top-1 完全正确（轮次+客户端）: {report['correct_round_count']} ({report['accuracy_round']*100:.2f}%)")
        print(f"Top-1 客户端正确: {report['correct_client_count']} ({report['accuracy_client']*100:.2f}%)")
        print("=" * 80)

        # 按错误类型分组
        wrong_cases = [r for r in report['results'] if not r.get('is_correct_round', False)]
        correct_cases = [r for r in report['results'] if r.get('is_correct_round', False)]

        if wrong_cases:
            print(f"\nTop-1 错误案例（前 10 个）:")
            print("-" * 80)
            for r in wrong_cases[:10]:
                suspected = r['suspected']
                predicted = r.get('predicted')
                top1_sim = r['top1_similarity']
                client_ok = "[客户端OK]" if r.get('is_correct_client', False) else "[客户端错]"
                pred_str = f"轮次 {predicted[0]}, 客户端 {predicted[1]}" if predicted else "(无预测)"
                print(f"  [X] 真实=(轮次 {suspected[0]}, 客户端 {suspected[1]}) -> "
                      f"Top-1={pred_str} {client_ok} (相似度={top1_sim:.6f})")
            if len(wrong_cases) > 10:
                print(f"  ... 还有 {len(wrong_cases) - 10} 个错误案例")

        # 统计 Top-2 是否有正确答案
        top2_correct = 0
        for r in wrong_cases:
            predicted = r.get('predicted')
            if predicted:
                # 检查 Top-2 及以后是否有正确答案
                for match in r.get('top_matches', [])[1:]:
                    if match[0] == r['suspected'][0] and match[1] == r['suspected'][1]:
                        top2_correct += 1
                        break

        if top2_correct > 0:
            print(f"\nTop-2 及以后包含正确答案的案例: {top2_correct}")

        print("=" * 80)


def main():
    # 提取我们自己的参数，剩下的传给 parser_args
    custom_arg_names = {'archive_dir', '--model-type', '--sample'}
    custom_argv = []
    remaining_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in custom_arg_names:
            custom_argv.append(arg)
            # 如果是 --model-type / --sample，下一个是 value
            if arg.startswith('--'):
                if i + 1 < len(sys.argv[1:]):
                    custom_argv.append(sys.argv[1:][i + 1])
                    skip_next = True
        elif arg.startswith('--model-type') or arg.startswith('--sample'):
            custom_argv.append(arg)
        else:
            remaining_argv.append(arg)

    # 用解析后的 argv 临时替换 sys.argv，调用 parser_args
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parser_args()
    sys.argv = old_argv

    # 解析自定义参数
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument(
        'archive_dir',
        nargs='?',
        default=None,
        help='模型存档目录路径（不指定则自动查找最新的）'
    )
    extra_parser.add_argument(
        '--model-type',
        type=str,
        default='trained',
        choices=['customized', 'trained', 'both'],
        help='检测的模型类型（默认trained=训练后模型）'
    )
    extra_parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='随机采样数量（默认检测全部）'
    )

    extra_args = extra_parser.parse_args(custom_argv)
    args.archive_dir = extra_args.archive_dir
    args.model_type = extra_args.model_type
    args.sample = extra_args.sample

    # 设置目录
    setup_directories()

    # 确定存档目录
    if args.archive_dir is None:
        archive_dir = find_latest_archive(os.path.join(project_root, 'save', 'model_archive'))
        if archive_dir is None:
            print("错误: 未找到模型存档目录!")
            print("请先运行 main.py 进行训练，或手动指定存档目录:")
            print("  python posterior_detection.py <archive_dir>")
            return
        print(f"自动选择最新存档: {archive_dir}")
    else:
        archive_dir = args.archive_dir
        if not os.path.isabs(archive_dir):
            archive_dir = os.path.join(project_root, archive_dir)

    if not os.path.exists(archive_dir):
        print(f"错误: 存档目录不存在: {archive_dir}")
        return

    # 创建检测器
    try:
        detector = PosteriorDetector(archive_dir, args=args)
    except ValueError as e:
        print(f"错误: {e}")
        return

    # 执行检测（默认检测 trained 模型，即训练后泄露场景）
    model_types = ['customized', 'trained'] if args.model_type == 'both' else [args.model_type]

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"开始检测: {model_type} 模型")
        print(f"{'='*80}\n")

        report = detector.full_coverage_detection(model_type, sample=args.sample)

        if report is not None:
            detector.print_summary(report)
            detector.save_report(report)

    print("\n检测完成！")


if __name__ == '__main__':
    main()
