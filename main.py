import copy
import os
import sys
import time
from datetime import datetime
import logging
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.globals import set_seed
from models.alexnet import AlexNet
from models.resnet import resnet18
from utils.args import parser_args
from utils.base import Experiment
from utils.dataset import get_data, DatasetSplit
from utils.trainer_private import TrainerPrivate, TesterPrivate
from utils.trainer_private_enhanced import TrainerPrivateEnhanced
from utils.autoencoder_finetuner import AutoencoderFinetuner, finetune_autoencoder_encoder
import pandas as pd

set_seed()

# 配置 logging
args = parser_args()
log_file_name = args.log_file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',  # 日期格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')  # 追加模式
    ]
)


class FederatedLearningOnChestMNIST(Experiment):
    def __init__(self, args):
        super().__init__(args)
        
        self.random_positions = None
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma
        self.key_matrix_dir = getattr(args, 'key_matrix_dir', './save/key_matrix')
        
        logging.info('--------------------------------Start--------------------------------------')
        logging.info(args)
        logging.info('==> Preparing data...')
        
        # 数据集配置
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels
            
        # 确保数据根目录存在以避免Windows权限/路径问题
        os.makedirs(self.data_root, exist_ok=True)

        self.train_set, self.test_set, self.dict_users = get_data(dataset_name=self.dataset,
                                                                  data_root=self.data_root,
                                                                  iid=self.iid,
                                                                  client_num=self.client_num,
                                                                  )
        logging.info('==> Training model...')
        self.logs = {'best_train_acc': -np.inf, 'best_train_loss': -np.inf,
                     'val_acc': [], 'val_loss': [],
                     'best_model_acc': -np.inf, 'best_model_loss': -np.inf,
                     'best_model_auc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     # 独立跟踪历史最高指标
                     'highest_acc_ever': -np.inf,     # 历史最高准确率（纯准确率）
                     'highest_auc_ever': -np.inf,     # 历史最高AUC（纯AUC）
                     'acc_when_highest_auc': -np.inf, # 达到历史最高AUC时的准确率
                     'auc_when_highest_acc': -np.inf, # 达到历史最高准确率时的AUC
                     }

        self.construct_model()
        self.w_t = copy.deepcopy(self.model.state_dict())

        # 根据参数选择水印模式
        self.random_positions = {}
        # 设置密钥矩阵目录
        self.args.key_matrix_dir = self.key_matrix_dir
        self.args.use_key_matrix = True
        
        # ========== 模型追踪功能初始化 ==========
        # 用于存储每轮下发给各客户端的"定制模型"
        self.customized_models = {}  # {client_id: model_state_dict}
        # 用于记录每次泄漏事件
        self.leakage_records = []  # [{round, leaked_client, model_params}]
        # 用于记录追踪结果
        self.trace_results = []  # [{round, actual_leaked, detected_leaked, similarity_rankings}]
        # 用于存储用于追踪的基准值（其他客户端水印区域的聚合结果）
        self.watermark_baseline = {}  # {client_id: aggregated_watermark_params}
        
        # 根据watermark_mode参数选择trainer
        if self.args.watermark_mode == 'enhanced':
            logging.info('==> 使用增强水印系统（密钥矩阵 + 自编码器）')
            self.trainer = TrainerPrivateEnhanced(self.model, self.device, self.dp, self.sigma, self.args)
            
            # 初始化自编码器微调器
            self.autoencoder_finetuner = AutoencoderFinetuner(self.device)
            logging.info('==> 自编码器微调器已初始化')
        else:
            logging.info('==> 使用普通水印系统')
            self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma, self.random_positions, self.args)
            self.autoencoder_finetuner = None
            
        self.tester = TesterPrivate(self.model, self.device, args=self.args)

    def construct_model(self):
        if self.model_name == 'resnet':
            model = resnet18(num_classes=self.num_classes, in_channels=self.in_channels, input_size=self.args.input_size)
        else:
            dropout_rate = getattr(self.args, 'dropout_rate', 0.5)
            model = AlexNet(self.in_channels, self.num_classes, input_size=self.args.input_size, dropout_rate=dropout_rate)
        self.model = model.to(self.device)

    def _cleanup_memory(self):
        """清理内存和GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _optimize_model_storage(self, model_state):
        """优化模型状态存储，减少内存占用"""
        # 将模型状态移到CPU，使用detach()避免梯度追踪
        optimized_state = {}
        for key, value in model_state.items():
            optimized_state[key] = value.detach().cpu()
        return optimized_state

    def training(self):
        start = time.time()
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)

        local_train_loader = []

        for i in range(self.client_num):
            local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]),
                                         batch_size=self.batch_size,
                                         shuffle=True, num_workers=0, pin_memory=False)
            local_train_loader.append(local_train_ldr)

        idxs_users = []

        # 决定模型选择依据：ChestMNIST 按 AUC，其他（如 CIFAR-10/100）按准确率
        dataset_key = (self.dataset or '').lower()
        select_by_auc = (dataset_key == 'chestmnist')

        # 早停配置
        use_early_stopping = self.args.patience > 0
        early_stop_counter = 0
        best_val_acc = -np.inf
        best_val_auc = -np.inf

        # 统计记录
        stats_rows = []

        for epoch in range(self.epochs): # 均匀采样，frac 默认为 1，即每轮中全体客户端参与训练
            # 均匀采样
            self.m = max(int(self.frac * self.client_num), 1)
            idxs_users = np.random.choice(range(self.client_num), self.m, replace=False)

            logging.info('Epoch: %d / %d' % (epoch + 1, self.epochs))
            
            # ========== 第一步：下发前生成定制模型 ==========
            # 基于上一轮聚合模型 w_t 生成每个客户端的独特模型
            self.customized_models = self._generate_all_customized_models(
                self.w_t, idxs_users, epoch
            )
            
            # ========== 第二步：下发前泄漏模拟 ==========
            # 只有启用泄漏追踪时才执行泄漏模拟
            leak_info = None
            if getattr(self.args, 'enable_leakage_tracking', True):
                leak_info = self._simulate_pre_training_leakage(epoch, idxs_users)

            # ========== 第三步：模型泄漏追踪 ==========
            if leak_info is not None:
                leaked_model_state = leak_info['model_params']
                actual_leaker = leak_info['leaked_client']

                # 执行追踪（基于下发的定制模型）
                similarity_rankings, detected_leaker = self._track_model_leakage(
                    leaked_model_state, idxs_users
                )

                # 记录追踪结果
                trace_result = {
                    'round': epoch + 1,
                    'actual_leaked_client': actual_leaker,
                    'detected_leaked_client': detected_leaker,
                    'is_correct': (actual_leaker == detected_leaker),
                    'similarity_rankings': similarity_rankings
                }
                self.trace_results.append(trace_result)

                # 输出追踪结果
                logging.info(f"[追踪结果] 轮次 {epoch + 1}")
                logging.info(f"  实际泄漏者: 客户端 {actual_leaker}")
                logging.info(f"  检测结果: 客户端 {detected_leaker}")
                logging.info(f"  追踪正确性: {'✓ 正确' if trace_result['is_correct'] else '✗ 错误'}")
                logging.info(f"  相似度排名:")
                for rank, (cid, sim) in enumerate(similarity_rankings[:5], 1):
                    marker = " <-- 检测" if cid == detected_leaker else ""
                    actual_marker = " (实际泄漏者)" if cid == actual_leaker else ""
                    logging.info(f"    {rank}. 客户端 {cid}: 相似度 = {sim:.6f}{marker}{actual_marker}")
            
            # ========== 第四步：本地训练（使用下发的定制模型） ==========
            local_ws, local_losses = [], []
            
            for i, idx in enumerate(tqdm(idxs_users, desc='Progress: %d / %d' % (epoch + 1, self.epochs))):
                # 使用下发给该客户端的定制模型作为起点
                self.model.load_state_dict(self.customized_models[idx])

                # 统一调用：始终传入 current_epoch/total_epochs；
                # 普通 Trainer 会通过 **kwargs 忽略
                local_w, local_loss, local_acc = self.trainer.local_update(
                    dataloader=local_train_loader[idx], 
                    local_ep=self.local_ep, 
                    lr=self.lr, 
                    client_id=idx,
                    current_epoch=epoch,
                    total_epochs=self.epochs
                )

                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)
                
                # 清理临时变量，释放内存
                del local_w, local_loss, local_acc

            # 计算参与训练的客户端的权重（相对于总数据集）
            client_weights = []
            for idx in idxs_users:
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[idx])) / len(self.train_set)
                client_weights.append(client_weight)

            # ========== 第五步：聚合更新全局模型 ==========
            watermark_baseline = self._fed_avg(local_ws, client_weights, idxs_users)

            # 加载聚合后的权重并确保模型处于正确状态
            self.model.load_state_dict(self.w_t)
            self.model.train()  # 确保模型处于训练状态
            
            # 清理优化器状态，避免梯度累积
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                self.trainer.optimizer.zero_grad()
            
            # 梯度统计（仅每10轮打印一次）
            if epoch >= 0 and hasattr(self.trainer, 'multi_loss') and (epoch + 1) % 10 == 0:
                try:
                    stats = self.trainer.get_gradient_stats()
                    if stats:
                        logging.info(f'轮次{epoch+1}梯度统计 - GM:{stats.get("prevGM", 0):.6f} GH:{stats.get("prevGH", 0):.6f} Ratio:{stats.get("prevRatio", 1):.6f}')
                except Exception as e:
                    pass  # 静默处理错误


            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                train_metrics = self.trainer.test(train_ldr)
                val_metrics = self.trainer.test(val_ldr)

                # (loss, acc_label, auc, acc_sample)
                loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean = train_metrics
                loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean = val_metrics

                self.logs['val_acc'].append(acc_val_label_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))

                # 更新历史最高值跟踪
                if self.logs['highest_acc_ever'] < acc_val_label_mean:
                    self.logs['highest_acc_ever'] = acc_val_label_mean
                    self.logs['auc_when_highest_acc'] = auc_val
                    
                if self.logs['highest_auc_ever'] < auc_val:
                    self.logs['highest_auc_ever'] = auc_val
                    self.logs['acc_when_highest_auc'] = acc_val_label_mean

                # 模型选择标准：ChestMNIST 按 AUC；否则按准确率
                if select_by_auc:
                    if self.logs['best_model_auc'] < auc_val:
                        self.logs['best_model_acc'] = acc_val_label_mean
                        self.logs['best_model_loss'] = loss_val_mean
                        self.logs['best_model_auc'] = auc_val
                        # 优化模型存储，减少内存占用
                        optimized_state = self._optimize_model_storage(self.model.state_dict())
                        self.logs['best_model'] = [optimized_state]
                        logging.info(f'🌟 最佳模型已保存! AUC↑{auc_val:.4f}')
                else:
                    if self.logs['best_model_acc'] < acc_val_label_mean:
                        self.logs['best_model_acc'] = acc_val_label_mean
                        self.logs['best_model_loss'] = loss_val_mean
                        self.logs['best_model_auc'] = auc_val
                        optimized_state = self._optimize_model_storage(self.model.state_dict())
                        self.logs['best_model'] = [optimized_state]
                        logging.info(f'🌟 最佳模型已保存! ACC↑{acc_val_label_mean:.4f}')

                if self.logs['best_train_acc'] < acc_train_label_mean:
                    self.logs['best_train_acc'] = acc_train_label_mean
                    self.logs['best_train_loss'] = loss_train_mean

                # 合并训练和验证指标到一行
                logging.info(
                    f"轮次{epoch+1} | Train Loss:{loss_train_mean:.4f} Acc:{acc_train_label_mean:.4f} AUC:{auc_train:.4f} | "
                    f"Val Loss:{loss_val_mean:.4f} Acc:{acc_val_label_mean:.4f} AUC:{auc_val:.4f} | "
                    f"Best Acc:{self.logs['highest_acc_ever']:.4f} Best AUC:{self.logs['highest_auc_ever']:.4f}")
                
                # MultiLoss统计信息（仅每10轮打印）
                if hasattr(self.trainer, 'multi_loss') and (epoch + 1) % 10 == 0:
                    stats = self.trainer.multi_loss.get_stats()
                    logging.info(f"MultiLoss - GM:{stats['prevGM']:.6f} GH:{stats['prevGH']:.6f} Ratio:{stats['prevRatio']:.6f}")
                
                # 记录本轮统计数据
                stats_row = {
                    'round': epoch + 1,
                    'lr': self.lr,
                    'train_loss': float(loss_train_mean),
                    'val_loss': float(loss_val_mean),
                    'train_acc_label': float(acc_train_label_mean),
                    'train_auc': float(auc_train),
                    'val_acc_label': float(acc_val_label_mean),
                    'val_auc': float(auc_val),
                    'best_val_acc_so_far': float(self.logs['highest_acc_ever']),
                    'best_val_auc_so_far': float(self.logs['highest_auc_ever']),
                    'train_acc_sample': float(acc_train_sample_mean),
                    'val_acc_sample': float(acc_val_sample_mean),
                }
                
                # 早停检查：patience > 0 时启用
                if use_early_stopping:
                    if select_by_auc:
                        if auc_val > best_val_auc:
                            best_val_auc = auc_val
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= self.args.patience:
                                logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val AUC: {best_val_auc:.4f}')
                                break
                    else:
                        if acc_val_label_mean > best_val_acc:
                            best_val_acc = acc_val_label_mean
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= self.args.patience:
                                logging.info(f'Early stopping triggered at epoch {epoch + 1}. Best Val ACC: {best_val_acc:.4f}')
                                break
                
                # 每轮训练后清理内存
                self._cleanup_memory()
                
                # 清理临时变量
                del train_metrics, val_metrics
                del loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean
                del loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean
                
                # 添加自编码器微调统计信息
                if self.autoencoder_finetuner is not None and hasattr(self.trainer, 'autoencoder'):
                    try:
                        # 获取当前自编码器性能
                        current_performance = self.autoencoder_finetuner.evaluate_encoder_performance(
                            self.trainer.autoencoder, 
                            test_samples=500  # 使用较少样本进行快速评估
                        )
                        stats_row['autoencoder_performance'] = float(current_performance)
                        # 自编码器性能（仅每20轮显示）
                        if (epoch + 1) % 20 == 0:
                            logging.info(f'📊 自编码器性能: {current_performance:.6f}')
                    except Exception as e:
                        stats_row['autoencoder_performance'] = float('inf')
                        logging.warning(f'⚠️ 无法评估自编码器性能: {e}')
                else:
                    stats_row['autoencoder_performance'] = None
                
                # 添加增强水印系统统计信息
                if hasattr(self.trainer, 'multi_loss'):
                    multi_loss_stats = self.trainer.multi_loss.get_stats()
                    stats_row.update({
                        'prevGM': float(multi_loss_stats['prevGM']),
                        'prevGH': float(multi_loss_stats['prevGH']),
                        'prevRatio': float(multi_loss_stats['prevRatio']),
                        'current_grad_M': float(multi_loss_stats['current_grad_M']),
                        'current_grad_H': float(multi_loss_stats['current_grad_H']),
                        'current_var_M': float(multi_loss_stats['current_var_M']),
                        'current_var_H': float(multi_loss_stats['current_var_H']),
                    })
                
                stats_rows.append(stats_row)

        logging.info('='*60 + ' 训练结果 ' + '='*60)

        # ========== 模型泄漏追踪结果汇总 ==========
        if len(self.trace_results) > 0:
            logging.info('='*60 + ' 模型泄漏追踪结果 ' + '='*60)
            logging.info(f'总共发生 {len(self.trace_results)} 次模型泄漏事件')
            correct_count = sum(1 for r in self.trace_results if r['is_correct'])
            accuracy = correct_count / len(self.trace_results) * 100
            logging.info(f'追踪准确率: {accuracy:.2f}% ({correct_count}/{len(self.trace_results)})')

            logging.info('详细追踪记录:')
            for trace in self.trace_results:
                status = '✓' if trace['is_correct'] else '✗'
                logging.info(
                    f"  轮次 {trace['round']}: 实际={trace['actual_leaked_client']}, "
                    f"检测={trace['detected_leaked_client']} [{status}]"
                )

            # 保存追踪结果到文件（保存到 save/trace 目录）
            try:
                import json
                # 创建 trace 专用目录
                trace_dir = os.path.join('save', 'trace')
                os.makedirs(trace_dir, exist_ok=True)
                trace_log_path = os.path.join(
                    trace_dir,
                    f'trace_results_{self.model_name}_{self.dataset}_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
                )
                # 将trace_results转换为可序列化的格式
                serializable_trace = []
                for trace in self.trace_results:
                    serializable_trace.append({
                        'round': int(trace['round']),
                        'actual_leaked_client': int(trace['actual_leaked_client']),
                        'detected_leaked_client': int(trace['detected_leaked_client']),
                        'is_correct': bool(trace['is_correct']),
                        'similarity_rankings': [
                            [int(cid), float(sim)] for cid, sim in trace['similarity_rankings']
                        ]
                    })

                def convert_to_serializable(obj):
                    """处理numpy类型以便JSON序列化"""
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    elif isinstance(obj, (np.floating,)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj

                with open(trace_log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_leakages': int(len(self.trace_results)),
                        'correct_detections': int(correct_count),
                        'accuracy': float(accuracy),
                        'details': serializable_trace
                    }, f, indent=2, ensure_ascii=False, default=convert_to_serializable)
                logging.info(f'追踪结果已保存到: {trace_log_path}')
            except Exception as e:
                logging.warning(f'保存追踪结果失败: {e}')
        else:
            logging.info('本轮训练未发生模型泄漏事件')

        logging.info(
            f'最佳模型 | Loss:{self.logs["best_model_loss"]:.4f} Acc:{self.logs["best_model_acc"]:.4f} AUC:{self.logs["best_model_auc"]:.4f}')
        logging.info(
            f'历史最高 | Acc:{self.logs["highest_acc_ever"]:.4f}(AUC:{self.logs["auc_when_highest_acc"]:.4f}) | '
            f'AUC:{self.logs["highest_auc_ever"]:.4f}(Acc:{self.logs["acc_when_highest_auc"]:.4f})')
        end = time.time()
        logging.info(f'训练耗时: {(end - start) / 60:.1f} 分钟')
        
        # 最终内存清理
        self._cleanup_memory()
        logging.info('🧹 训练完成，已清理内存缓存')

        # 导出Excel
        try:
            os.makedirs(self.args.save_excel_dir, exist_ok=True)
            # 基础列 + 增强水印系统统计列 + 自编码器性能列
            columns = ['round', 'lr', 'train_loss', 'val_loss', 'train_acc_label', 'train_auc', 
                     'val_acc_label', 'val_auc', 'best_val_acc_so_far', 'best_val_auc_so_far', 
                     'train_acc_sample', 'val_acc_sample', 'autoencoder_performance',
                     'prevGM', 'prevGH', 'prevRatio', 
                     'current_grad_M', 'current_grad_H', 'current_var_M', 'current_var_H']
            df = pd.DataFrame(stats_rows, columns=columns)
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_path = f'{self.args.save_excel_dir}/metrics_{self.model_name}_{self.dataset}_{now}.xlsx'
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logging.info(f'Excel metrics saved to: {excel_path}')
        except Exception as e:
            logging.warning(f'Failed to export Excel metrics: {e}')

        # 返回用于模型选择的指标与其名称，便于上层命名保存
        best_metric_value = self.logs['best_model_auc'] if select_by_auc else self.logs['best_model_acc']
        best_metric_name = 'auc' if select_by_auc else 'acc'
        return self.logs, best_metric_value, best_metric_name


    def _fed_avg(self, local_ws, client_weights, idxs_users):
        """联邦平均算法，FedAvg with exclusive watermark aggregation"""
        # 计算参与训练的客户端权重总和
        total_weight = sum(client_weights)

        # 归一化权重，确保权重和为1
        normalized_weights = [w / total_weight for w in client_weights]

        # 验证权重和是否为1
        weight_sum = sum(normalized_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logging.warning(f"Weight sum is {weight_sum:.6f}, not 1.0. Normalizing...")
            normalized_weights = [w / weight_sum for w in normalized_weights]

        # 初始化平均权重
        w_avg = {}
        for k in local_ws[0].keys():
            w_avg[k] = local_ws[0][k].clone() * normalized_weights[0]

        # 累加其他客户端的权重
        for i in range(1, len(local_ws)):
            for k in w_avg.keys():
                w_avg[k] += local_ws[i][k] * normalized_weights[i]

        # 水印聚合：使用密钥矩阵的独占式聚合，返回其他客户端水印区域的聚合结果
        watermark_baseline = self._watermark_aggregation(local_ws, idxs_users, w_avg)

        # 更新全局模型权重
        for k in w_avg.keys():
            self.w_t[k] = w_avg[k]

        return watermark_baseline

    def _get_watermark_positions_for_client(self, client_id):
        """
        获取指定客户端的水印位置集合

        Args:
            client_id: 客户端ID

        Returns:
            watermark_positions: 集合 {(param_name, param_idx), ...}
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)
            positions = key_manager.load_positions(client_id)
            return set(positions)
        except Exception as e:
            logging.warning(f"Failed to load watermark positions for client {client_id}: {e}")
            return set()

    def _generate_all_customized_models(self, global_model, idxs_users, epoch):
        """
        为所有参与本轮训练的客户端生成定制模型（下发前生成）

        每个客户端的定制模型包含：
        - 该客户端的独特水印区域：嵌入客户端专属的"指纹值"
        - 其他区域：使用全局模型的参数

        Args:
            global_model: 全局模型参数 (w_t)
            idxs_users: 参与训练的客户端ID列表
            epoch: 当前轮次

        Returns:
            customized_models: 字典 {client_id: model_state_dict}
        """
        customized_models = {}
        
        # 为每个客户端生成独特的基准偏移量
        torch.manual_seed(42 + epoch)  # 保证每轮的可重复性
        base_offset = torch.randn(1).item() * 0.01  # 基础偏移量
        
        for client_id in idxs_users:
            customized_models[client_id] = self._generate_customized_model(
                client_id, global_model, base_offset, epoch
            )
        return customized_models

    def _generate_customized_model(self, client_id, global_model, base_offset, epoch):
        """
        为指定客户端生成"定制模型"

        定制模型规则（泄漏前版本）：
        - 客户端k的水印区域：嵌入客户端k专属的指纹值（基于 client_id 和轮次生成）
        - 非水印区域：使用全局模型的参数

        Args:
            client_id: 目标客户端ID
            global_model: 全局模型参数
            base_offset: 基础偏移量
            epoch: 当前轮次

        Returns:
            customized_model: 定制模型参数字典
        """
        customized_model = {}

        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 构建参数偏移映射
            offset_map, param_order = self._build_param_offset_map(global_model)

            # 获取当前客户端的水印位置
            client_positions = self._get_watermark_positions_for_client(client_id)

            for param_name in global_model.keys():
                param_tensor = global_model[param_name].clone()
                param_flat = param_tensor.view(-1)

                # 在该客户端的水印位置嵌入独特指纹
                for param_name_w, global_idx in client_positions:
                    if param_name_w != param_name:
                        continue
                    
                    local_idx = None
                    
                    # 直接使用局部索引
                    if global_idx < param_flat.numel():
                        local_idx = global_idx
                    # 全局索引转换
                    elif param_name in offset_map:
                        param_offset = offset_map[param_name]
                        if param_offset <= global_idx < param_offset + param_flat.numel():
                            local_idx = global_idx - param_offset
                    
                    if local_idx is not None:
                        # 生成客户端专属的指纹值
                        torch.manual_seed(1000 + client_id + epoch * 100)
                        fingerprint = base_offset + (torch.rand(1).item() - 0.5) * 0.02
                        
                        # 将指纹值添加到原有参数上（产生独特但微小的差异）
                        original_value = param_flat[local_idx].item()
                        param_flat[local_idx] = original_value + fingerprint

                customized_model[param_name] = param_tensor

        except Exception as e:
            logging.warning(f"Failed to generate customized model for client {client_id}: {e}")
            return global_model.copy()

        return customized_model

    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量之间的余弦相似度

        Args:
            vec1, vec2: torch.Tensor

        Returns:
            similarity: float
        """
        vec1_flat = vec1.view(-1).float()
        vec2_flat = vec2.view(-1).float()

        dot_product = torch.dot(vec1_flat, vec2_flat)
        norm1 = torch.norm(vec1_flat)
        norm2 = torch.norm(vec2_flat)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return (dot_product / (norm1 * norm2)).item()

    def _extract_watermark_params_for_tracking(self, model_state, exclude_client_id):
        """
        提取用于追踪的参数（水印区域，但排除指定客户端的水印）

        Args:
            model_state: 模型参数字典
            exclude_client_id: 排除的客户端ID（通常是泄漏模型的那个客户端）

        Returns:
            concatenated_params: 拼接后的参数向量
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 构建参数偏移映射
            offset_map, param_order = self._build_param_offset_map(model_state)

            all_watermark_indices = []  # [(param_name, global_idx), ...]

            # 收集所有客户端的水印位置（除非指定了排除的客户端）
            for client_id in range(self.client_num):
                if exclude_client_id is not None and client_id == exclude_client_id:
                    continue
                positions = key_manager.load_positions(client_id)
                all_watermark_indices.extend(positions)

            # 去重
            all_watermark_indices = list(set(all_watermark_indices))

            # 提取对应的参数值
            watermark_values = []
            for param_name, global_idx in all_watermark_indices:
                local_idx = None
                actual_param_name = None

                # 首先检查是否可以直接使用
                if param_name in model_state:
                    param_size = model_state[param_name].numel()
                    if global_idx < param_size:
                        actual_param_name = param_name
                        local_idx = global_idx

                # 如果是全局索引，需要转换
                if actual_param_name is None and param_name in offset_map:
                    param_offset = offset_map[param_name]
                    param_size = model_state[param_name].numel()
                    if param_offset <= global_idx < param_offset + param_size:
                        actual_param_name = param_name
                        local_idx = global_idx - param_offset

                # 如果还是找不到，遍历找正确的参数
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

            return torch.tensor(watermark_values) if watermark_values else torch.tensor([])

        except Exception as e:
            logging.warning(f"Failed to extract watermark params for tracking: {e}")
            return torch.tensor([])

    def _track_model_leakage(self, leaked_model_state, idxs_users):
        """
        追踪模型泄漏源

        原理：
        - 每个客户端的定制模型在其水印区域嵌入了客户端专属的指纹值
        - 如果客户端k的定制模型泄漏，我们可以从泄漏模型中提取其水印区域
        - 比较泄漏模型与每个客户端的定制模型的水印区域相似度
        - 客户端k的相似度最高（因为泄漏的就是k的模型）

        Args:
            leaked_model_state: 泄漏的模型参数字典
            idxs_users: 参与训练的客户端ID列表

        Returns:
            similarity_rankings: 相似度排名列表 [(client_id, similarity), ...]
            detected_leaker: 检测到的泄漏者ID
        """
        try:
            # 计算每个客户端的定制模型与泄漏模型的相似度
            similarity_rankings = []

            for client_id in idxs_users:
                # 获取该客户端的定制模型
                if client_id not in self.customized_models:
                    continue

                customized_model = self.customized_models[client_id]

                # 提取该客户端的水印区域参数
                leaked_watermark_params = self._extract_watermark_params_for_tracking(
                    leaked_model_state, exclude_client_id=None  # 不排除任何客户端
                )
                client_watermark_params = self._extract_watermark_params_for_tracking(
                    customized_model, exclude_client_id=None
                )

                if len(leaked_watermark_params) > 0 and len(client_watermark_params) > 0:
                    # 确保长度一致
                    min_len = min(len(leaked_watermark_params), len(client_watermark_params))
                    similarity = self._cosine_similarity(
                        leaked_watermark_params[:min_len],
                        client_watermark_params[:min_len]
                    )
                else:
                    similarity = 0.0

                similarity_rankings.append((client_id, similarity))

            # 按相似度降序排序
            similarity_rankings.sort(key=lambda x: x[1], reverse=True)

            # 检测到的泄漏者就是相似度最高的客户端
            detected_leaker = similarity_rankings[0][0] if similarity_rankings else None

            return similarity_rankings, detected_leaker

        except Exception as e:
            logging.warning(f"Failed to track model leakage: {e}")
            return [], None

    def _simulate_pre_training_leakage(self, epoch, idxs_users):
        """
        模拟模型泄漏事件（发生在本地训练之前）

        根据配置的泄漏间隔从参与训练的客户端中随机选择一个泄漏其下发的定制模型

        Args:
            epoch: 当前轮次
            idxs_users: 参与训练的客户端ID列表

        Returns:
            leak_info: 泄漏信息字典，如果本轮不泄漏则返回None
        """
        leak_interval = self.args.leak_interval

        # 如果泄漏间隔为0或负数，则禁用泄漏模拟
        if leak_interval <= 0:
            return None

        if (epoch + 1) % leak_interval != 0:
            return None

        # 从参与训练的客户端中随机选择一个
        leaked_client = int(np.random.choice(idxs_users, 1)[0])

        # 获取该客户端下发的定制模型（此时尚未进行本地训练）
        if leaked_client not in self.customized_models:
            logging.warning(f"定制模型不存在于客户端 {leaked_client}，跳过泄漏模拟")
            return None

        leaked_model = self.customized_models[leaked_client]

        # 深拷贝以保存泄漏的模型
        leaked_model_copy = {}
        for k, v in leaked_model.items():
            if isinstance(v, torch.Tensor):
                leaked_model_copy[k] = v.clone().detach().cpu()
            else:
                leaked_model_copy[k] = v

        leak_info = {
            'round': epoch + 1,
            'leaked_client': leaked_client,
            'model_params': leaked_model_copy
        }

        # 记录泄漏事件
        self.leakage_records.append(leak_info)

        logging.info(f"=" * 60)
        logging.info(f"[模型泄漏模拟] 第 {epoch + 1} 轮 - 客户端 {leaked_client} 的定制模型在下发时泄漏")
        logging.info(f"=" * 60)

        return leak_info

    def _build_param_offset_map(self, model_params):
        """
        构建参数全局偏移映射（用于将全局索引转换为局部索引）

        Args:
            model_params: 模型参数字典

        Returns:
            offset_map: 每个参数的全局起始偏移量字典
            param_order: 参数名列表（按顺序）
        """
        offset_map = {}
        param_order = []
        current_offset = 0

        for name, param in model_params.items():
            # 检查是否为卷积层参数
            is_conv_weight = (
                'conv' in name.lower() and 'weight' in name.lower()
            ) or (
                'downsample.0.weight' in name.lower()
            )

            if is_conv_weight and len(param.shape) == 4:
                offset_map[name] = current_offset
                param_order.append(name)
                current_offset += param.numel()

        return offset_map, param_order

    def _watermark_aggregation(self, local_ws, idxs_users, w_avg):
        """
        水印聚合：使用密钥矩阵的独占式聚合

        Args:
            local_ws: 本地模型权重列表
            idxs_users: 参与训练的客户端ID列表
            w_avg: 平均聚合后的权重字典

        Returns:
            watermark_baseline: 其他客户端水印区域的聚合结果（用于追踪）
        """
        watermark_baseline = {}
        try:
            from utils.key_matrix_utils import KeyMatrixManager

            # 加载密钥矩阵管理器
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 构建参数偏移映射
            offset_map, param_order = self._build_param_offset_map(local_ws[0])

            # 初始化水印区域的聚合结果
            watermark_param_names = set()
            for i, client_id in enumerate(idxs_users):
                positions = key_manager.load_positions(client_id)
                for param_name, _ in positions:
                    watermark_param_names.add(param_name)

            # 为每个参数初始化聚合结果
            for param_name in watermark_param_names:
                if param_name in local_ws[0]:
                    watermark_baseline[param_name] = torch.zeros_like(local_ws[0][param_name])

            # 统计参与聚合的客户端数量
            client_count = len(idxs_users)

            # 对每个参与客户端的水印区域进行聚合
            for i, client_id in enumerate(idxs_users):
                try:
                    positions = key_manager.load_positions(client_id)
                    for param_name, global_idx in positions:
                        # 尝试将全局索引转换为局部索引
                        local_idx = None
                        actual_param_name = None

                        # 首先检查是否可以直接使用（局部索引）
                        if param_name in local_ws[i]:
                            param_size = local_ws[i][param_name].numel()
                            if global_idx < param_size:
                                actual_param_name = param_name
                                local_idx = global_idx

                        # 如果是全局索引，需要转换
                        if actual_param_name is None and param_name in offset_map:
                            param_offset = offset_map[param_name]
                            param_size = local_ws[i][param_name].numel()
                            if param_offset <= global_idx < param_offset + param_size:
                                actual_param_name = param_name
                                local_idx = global_idx - param_offset

                        # 如果还是找不到，遍历找正确的参数
                        if actual_param_name is None:
                            for name, offset in offset_map.items():
                                param_size = local_ws[i][name].numel()
                                if offset <= global_idx < offset + param_size:
                                    actual_param_name = name
                                    local_idx = global_idx - offset
                                    break

                        if actual_param_name and local_idx is not None:
                            if actual_param_name in local_ws[i] and actual_param_name in watermark_baseline:
                                local_param = local_ws[i][actual_param_name].view(-1)
                                watermark_baseline[actual_param_name].view(-1)[local_idx] += local_param[local_idx].item() / client_count

                except Exception as e:
                    logging.warning(f"Failed to aggregate watermark for client {client_id}: {e}")
                    continue

            # 对每个客户端的水印位置进行独占式聚合
            for i, client_id in enumerate(idxs_users):
                try:
                    positions = key_manager.load_positions(client_id)

                    for param_name, global_idx in positions:
                        # 尝试将全局索引转换为局部索引
                        local_idx = None
                        actual_param_name = None

                        # 首先检查是否可以直接使用
                        if param_name in local_ws[i]:
                            param_size = local_ws[i][param_name].numel()
                            if global_idx < param_size:
                                actual_param_name = param_name
                                local_idx = global_idx

                        # 如果是全局索引，需要转换
                        if actual_param_name is None and param_name in offset_map:
                            param_offset = offset_map[param_name]
                            param_size = local_ws[i][param_name].numel()
                            if param_offset <= global_idx < param_offset + param_size:
                                actual_param_name = param_name
                                local_idx = global_idx - param_offset

                        # 如果还是找不到，遍历找正确的参数
                        if actual_param_name is None:
                            for name, offset in offset_map.items():
                                param_size = local_ws[i][name].numel()
                                if offset <= global_idx < offset + param_size:
                                    actual_param_name = name
                                    local_idx = global_idx - offset
                                    break

                        if actual_param_name and local_idx is not None:
                            if actual_param_name in local_ws[i] and actual_param_name in w_avg:
                                local_param = local_ws[i][actual_param_name]
                                avg_param = w_avg[actual_param_name]

                                if local_param.shape == avg_param.shape:
                                    param_flat = avg_param.view(-1)
                                    local_flat = local_param.view(-1)
                                    param_flat[local_idx] = local_flat[local_idx]

                except Exception as e:
                    logging.warning(f"Failed to apply watermark aggregation for client {client_id}: {e}")
                    continue

        except Exception as e:
            logging.warning(f"Failed to load key matrix manager for watermark aggregation: {e}")

        return watermark_baseline

def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'client_num': args.client_num,
                'console_log': os.path.basename(log_file_name),
            }
            }
    fl = FederatedLearningOnChestMNIST(args)
    logg, best_metric_value, best_metric_name = fl.training()
    logs['net_info'] = logg
    # 兼容：总是记录 test_auc；当指标为 acc 时也额外记录
    logs['test_auc'] = {'value': logg.get('best_model_auc', best_metric_value if best_metric_name == 'auc' else 0.0)}
    if best_metric_name == 'acc':
        logs['test_acc'] = {'value': best_metric_value}
    logs['bp_local'] = {'value': False}

    save_dir = os.path.join(args.save_model_dir, args.model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    # 构建文件名
    enhanced = "_enhanced" if args.watermark_mode == 'enhanced' else ""

    # 根据选择指标命名文件
    watermark_suffix = f"wm_{args.watermark_mode}" if hasattr(args, 'watermark_mode') and args.watermark_mode else "wm_basic"
    file_name = '{}_Dp_{}_iid_{}_{}_ep_{}_le_{}_cn_{}_fra_{:.4f}_{}_{{:.4f}}{}.pkl'.format(
        formatted_now, args.sigma, args.iid, watermark_suffix,
        args.epochs, args.local_ep, args.client_num, args.frac, best_metric_name, enhanced
    )
    file_name = file_name.format(best_metric_value)
    torch.save(logs, os.path.join(save_dir, file_name))
    logging.info(f"训练日志已保存: {file_name}")
    logging.info('-------------------------------Finish--------------------------------------')

    return

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)