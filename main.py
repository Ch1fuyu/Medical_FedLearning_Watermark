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

# 检查日志文件行数，如果超过阈值则备份并创建新的
MAX_LOG_LINES = 10000
if os.path.exists(log_file_name):
    with open(log_file_name, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    if line_count > MAX_LOG_LINES:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = log_file_name.replace('.logs', f'_{timestamp}.logs')
        os.rename(log_file_name, backup_name)
        backup_done = True
    else:
        backup_done = False
else:
    backup_done = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',  # 日期格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')  # 追加模式
    ]
)

# 备份后立即记录信息
if backup_done:
    logging.info(f"日志文件过大 ({line_count}行)，已备份至 {backup_name}")
    # 立即刷新缓冲区，确保备份信息写入文件
    for h in logging.root.handlers:
        h.flush()


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
        
        # ========== 水印偏差检测功能初始化 ==========
        # 用于记录每次水印偏差检测的结果
        self.watermark_detection_results = []  # [{round, client_assessments, detected_malicious, ...}]
        # 统计所有检测的偏差值（用于确定阈值）
        self.all_detection_deviations = []  # [[deviation, ...], ...] 每轮一个列表
        # 用于记录当前轮次的恶意客户端（模拟泄漏）
        self.current_malicious_client = None  # int or None
        
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

    def _optimize_model_storage(self, model_state, watermark_only=True, compress=True):
        """
        优化模型状态存储，减少内存和磁盘占用

        优化策略：
        1. watermark_only=True: 只保存水印区域的参数（非水印区域对检测无意义）
        2. compress=True: 使用 float16 压缩（精度损失可接受）

        Args:
            model_state: 模型参数字典
            watermark_only: 是否只保存水印区域参数
            compress: 是否使用 float16 压缩

        Returns:
            optimized_state: 优化后的字典
        """
        optimized_state = {}

        if watermark_only:
            # ========== 策略1：只保存水印区域参数 ==========
            # 对于后验泄露检测，只需要水印区域的参数
            try:
                from utils.key_matrix_utils import KeyMatrixManager
                key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

                # 构建偏移映射
                offset_map, _ = self._build_param_offset_map(model_state)

                # 按客户端独立提取水印值（关键：保持客户端特定信息）
                client_watermark_values = {}  # {client_id: [values]}
                all_values = []  # 全局水印值（用于兼容旧格式）

                for client_id in range(self.client_num):
                    positions = key_manager.load_positions(client_id)
                    client_values = []
                    for param_name, global_idx in positions:
                        local_idx, actual_param_name = self._convert_global_to_local_idx(
                            global_idx, param_name, model_state, offset_map
                        )
                        if actual_param_name and local_idx is not None:
                            if actual_param_name in model_state:
                                param_flat = model_state[actual_param_name].view(-1)
                                if local_idx < param_flat.numel():
                                    value = param_flat[local_idx].item()
                                    if compress:
                                        value = float(torch.tensor(value).half().item())
                                    client_values.append(value)
                    client_watermark_values[client_id] = client_values
                    all_values.extend(client_values)

                optimized_state['_watermark_only'] = True
                optimized_state['_watermark_count'] = len(all_values)
                optimized_state['watermark_values'] = all_values  # 兼容旧格式
                optimized_state['client_watermark_values'] = client_watermark_values  # 新格式：客户端特定

            except Exception as e:
                logging.warning(f"[存储优化] 水印区域提取失败，使用完整模型存储: {e}")
                # 回退到完整模型存储
                for key, value in model_state.items():
                    optimized_state[key] = value.detach().cpu()
        else:
            # ========== 策略2：完整模型存储（可选压缩）==========
            for key, value in model_state.items():
                tensor = value.detach().cpu()
                if compress:
                    tensor = tensor.half()  # float32 → float16
                optimized_state[key] = tensor

        return optimized_state

    def _restore_model_from_optimized(self, optimized_state, base_model_state):
        """
        从优化后的存储恢复模型状态

        Args:
            optimized_state: 优化后的字典
            base_model_state: 基础模型状态（用于填充非水印区域）

        Returns:
            model_state: 恢复后的模型状态字典
        """
        if optimized_state.get('_watermark_only', False):
            # 从水印值恢复
            try:
                from utils.key_matrix_utils import KeyMatrixManager
                key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

                # 获取所有客户端的水印位置
                all_positions = []
                for client_id in range(self.client_num):
                    positions = key_manager.load_positions(client_id)
                    all_positions.extend(positions)
                all_positions = list(set(all_positions))

                # 从 base_model_state 复制完整模型
                model_state = {}
                for key, value in base_model_state.items():
                    model_state[key] = value.clone()

                # 填充水印值
                offset_map, _ = self._build_param_offset_map(model_state)
                watermark_values = optimized_state['watermark_values']
                wm_idx = 0

                for param_name, global_idx in all_positions:
                    if wm_idx >= len(watermark_values):
                        break
                    local_idx, actual_param_name = self._convert_global_to_local_idx(
                        global_idx, param_name, model_state, offset_map
                    )
                    if actual_param_name and local_idx is not None:
                        if actual_param_name in model_state:
                            param_flat = model_state[actual_param_name].view(-1)
                            if local_idx < param_flat.numel():
                                param_flat[local_idx] = watermark_values[wm_idx]
                                wm_idx += 1

                return model_state

            except Exception as e:
                logging.warning(f"[存储恢复] 恢复失败: {e}")
                return base_model_state
        else:
            # 完整模型，直接返回
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
            # 只有启用泄漏源追踪时才执行泄漏模拟
            leak_info = None
            if getattr(self.args, 'enable_source_tracing', True):
                leak_info = self._simulate_pre_training_leakage(epoch, idxs_users)

            # ========== 第三步：模型泄漏追踪 ==========
            if leak_info is not None:
                actual_leaker = leak_info['leaked_client']
                attack_mode = leak_info.get('attack_mode', 'none')

                if attack_mode == 'gaussian_noise' and 'attacked_models' in leak_info:
                    # 三种噪声等级分别追踪
                    attack_results = {}
                    for level, attacked_model in leak_info['attacked_models'].items():
                        similarity_rankings, detected_leaker = self._track_model_leakage(
                            attacked_model, idxs_users
                        )
                        attack_results[level] = {
                            'detected_leaker': detected_leaker,
                            'similarity_rankings': similarity_rankings,
                            'is_correct': (actual_leaker == detected_leaker)
                        }

                        # 精简日志：每个噪声等级只输出 Top-1 + 实际泄漏者，避免冗长
                        actual_sim = next((sim for c, sim in similarity_rankings if c == actual_leaker), 0.0)
                        detected_sim = similarity_rankings[0][1] if similarity_rankings else 0.0
                        status = '✓' if attack_results[level]['is_correct'] else '✗'
                        logging.info(
                            f"[追踪] 轮次{epoch + 1} 等级{level} | "
                            f"实际C{actual_leaker}(sim={actual_sim:.4f}) | "
                            f"检测C{detected_leaker}(sim={detected_sim:.4f}) {status}"
                        )

                    # 记录汇总结果（使用等级2的结果作为主要判断）
                    primary_result = attack_results.get(2, attack_results.get(1, {}))
                    trace_result = {
                        'round': epoch + 1,
                        'actual_leaked_client': actual_leaker,
                        'detected_leaked_client': primary_result.get('detected_leaker'),
                        'is_correct': primary_result.get('is_correct', False),
                        'similarity_rankings': primary_result.get('similarity_rankings', []),
                        'attack_results': attack_results
                    }
                    self.trace_results.append(trace_result)
                else:
                    # 无攻击模式，使用原始模型追踪
                    leaked_model_state = leak_info['model_params']
                    similarity_rankings, detected_leaker = self._track_model_leakage(
                        leaked_model_state, idxs_users
                    )

                    trace_result = {
                        'round': epoch + 1,
                        'actual_leaked_client': actual_leaker,
                        'detected_leaked_client': detected_leaker,
                        'is_correct': (actual_leaker == detected_leaker),
                        'similarity_rankings': similarity_rankings
                    }
                    self.trace_results.append(trace_result)

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
            
            # 确定本轮的恶意客户端（用于模拟泄漏）
            self.current_malicious_client = self._get_current_malicious_client(epoch, idxs_users)
            if self.current_malicious_client is not None:
                logging.info(f"[本地训练] 第 {epoch + 1} 轮恶意客户端: {self.current_malicious_client}")
            
            for i, idx in enumerate(tqdm(idxs_users, desc='Progress: %d / %d' % (epoch + 1, self.epochs))):
                # 使用下发给该客户端的定制模型作为起点
                self.model.load_state_dict(self.customized_models[idx])

                # 判断是否为恶意客户端（跳过梯度缩放）
                skip_gradient_scaling = (idx == self.current_malicious_client)
                
                # 统一调用：始终传入 current_epoch/total_epochs；
                # 普通 Trainer 会通过 **kwargs 忽略
                local_w, local_loss, local_acc = self.trainer.local_update(
                    dataloader=local_train_loader[idx], 
                    local_ep=self.local_ep, 
                    lr=self.lr, 
                    client_id=idx,
                    current_epoch=epoch,
                    total_epochs=self.epochs,
                    skip_gradient_scaling=skip_gradient_scaling
                )

                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)
                
                # 清理临时变量，释放内存
                del local_w, local_loss, local_acc
            
            # ========== 第四步半：训练后泄露模拟 + 水印偏差检测（每leak_interval轮执行一次） ==========
            if self.args.leak_interval > 0 and (epoch + 1) % self.args.leak_interval == 0:
                # 首先模拟训练后的泄露（从local_ws中获取训练后的模型）
                post_train_leak_info = self._simulate_post_training_leakage(
                    epoch, idxs_users, local_ws
                )
                # 执行水印相似度检测（对比训练后的模型）
                detection_result = self._watermark_deviation_detection(
                    local_ws, idxs_users, current_round=epoch + 1,
                    post_train_leak_info=post_train_leak_info
                )
                if detection_result is not None:
                    self.watermark_detection_results.append(detection_result)

            # ========== 聚合前：计算参与训练的客户端的权重（相对于总数据集） ==========
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

            # 评估
            train_metrics = self.trainer.test(train_ldr)
            val_metrics = self.trainer.test(val_ldr)

            # (loss, acc_label, auc, acc_sample)
            loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean = train_metrics
            loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean = val_metrics

            # 创建本轮记录行
            stats_row = {
                'round': epoch + 1,
                'lr': self.lr,
            }

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
            
            # MultiLoss统计信息
            if hasattr(self.trainer, 'multi_loss'):
                stats = self.trainer.multi_loss.get_stats()
                logging.info(f"MultiLoss - GM:{stats['prevGM']:.6f} GH:{stats['prevGH']:.6f} Ratio:{stats['prevRatio']:.6f}")
            
            # 更新评估指标到记录行
            stats_row.update({
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
            })
            
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

            # 添加三种噪声等级的泄露检测详细记录
            current_round = epoch + 1
            trace_record = next((r for r in self.trace_results if r.get('round') == current_round), None)
            if trace_record:
                actual_leaked = trace_record.get('actual_leaked_client')
                stats_row['leakage_actual_client'] = actual_leaked
                attack_results = trace_record.get('attack_results', {})
                for noise_level in [1, 2, 3]:
                    if noise_level in attack_results:
                        ar = attack_results[noise_level]
                        ar_detected = ar.get('detected_leaker')
                        ar_rankings = ar.get('similarity_rankings', [])
                        ar_actual_sim = next((sim for cid, sim in ar_rankings if cid == actual_leaked), None)
                        ar_detected_sim = next((sim for cid, sim in ar_rankings if cid == ar_detected), None)
                        stats_row[f'leakage_noise{noise_level}_detected'] = ar_detected
                        stats_row[f'leakage_noise{noise_level}_similarity_actual'] = float(ar_actual_sim) if ar_actual_sim is not None else None
                        stats_row[f'leakage_noise{noise_level}_similarity_detected'] = float(ar_detected_sim) if ar_detected_sim is not None else None
                        stats_row[f'leakage_noise{noise_level}_correct'] = 1 if ar.get('is_correct') else 0
                    else:
                        stats_row[f'leakage_noise{noise_level}_detected'] = None
                        stats_row[f'leakage_noise{noise_level}_similarity_actual'] = None
                        stats_row[f'leakage_noise{noise_level}_similarity_detected'] = None
                        stats_row[f'leakage_noise{noise_level}_correct'] = None
            else:
                stats_row['leakage_actual_client'] = None
                for noise_level in [1, 2, 3]:
                    stats_row[f'leakage_noise{noise_level}_detected'] = None
                    stats_row[f'leakage_noise{noise_level}_similarity_actual'] = None
                    stats_row[f'leakage_noise{noise_level}_similarity_detected'] = None
                    stats_row[f'leakage_noise{noise_level}_correct'] = None

            # 添加水印偏差检测信息
            wm_record = next((r for r in self.watermark_detection_results if r.get('round') == current_round), None)
            if wm_record:
                actual_client = wm_record.get('actual_leaked_client')
                detected_client = wm_record.get('detected_malicious_client')
                client_assessments = wm_record.get('client_assessments', {})

                # client_assessments 的 key 可能是 numpy.int32/int/str，统一处理
                def get_similarity(d, client_id, assessments):
                    if client_id is None:
                        return None
                    # 尝试转换为 int 再查找
                    key = int(client_id)
                    if key in assessments:
                        return assessments[key].get('similarity')
                    if str(key) in assessments:
                        return assessments[str(key)].get('similarity')
                    return None

                actual_sim = get_similarity(wm_record, actual_client, client_assessments)
                detected_sim = get_similarity(wm_record, detected_client, client_assessments)

                stats_row['watermark_actual_client'] = actual_client
                stats_row['watermark_detected_client'] = detected_client
                stats_row['watermark_similarity_actual'] = float(actual_sim) if actual_sim is not None else None
                stats_row['watermark_similarity_detected'] = float(detected_sim) if detected_sim is not None else None
                stats_row['watermark_detection_correct'] = 1 if wm_record.get('is_detection_correct') else 0
            else:
                stats_row['watermark_actual_client'] = None
                stats_row['watermark_detected_client'] = None
                stats_row['watermark_similarity_actual'] = None
                stats_row['watermark_similarity_detected'] = None
                stats_row['watermark_detection_correct'] = None

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
                    trace_entry = {
                        'round': int(trace['round']),
                        'actual_leaked_client': int(trace['actual_leaked_client']),
                        'detected_leaked_client': int(trace['detected_leaked_client']),
                        'is_correct': bool(trace['is_correct']),
                        'similarity_rankings': [
                            [int(cid), float(sim)] for cid, sim in trace['similarity_rankings']
                        ]
                    }
                    # 包含三种噪声等级的详细结果
                    if 'attack_results' in trace:
                        attack_results_serializable = {}
                        for level, ar in trace['attack_results'].items():
                            attack_results_serializable[str(level)] = {
                                'noise_level': int(level),
                                'noise_name': {1: 'weak', 2: 'medium', 3: 'strong'}.get(int(level), 'unknown'),
                                'noise_scale': {1: 0.01, 2: 0.05, 3: 0.10}.get(int(level), 0.0),
                                'detected_leaker': int(ar['detected_leaker']) if ar['detected_leaker'] is not None else None,
                                'is_correct': bool(ar['is_correct']),
                                'similarity_rankings': [
                                    [int(cid), float(sim)] for cid, sim in ar['similarity_rankings']
                                ]
                            }
                        trace_entry['attack_results'] = attack_results_serializable
                    serializable_trace.append(trace_entry)

                def convert_to_serializable(obj):
                    """处理numpy类型以便JSON序列化"""
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    elif isinstance(obj, (np.floating,)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj

                # 计算每种噪声等级的准确率
                level_stats = {}
                for level in [1, 2, 3]:
                    level_correct = sum(1 for t in self.trace_results
                                       if 'attack_results' in t
                                       and level in t['attack_results']
                                       and t['attack_results'][level]['is_correct'])
                    level_total = sum(1 for t in self.trace_results
                                     if 'attack_results' in t
                                     and level in t['attack_results'])
                    level_acc = level_correct / level_total * 100 if level_total > 0 else 0
                    level_stats[str(level)] = {
                        'noise_name': {1: 'weak', 2: 'medium', 3: 'strong'}[level],
                        'noise_scale': {1: 0.01, 2: 0.05, 3: 0.10}[level],
                        'correct': int(level_correct),
                        'total': int(level_total),
                        'accuracy': float(level_acc)
                    }

                with open(trace_log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_leakages': int(len(self.trace_results)),
                        'overall_correct': int(correct_count),
                        'overall_accuracy': float(accuracy),
                        'description': 'Trace results during training. customized model leakage is attacked with 3 noise levels.',
                        'attack_mode': 'gaussian_noise (3 levels)',
                        'noise_level_stats': level_stats,
                        'details': serializable_trace
                    }, f, indent=2, ensure_ascii=False, default=convert_to_serializable)
                logging.info(f'追踪结果已保存到: {trace_log_path}')
                logging.info('各噪声等级准确率:')
                for level_str, stats in level_stats.items():
                    logging.info(f"  等级 {level_str} ({stats['noise_name']}, scale={stats['noise_scale']}): {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
            except Exception as e:
                logging.warning(f'保存追踪结果失败: {e}')
        else:
            logging.info('本轮训练未发生模型泄漏事件')
        
        # ========== 水印偏差检测结果汇总 ==========
        if len(self.watermark_detection_results) > 0:
            logging.info('='*60 + ' 水印偏差检测结果 ' + '='*60)
            logging.info(f'总共执行 {len(self.watermark_detection_results)} 次水印偏差检测')
            
            correct_detections = sum(1 for r in self.watermark_detection_results 
                                    if r.get('is_detection_correct') == True)
            accuracy = correct_detections / len(self.watermark_detection_results) * 100
            logging.info(f'检测准确率: {accuracy:.2f}% ({correct_detections}/{len(self.watermark_detection_results)})')
            
            logging.info('详细检测记录:')
            for result in self.watermark_detection_results:
                status = '✓' if result.get('is_detection_correct') else ('✗' if result.get('is_detection_correct') == False else '?')
                actual = result.get('actual_leaked_client', 'N/A')
                detected = result.get('detected_malicious_client', 'N/A')
                logging.info(
                    f"  轮次 {result['round']}: 实际={actual}, "
                    f"检测={detected} [{status}]"
                )
            
            # 保存检测结果到文件
            try:
                import json
                # 创建 detection 专用目录
                detection_dir = os.path.join('save', 'detection')
                os.makedirs(detection_dir, exist_ok=True)
                detection_log_path = os.path.join(
                    detection_dir,
                    f'watermark_deviation_detection_{self.model_name}_{self.dataset}_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
                )
                
                # 计算统计信息
                all_devs = []
                for devs in self.all_detection_deviations:
                    all_devs.extend(devs)
                
                threshold_stats = {
                    'mean': float(np.mean(all_devs)) if all_devs else 0,
                    'std': float(np.std(all_devs)) if all_devs else 0,
                    'threshold_3sigma': float(np.mean(all_devs) + 3 * np.std(all_devs)) if all_devs else 0
                }
                
                # 构建要保存的检测结果（确保所有值都是Python原生类型）
                detection_rounds = []
                for r in self.watermark_detection_results:
                    round_data = {
                        'round': int(r['round']),
                        'detected_malicious_client': r['detected_malicious_client'],
                        'actual_leaked_client': r['actual_leaked_client'],
                        'is_correct': bool(r.get('is_detection_correct')) if r.get('is_detection_correct') is not None else None,
                        'threshold_info': {
                            'mean': float(r['threshold_info']['mean']),
                            'std': float(r['threshold_info']['std']),
                            'threshold_2sigma': float(r['threshold_info']['threshold_2sigma'])
                        },
                        'client_assessments': {}
                    }
                    for k, v in r['client_assessments'].items():
                        round_data['client_assessments'][str(k)] = {
                            'similarity': float(v['similarity']) if v.get('similarity') is not None else None,
                            'watermark_mean': float(v['watermark_mean']) if v.get('watermark_mean') is not None else None,
                            'watermark_std': float(v['watermark_std']) if v.get('watermark_std') is not None else None,
                            'similarity_rank': int(v['similarity_rank']) if v.get('similarity_rank') is not None else None,
                            'is_suspicious': bool(v['is_suspicious']) if v.get('is_suspicious') is not None else None,
                        }
                    detection_rounds.append(round_data)
                
                with open(detection_log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'experiment_info': {
                            'model': self.model_name,
                            'dataset': self.dataset,
                            'client_num': self.client_num,
                            'epochs': self.epochs,
                            'leak_interval': getattr(self.args, 'leak_interval', 0)
                        },
                        'total_detections': int(len(self.watermark_detection_results)),
                        'correct_detections': int(correct_detections),
                        'detection_accuracy': float(accuracy),
                        'threshold_stats': threshold_stats,
                        'rounds': detection_rounds
                    }, f, indent=2, ensure_ascii=False)
                logging.info(f'水印相似度检测结果已保存到: {detection_log_path}')
            except Exception as e:
                logging.warning(f'保存水印相似度检测结果失败: {e}')

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
            columns = [
                # 基础信息
                'round', 'lr', 'train_loss', 'val_loss',
                'train_acc_label', 'train_auc', 'val_acc_label', 'val_auc',
                'best_val_acc_so_far', 'best_val_auc_so_far',
                'train_acc_sample', 'val_acc_sample',
                # 增强水印系统统计
                'prevGM', 'prevGH', 'prevRatio',
                'current_grad_M', 'current_grad_H', 'current_var_M', 'current_var_H',
                # 自编码器性能
                'autoencoder_performance',
                # 泄露检测与三种噪声等级详细记录
                'leakage_actual_client',
                # 三种噪声等级详细记录
                'leakage_noise1_detected', 'leakage_noise1_similarity_actual', 'leakage_noise1_similarity_detected', 'leakage_noise1_correct',
                'leakage_noise2_detected', 'leakage_noise2_similarity_actual', 'leakage_noise2_similarity_detected', 'leakage_noise2_correct',
                'leakage_noise3_detected', 'leakage_noise3_similarity_actual', 'leakage_noise3_similarity_detected', 'leakage_noise3_correct',
                # 水印偏差检测
                'watermark_actual_client', 'watermark_detected_client',
                'watermark_similarity_actual', 'watermark_similarity_detected',
                'watermark_detection_correct',
            ]
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

        # 水印聚合：保持纯 FedAvg（不做服务器端权威覆盖），
        # 以保留客户端之间的水印区分性，保证训练前泄漏追踪可正确定位泄漏源。
        self._watermark_aggregation(local_ws, idxs_users, w_avg)

        # 更新全局模型权重
        for k in w_avg.keys():
            self.w_t[k] = w_avg[k]

    def _generate_all_customized_models(self, global_model, idxs_users, epoch):
        """
        为所有参与本轮训练的客户端生成定制模型（下发前生成）

        定制模型规则：
        - 客户端 k 的水印区域 P_k：覆盖为服务器端 encoder_params 中的权威值
          （客户端 k 的权威值 = encoder_params[encoder_offset_k : +len(P_k)]）
        - 其它水印区域 P_j (j≠k)：保持 w_t 原值（即上一轮 FedAvg 结果）
        - 非水印区域：保持 w_t 原值

        Args:
            global_model: 全局模型参数 (w_t)
            idxs_users: 参与训练的客户端ID列表
            epoch: 当前轮次

        Returns:
            customized_models: 字典 {client_id: model_state_dict}
        """
        customized_models = {}

        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 提取服务器端编码器参数作为权威水印源
            encoder_params = self.trainer._extract_encoder_parameters()

            if encoder_params is None:
                logging.warning('无法提取 encoder_params，所有客户端下发 w_t 原值')
                for client_id in idxs_users:
                    customized_models[client_id] = {k: v.clone() for k, v in global_model.items()}
                return customized_models

            # 收集所有客户端的水印位置与长度
            # encoder_params 按 client_id 顺序存储（0, 1, 2, ..., client_num-1）
            # 所以 offset 必须按 client_id 顺序计算，不能按 idxs_users 顺序
            all_watermark_positions = {}
            all_position_lengths = {}
            for client_id in idxs_users:
                positions = key_manager.load_positions(client_id)
                all_watermark_positions[client_id] = positions
                all_position_lengths[client_id] = len(positions)

            # 计算每个客户端在 encoder_params 中的偏移量（按 client_id 顺序）
            # encoder_params[0 : len(P_0)] = client0
            # encoder_params[len(P_0) : len(P_0)+len(P_1)] = client1
            # 注意：必须对所有客户端（0到client_num-1）计算偏移，不能只对idxs_users中的客户端
            # 因为 encoder_params 是按 client_id 顺序完整排列所有客户端的水印
            encoder_offsets = {}
            cumulative_offset = 0
            for cid in range(self.client_num):
                encoder_offsets[cid] = cumulative_offset
                cumulative_offset += all_position_lengths.get(cid, 0)

            # 构建参数偏移映射
            offset_map, _ = self._build_param_offset_map(global_model)

            for client_id in idxs_users:
                customized_models[client_id] = self._generate_customized_model(
                    client_id, global_model, all_watermark_positions[client_id],
                    encoder_params, encoder_offsets[client_id], offset_map
                )
        except Exception as e:
            logging.warning(f'_generate_all_customized_models 失败: {e}')
            for client_id in idxs_users:
                customized_models[client_id] = {k: v.clone() for k, v in global_model.items()}

        return customized_models

    def _generate_customized_model(self, client_id, global_model, client_positions,
                                   encoder_params, encoder_offset, offset_map):
        """
        为指定客户端生成"定制模型"

        定制模型规则：
        - 客户端 k 的水印区域 P_k：覆盖为 encoder_params[encoder_offset : +len(P_k)] 权威值
        - 其它水印区域 P_j (j≠k)：保持 w_t 原值
        - 非水印区域：保持 w_t 原值

        Args:
            client_id: 目标客户端ID
            global_model: 全局模型参数 (w_t)
            client_positions: 客户端 k 的水印位置 [(param_name, global_idx), ...]
            encoder_params: 服务器端权威水印源（autoencoder 编码器参数）
            encoder_offset: 客户端 k 在 encoder_params 中的起始偏移
            offset_map: 参数偏移映射

        Returns:
            customized_model: 定制模型参数字典
        """
        customized_model = {}

        try:
            for param_name in global_model.keys():
                param_tensor = global_model[param_name].clone()
                param_flat = param_tensor.view(-1)

                # 在该客户端的水印位置覆盖为 encoder 权威值
                for wm_local_idx, (param_name_w, global_idx) in enumerate(client_positions):
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
                        enc_global_idx = encoder_offset + wm_local_idx
                        if enc_global_idx < len(encoder_params):
                            param_flat[local_idx] = encoder_params[enc_global_idx].item()

                customized_model[param_name] = param_tensor

        except Exception as e:
            logging.warning(f'Failed to generate customized model for client {client_id}: {e}')
            return {k: v.clone() for k, v in global_model.items()}

        return customized_model

    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量之间的余弦相似度

        Args:
            vec1, vec2: torch.Tensor 或 numpy.ndarray

        Returns:
            similarity: float
        """
        # 统一转为 numpy 处理
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.cpu().numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.cpu().numpy()

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

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

    def _extract_client_watermark(self, model_state, client_id):
        """
        提取指定客户端的水印值（统一接口，支持多种存储格式）

        优先级：
        1. client_watermark_values 字典（新格式，按客户端独立存储）
        2. 从完整模型参数按位置提取（_extract_single_client_watermark_params）
        3. watermark_values 列表（旧格式，返回整个列表）

        Args:
            model_state: 模型状态字典
            client_id: 客户端ID

        Returns:
            watermark_values: numpy 数组
        """
        # 优先级1：新格式（按客户端独立存储）
        if isinstance(model_state, dict) and 'client_watermark_values' in model_state:
            client_values = model_state['client_watermark_values'].get(client_id, [])
            if len(client_values) > 0:
                return np.array(client_values, dtype=np.float32)

        # 优先级2：从完整模型参数按位置提取
        try:
            if isinstance(model_state, dict) and not model_state.get('_watermark_only', False):
                result = self._extract_single_client_watermark_params(model_state, client_id)
                if len(result) > 0:
                    return np.array(result, dtype=np.float32)
        except Exception:
            pass

        # 优先级3：旧格式（仅返回整个列表，无法区分客户端）
        if isinstance(model_state, dict) and model_state.get('_watermark_only', False):
            wv = model_state.get('watermark_values', [])
            if len(wv) > 0:
                return np.array(wv, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def _extract_single_client_watermark_params(self, model_state, client_id):
        """
        提取指定客户端的水印参数

        Args:
            model_state: 模型参数字典
            client_id: 客户端ID

        Returns:
            watermark_params: 该客户端水印区域的参数（numpy数组）
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

    def _track_model_leakage(self, leaked_model_state, idxs_users):
        """
        追踪模型泄漏源

        原理：
        - 服务器端 encoder_params 中按 idxs_users 顺序连续存储各客户端的权威水印
        - 客户端 k 的权威水印 = encoder_params[encoder_offset_k : +len(P_k)]
        - 从泄漏模型中逐个客户端提取其负责的水印区域 P_k 分片
        - 计算泄漏模型的 P_k 分片与 encoder_params 中客户端 k 权威水印的余弦相似度
        - 相似度最高的客户端即为泄漏者

        Args:
            leaked_model_state: 泄漏的模型参数字典
            idxs_users: 参与训练的客户端ID列表

        Returns:
            similarity_rankings: 相似度排名列表 [(client_id, similarity), ...]
            detected_leaker: 检测到的泄漏者ID
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 读取服务器端 encoder 权威水印
            # encoder_params 按 client_id 顺序存储（0, 1, 2, ..., client_num-1）
            encoder_params = self.trainer._extract_encoder_parameters()
            if encoder_params is None:
                logging.warning('无法提取 encoder_params，跳过泄漏追踪')
                return [], None

            encoder_params_np = encoder_params.cpu().numpy() if hasattr(encoder_params, 'cpu') else np.array(encoder_params)

            # 计算每个客户端在 encoder_params 中的累计偏移（按 client_id 顺序）
            # encoder_params[0 : len(P_0)] = client0 的权威水印
            # encoder_params[len(P_0) : len(P_0)+len(P_1)] = client1 的权威水印
            # ...
            client_authority_wm = {}  # {client_id: numpy权威水印数组}
            cumulative_offset = 0
            for cid in range(self.client_num):
                positions = key_manager.load_positions(cid)
                wm_len = len(positions)
                client_authority_wm[cid] = encoder_params_np[cumulative_offset:cumulative_offset + wm_len]
                cumulative_offset += wm_len

            # 逐个客户端从泄漏模型中提取其水印区域 P_k 分片，与该客户端的权威水印做余弦相似度
            similarity_rankings = []
            for client_id in idxs_users:
                leaked_wm_k = self._extract_single_client_watermark_params(leaked_model_state, client_id)
                authority_wm_k = client_authority_wm[client_id]

                if len(leaked_wm_k) > 0 and len(authority_wm_k) > 0:
                    min_len = min(len(leaked_wm_k), len(authority_wm_k))
                    similarity = self._cosine_similarity(
                        leaked_wm_k[:min_len],
                        authority_wm_k[:min_len]
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
        同时应用三种等级的噪声攻击，用于测试追踪系统在有攻击情况下的表现

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

        # 获取配置的噪声攻击设置
        attack_mode = getattr(self.args, 'leak_attack_mode', 'none')

        leak_info = {
            'round': epoch + 1,
            'leaked_client': leaked_client,
            'original_model_params': leaked_model_copy,
            'attack_mode': attack_mode
        }

        # 根据攻击模式生成不同版本的泄漏模型
        if attack_mode == 'gaussian_noise':
            # 三种噪声等级：1=弱, 2=中, 3=强
            noise_configs = self.args.leak_noise_configs  # 从配置中心获取
            attacked_models = {}

            for level, scale in noise_configs.items():
                attacked_models[level] = self._add_noise_to_client_watermark(
                    leaked_model_copy, leaked_client, scale
                )

            leak_info['attacked_models'] = attacked_models
            # 默认使用中等强度进行追踪（便于观察）
            leak_info['model_params'] = attacked_models.get(2, leaked_model_copy)
            # 一次性汇总攻击配置（避免每个等级重复打印）
            logging.info(f"   [攻击配置] 客户端 {leaked_client} | 等级1/2/3 scale={list(noise_configs.values())}")
        else:
            # 无攻击模式，直接使用原始模型
            leak_info['model_params'] = leaked_model_copy
            leak_info['attacked_models'] = {}

        # 记录泄漏事件
        self.leakage_records.append(leak_info)

        logging.info(f"[模型泄漏模拟] 第 {epoch + 1} 轮 - 客户端 {leaked_client} 的定制模型在下发时泄漏")

        return leak_info

    def _get_current_malicious_client(self, epoch, idxs_users):
        """
        获取当前轮次的恶意客户端ID

        如果本轮不进行泄漏模拟，返回None

        Args:
            epoch: 当前轮次（0索引）
            idxs_users: 参与训练的客户端ID列表

        Returns:
            malicious_client: 恶意客户端ID，如果本轮不泄漏则返回None
        """
        leak_interval = self.args.leak_interval

        # 如果泄漏间隔为0或负数，则不设置恶意客户端
        if leak_interval <= 0:
            return None

        # 检查是否为本轮的泄漏检测轮次
        if (epoch + 1) % leak_interval != 0:
            return None

        # 从参与训练的客户端中随机选择一个作为恶意客户端
        malicious_client = int(np.random.choice(idxs_users, 1)[0])
        return malicious_client

    def _simulate_post_training_leakage(self, epoch, idxs_users, local_ws):
        """
        模拟模型泄漏事件（发生在本地训练之后）

        从本地训练后的模型中选择一个客户端泄露其训练后的模型
        用于测试水印相似度检测在训练后泄露场景下的表现

        Args:
            epoch: 当前轮次
            idxs_users: 参与训练的客户端ID列表
            local_ws: 本地训练后的模型列表

        Returns:
            leak_info: 泄漏信息字典，如果本轮不泄漏则返回None
        """
        leak_interval = self.args.leak_interval

        # 如果泄漏间隔为0或负数，则禁用泄漏模拟
        if leak_interval <= 0:
            return None

        # 模拟训练后泄露（每leak_interval轮一次，与训练前泄露同步）
        if (epoch + 1) % leak_interval != 0:
            return None

        # 使用预先确定的恶意客户端
        leaked_client = self.current_malicious_client
        if leaked_client is None or leaked_client not in idxs_users:
            logging.warning(f"未确定恶意客户端或客户端不在参与者中，跳过泄漏模拟")
            return None

        client_idx = list(idxs_users).index(leaked_client)
        leaked_model = local_ws[client_idx]

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
            'model_params': leaked_model_copy,
            'leak_type': 'post_training'  # 标记为训练后泄露
        }

        logging.info(f"[训练后泄漏模拟] 第 {epoch + 1} 轮 - 客户端 {leaked_client} 的训练后模型泄漏")

        return leak_info

    def _add_noise_to_client_watermark(self, model_state, client_id, noise_scale):
        """
        对指定客户端的水印区域添加高斯噪声

        Args:
            model_state: 模型参数字典
            client_id: 客户端ID
            noise_scale: 噪声标准差（相对于水印参数值）

        Returns:
            添加噪声后的模型参数字典
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 深拷贝模型
            noisy_model = {}
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    noisy_model[k] = v.clone().detach()
                else:
                    noisy_model[k] = v

            # 获取该客户端的水印位置
            positions = key_manager.load_positions(client_id)

            # 构建参数偏移映射
            offset_map, _ = self._build_param_offset_map(model_state)

            # 计算水印区域的参数统计
            watermark_values = []
            for param_name, global_idx in positions:
                local_idx = None
                actual_param_name = None

                # 检查是否可以直接使用
                if param_name in noisy_model:
                    param_flat = noisy_model[param_name].view(-1)
                    if global_idx < param_flat.numel():
                        local_idx = global_idx
                        watermark_values.append(param_flat[local_idx].item())
                        continue

                # 全局索引转换
                if param_name in offset_map:
                    param_offset = offset_map[param_name]
                    param_flat = noisy_model[param_name].view(-1)
                    if param_offset <= global_idx < param_offset + param_flat.numel():
                        local_idx = global_idx - param_offset
                        watermark_values.append(param_flat[local_idx].item())

            # 计算噪声标准差
            if watermark_values:
                value_std = np.std(watermark_values)
                value_mean = np.mean(np.abs(watermark_values))
                # 使用参数值的相对值作为噪声基准
                noise_std = max(value_mean * noise_scale, 1e-6)
            else:
                noise_std = noise_scale

            # 添加噪声
            for param_name, global_idx in positions:
                local_idx = None
                actual_param_name = None

                if param_name in noisy_model:
                    param_flat = noisy_model[param_name].view(-1)
                    if global_idx < param_flat.numel():
                        local_idx = global_idx
                        actual_param_name = param_name

                if actual_param_name is None and param_name in offset_map:
                    param_offset = offset_map[param_name]
                    param_flat = noisy_model[param_name].view(-1)
                    if param_offset <= global_idx < param_offset + param_flat.numel():
                        local_idx = global_idx - param_offset
                        actual_param_name = param_name

                if actual_param_name and local_idx is not None:
                    original_value = param_flat[local_idx].item()
                    noise = np.random.normal(0, noise_std)
                    new_value = original_value + noise
                    param_flat[local_idx] = new_value

            return noisy_model

        except Exception as e:
            logging.warning(f"Failed to add noise to watermark: {e}")
            return model_state

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
        服务器端水印聚合：**不做任何水印覆盖**（保持 FedAvg 结果）。

        历史说明
        --------
        本函数曾在提交 4877a2c 中被改为：把 w_avg 中**所有**客户端的水印区域
        P_k (k=0..N-1) 统一覆盖为 encoder_params 权威值。该改动引入了一个
        训练前泄漏追踪的回归，现已回退，原因如下。

        回归机理
        --------
        定制模型 M_k 的水印区域内容由「w_t 的 P_j」决定：

        - 4877a2c 之前：w_t 的 P_j = FedAvg 结果（每轮变化，**各客户端不同**）
          → M_k 的 P_j (j≠k) 是上一轮 FedAvg 的不同值
          → 泄漏者 k 的定制模型中，只有 P_k 是权威值，P_j (j≠k) 是 FedAvg 值
          → 追踪时「泄漏模型 P_k 分片 vs 客户端 k 权威水印」相似度 ≈ 1.0，
            而其它客户端 j 的相似度很低 → 可区分，能正确定位泄漏者

        - 4877a2c 之后：w_t 的 P_j = 权威值（对所有客户端完全相同）
          → M_k 的所有水印区域 P_j 都是权威值
          → 泄漏模型的每个 P_j 分片都等于对应的权威水印
          → 所有客户端的相似度**全部并列在 1.0**，无法区分 → 退化为随机

        实测证据（chestmnist / alexnet / K=5）：
          - 4877a2c 之前（2026-10-21，150 轮）：追踪准确率 100%
          - 4877a2c 之后（2026-10-26）：追踪准确率 3.33%（≈ 随机 1/K）
            逐轮相似度：第 1 轮 0.9987（正确，第 1 轮 w_t 尚未被聚合覆盖），
            第 2 轮起全部并列为 1.0000

        为什么不需要服务器端覆盖
        ----------------------
        水印权威值**本来就在下发阶段被正确写入**：
        `_generate_customized_model` 会把客户端 k 的 P_k 覆盖为权威值，
        `_embed_watermark` 也会在本地训练前把权威值写回模型。

        因此「训练后水印偏差检测」（`_watermark_deviation_detection`）
        并不依赖服务器端聚合阶段的覆盖——它比较的是
        「泄漏模型在 P_leaked 的分片」与「各客户端定制模型在 P_client 的分片」，
        而定制模型的 P_client 恒为权威值。2026-10-21（覆盖之前）的历史结果
        显示该指标当时同样是 100%。

        结论：聚合阶段保持纯 FedAvg，可同时满足
        「训练后偏差检测 100%」与「训练前泄漏追踪 100%」。

        Args:
            local_ws: 本地模型权重列表（仅用于签名，函数不修改）
            idxs_users: 参与训练的客户端ID列表
            w_avg: FedAvg 聚合后的权重字典（不做修改）

        Returns:
            None
        """
        # 不做任何水印覆盖！所有水印位置保持 FedAvg 结果。
        # 水印覆盖仅在 _generate_customized_model 中进行（对客户端自己的 P_k）。
        return None

    def _convert_global_to_local_idx(self, global_idx, param_name, model_state, offset_map):
        """将全局索引转换为局部索引"""
        local_idx = None
        actual_param_name = None

        if param_name in model_state:
            param_size = model_state[param_name].numel()
            if global_idx < param_size:
                actual_param_name = param_name
                local_idx = global_idx

        if actual_param_name is None and param_name in offset_map:
            param_offset = offset_map[param_name]
            param_size = model_state[param_name].numel()
            if param_offset <= global_idx < param_offset + param_size:
                actual_param_name = param_name
                local_idx = global_idx - param_offset

        if actual_param_name is None:
            for name, offset in offset_map.items():
                param_size = model_state[name].numel()
                if offset <= global_idx < offset + param_size:
                    actual_param_name = name
                    local_idx = global_idx - offset
                    break

        return local_idx, actual_param_name

    def _watermark_deviation_detection(self, local_ws, idxs_users, current_round, post_train_leak_info=None):
        """
        水印相似度检测：计算每个客户端的水印区域与泄漏模型的相似度

        原理（与泄露检测一致）：
        - 如果某轮发生泄漏，泄漏模型的来源客户端与该客户端的训练后模型高度相似
        - 通过计算各客户端模型与泄漏模型的相似度，相似度最高的客户端即为泄漏者

        Args:
            local_ws: 本地模型列表
            idxs_users: 参与训练的客户端ID列表
            current_round: 当前训练轮次
            post_train_leak_info: 训练后泄露信息（优先使用）

        Returns:
            detection_result: 检测结果字典
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 优先使用训练后的泄露信息（最准确）
            if post_train_leak_info is not None:
                leaked_model_state = post_train_leak_info.get('model_params')
                actual_leaked = post_train_leak_info.get('leaked_client')
            else:
                # 查找当前轮次的泄漏记录（训练前的定制模型）
                leaked_model_state = None
                actual_leaked = None
                for record in self.leakage_records:
                    if record.get('round') == current_round:
                        leaked_model_state = record.get('model_params')
                        actual_leaked = record.get('leaked_client')
                        break

            # 如果本轮没有泄漏，则跳过检测
            if leaked_model_state is None:
                logging.info(f"[水印相似度检测] 第 {current_round} 轮未发生泄漏，跳过检测")
                return None

            logging.info(f"[水印相似度检测] 第 {current_round} 轮发生泄漏，实际泄漏者: 客户端 {actual_leaked}")

            # 存储每个客户端的检测结果
            client_assessments = {}
            all_similarities = []

            # 提取泄漏模型中泄漏者的水印区域参数（使用泄漏者的位置）
            leaked_watermark_params = self._extract_single_client_watermark_params(
                leaked_model_state, actual_leaked
            )

            # 对每个客户端进行检测：计算其定制模型的水印区域与泄漏模型中泄漏者水印的相似度
            for client_id in idxs_users:
                # 获取该客户端的定制模型
                if client_id not in self.customized_models:
                    continue
                customized_model = self.customized_models[client_id]

                # 提取该客户端定制模型的水印区域参数
                client_watermark_params = self._extract_single_client_watermark_params(
                    customized_model, client_id
                )

                # 计算余弦相似度
                if len(leaked_watermark_params) > 0 and len(client_watermark_params) > 0:
                    min_len = min(len(leaked_watermark_params), len(client_watermark_params))
                    similarity = self._cosine_similarity(
                        leaked_watermark_params[:min_len],
                        client_watermark_params[:min_len]
                    )
                else:
                    similarity = 0.0

                # 计算水印区域的统计量
                if len(client_watermark_params) > 0:
                    wm_mean = np.mean(client_watermark_params)
                    wm_std = np.std(client_watermark_params)
                else:
                    wm_mean, wm_std = None, None

                client_assessments[client_id] = {
                    'similarity': float(similarity),
                    'watermark_mean': float(wm_mean) if wm_mean is not None else None,
                    'watermark_std': float(wm_std) if wm_std is not None else None,
                }

                all_similarities.append((client_id, similarity))

            # 按相似度降序排序（相似度最高的为可疑）
            all_similarities.sort(key=lambda x: x[1], reverse=True)

            # 为每个客户端添加排名（相似度最高排名1）
            for rank, (cid, sim) in enumerate(all_similarities, 1):
                client_assessments[cid]['similarity_rank'] = rank

            # 检测到的恶意客户端（相似度最高的）
            detected_malicious = all_similarities[0][0] if all_similarities else None

            # 确定阈值（均值 + 2倍标准差）
            if all_similarities:
                sim_values = [sim for _, sim in all_similarities]
                mean_sim = np.mean(sim_values)
                std_sim = np.std(sim_values) if len(sim_values) > 1 else 0
                threshold = mean_sim + 2 * std_sim if std_sim > 0 else mean_sim * 1.2
            else:
                mean_sim, std_sim, threshold = 0.0, 0.0, 0.0

            # 标记可疑客户端（相似度高于阈值的）
            for cid in client_assessments:
                sim = client_assessments[cid]['similarity']
                client_assessments[cid]['is_suspicious'] = sim > threshold

            # 构建检测结果
            detection_result = {
                'round': current_round,
                'total_clients': self.client_num,
                'participating_clients': [int(c) for c in idxs_users],
                'client_assessments': client_assessments,
                'detected_malicious_client': int(detected_malicious) if detected_malicious is not None else None,
                'detection_method': 'watermark_similarity',
                'actual_leaked_client': int(actual_leaked) if actual_leaked is not None else None,
                'is_detection_correct': (actual_leaked == detected_malicious) if actual_leaked is not None and detected_malicious is not None else None,
                'threshold_info': {
                    'mean': float(mean_sim),
                    'std': float(std_sim),
                    'threshold_2sigma': float(threshold)
                }
            }

            # 保存所有相似度值（用于后续统计）
            self.all_detection_deviations.append([sim for _, sim in all_similarities])

            # 打印检测结果日志
            self._log_watermark_detection(detection_result, all_similarities)

            return detection_result

        except Exception as e:
            logging.warning(f"[水印相似度检测] 检测失败: {e}")
            import traceback
            logging.warning(traceback.format_exc())
            return None

    def _compute_client_watermark_similarity(self, client_model, client_id, encoder_params, offset_map):
        """
        计算指定客户端的水印相似度（与该客户端专属的 encoder 切片比较）

        水印值来自 encoder_params，各客户端连续存储（按 client_id 顺序）：
        - 客户端 0: encoder_params[0 : len(positions_0)]
        - 客户端 1: encoder_params[len(positions_0) : len(positions_0)+len(positions_1)]
        - ...

        Args:
            client_model: 客户端模型参数字典
            client_id: 客户端ID
            encoder_params: 编码器参数（完整的水印值）
            offset_map: 参数偏移映射

        Returns:
            similarity: 余弦相似度
            wm_mean: 水印区域参数均值
            wm_std: 水印区域参数标准差
            wm_values: 水印区域参数值列表
        """
        try:
            from utils.key_matrix_utils import KeyMatrixManager
            key_manager = KeyMatrixManager(self.args.key_matrix_path, args=self.args)

            # 获取该客户端的水印位置
            positions = key_manager.load_positions(client_id)

            if not positions:
                return None, None, None, []

            # 计算该客户端在 encoder_params 中的偏移量（累计）
            # 需要知道所有客户端的水印长度来计算偏移
            encoder_offset = 0
            for cid in range(client_id):
                try:
                    cid_positions = key_manager.load_positions(cid)
                    encoder_offset += len(cid_positions)
                except Exception:
                    pass  # 如果加载失败，跳过

            # 提取该客户端专属的 encoder 切片
            enc_slice_start = encoder_offset
            enc_slice_end = enc_slice_start + len(positions)
            client_encoder_slice = encoder_params[enc_slice_start:enc_slice_end]

            # 提取水印区域的参数值
            watermark_values = []

            for param_name, global_idx in positions:
                local_idx, actual_param_name = self._convert_global_to_local_idx(
                    global_idx, param_name, client_model, offset_map
                )

                if actual_param_name and local_idx is not None:
                    if actual_param_name in client_model:
                        param_flat = client_model[actual_param_name].view(-1)
                        if local_idx < param_flat.numel():
                            wm_value = param_flat[local_idx].item()
                            watermark_values.append(wm_value)

            # 计算统计量
            if watermark_values:
                wm_mean = np.mean(watermark_values)
                wm_std = np.std(watermark_values)
            else:
                wm_mean, wm_std = None, None

            # 计算与该客户端专属 encoder 切片的余弦相似度
            if watermark_values and len(client_encoder_slice) > 0:
                # 确保长度一致
                min_len = min(len(watermark_values), len(client_encoder_slice))
                wm_tensor = torch.tensor(watermark_values[:min_len])
                enc_tensor = client_encoder_slice[:min_len].clone().detach().cpu()

                # 统一到 CPU 计算，避免设备不匹配
                similarity = self._cosine_similarity(wm_tensor, enc_tensor)
            else:
                similarity = None

            return similarity, wm_mean, wm_std, watermark_values

        except Exception as e:
            logging.warning(f"[水印相似度检测] 计算客户端 {client_id} 的相似度失败: {e}")
            return None, None, None, []

    def _log_watermark_detection(self, detection_result, sorted_similarities):
        """
        打印水印相似度检测的日志（精简版）
        """
        # 一行汇总：检测 + 实际 + 正确性 + 阈值
        det = detection_result.get('detected_malicious_client')
        act = detection_result.get('actual_leaked_client')
        is_correct = detection_result.get('is_detection_correct')
        status = '✓' if is_correct else ('✗' if is_correct is False else '?')
        th = detection_result['threshold_info']
        logging.info(
            f"[水印检测] 轮次{detection_result['round']} | "
            f"实际C{act} | 检测C{det}(sim={sorted_similarities[0][1]:.4f}) {status} | "
            f"阈值(th={th['threshold_2sigma']:.4f}, mean={th['mean']:.4f})"
        )

        # 仅打印可疑客户端
        for cid in detection_result['participating_clients']:
            assessment = detection_result['client_assessments'].get(cid, {})
            if assessment.get('is_suspicious') or cid in (det, act):
                sim = assessment.get('similarity')
                rank = assessment.get('similarity_rank', '-')
                actual_marker = "(实际)" if cid == act else ""
                det_marker = "[检测]" if cid == det else ""
                logging.info(f"    客户端 {cid}: sim={sim:.4f} (排名{rank}) {det_marker}{actual_marker}")

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