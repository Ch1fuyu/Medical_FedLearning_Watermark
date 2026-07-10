import copy
import os
import sys
import time
from datetime import datetime
import logging
import gc
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.globals import set_seed
from models.alexnet import AlexNet
from models.resnet import resnet18
from utils.base import Experiment
from utils.dataset import get_data, DatasetSplit
from utils.trainer_private import TrainerPrivate, TesterPrivate
import pandas as pd

set_seed()


# ========================= 基准测试配置参数 ========================
def get_baseline_config():
    """
    基准测试配置 - 所有参数都在这里设置
    不依赖 args.py 和命令行参数
    """
    config = SimpleNamespace()
    
    # ==================== 实验基础配置 ====================
    config.gpu = '0'  # GPU设备ID
    config.dataset = 'chestmnist'  # 数据集: 'chestmnist', 'cifar10', 'cifar100'
    config.model_name = 'alexnet'  # 模型: 'alexnet', 'resnet'
    
    # ==================== 联邦学习参数 ====================
    config.epochs = 150  # 全局训练轮次
    config.local_ep = 2  # 每个客户端的本地训练轮次
    config.batch_size = 128  # 批次大小
    config.frac = 1.0  # 参与训练的客户端比例 (1.0 = 100%)
    config.iid = True  # IID数据分布
    
    # ==================== 优化器参数 ====================
    config.optim = 'adam'  # 优化器: 'sgd', 'adam'
    config.lr = 0.001  # 学习率
    config.wd = 0.0001  # 权重衰减 (L2正则化)
    config.use_lr_scheduler = True  # 使用余弦退火学习率调度器
    config.dropout_rate = 0.5  # Dropout率
    
    # ==================== 差分隐私参数 ====================
    config.dp = False  # 启用差分隐私
    config.sigma = 0.1  # 高斯噪声标准差
    
    # ==================== 数据集特定参数 ====================
    # 这些参数会根据数据集自动设置
    if config.dataset == 'chestmnist':
        config.num_classes = 14
        config.in_channels = 1  # 灰度图像
        config.input_size = 28  # ChestMNIST图像尺寸
        config.task_type = 'multilabel'  # 多标签分类
    elif config.dataset == 'cifar10':
        config.num_classes = 10
        config.in_channels = 3  # RGB图像
        config.input_size = 32  # CIFAR-10图像尺寸
        config.task_type = 'multiclass'  # 多类别分类
    elif config.dataset == 'cifar100':
        config.num_classes = 100
        config.in_channels = 3  # RGB图像
        config.input_size = 32  # CIFAR-100图像尺寸
        config.task_type = 'multiclass'  # 多类别分类
    else:
        raise ValueError(f"不支持的数据集: {config.dataset}")
    
    # ==================== 损失函数参数 ====================
    config.class_weights = False  # 不使用类别权重
    config.pos_weight_factor = 1.0  # 正样本权重因子（仅在class_weights=True时使用）
    config.use_multiloss = False  # 基准模式不使用多重损失
    config.use_focal_loss = False  # 基准模式不使用Focal Loss
    
    # ==================== 水印相关参数（基准模式禁用）====================
    config.enable_watermark = False  # 基准模式禁用水印
    config.watermark_mode = 'baseline'
    config.use_key_matrix = False
    
    # ==================== 保存路径参数 ====================
    config.save_model_dir = 'save'  # 模型保存目录
    config.save_excel_dir = 'save/excel'  # Excel保存目录
    config.log_file = './logs/console_baseline.logs'  # 日志文件路径
    config.data_root = './data'  # 数据集根目录
    
    # ==================== 其他参数 ====================
    config.log_interval = 1  # 评估间隔
    config.baseline_mode = True  # 基准模式标志
    config.patience = 10  # Early stopping耐心值（如果验证指标连续N轮未提升则停止训练，设置为0禁用）
    
    return config


# ========================= 配置日志系统 ========================
def setup_logging(log_file):
    """配置日志系统"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H-%M-%S',
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到控制台
            logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 追加模式
        ]
    )


class BaselineFederatedLearning(Experiment):
    """基准联邦学习训练类 - 不包含任何水印功能"""
    
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma
        
        # 强制设置为基准模式
        self.args.enable_watermark = False
        self.args.watermark_mode = 'baseline'
        
        logging.info("=" * 60)
        logging.info("🚀 启动基准联邦学习训练")
        logging.info(f"📊 数据集: {args.dataset}")
        logging.info(f"🏗️ 模型: {args.model_name}")
        logging.info(f"📈 训练轮次: {args.epochs}")
        logging.info(f"🔒 差分隐私: {'启用' if args.dp else '禁用'}")
        if args.dp:
            logging.info(f"📊 噪声参数: {args.sigma}")
        logging.info("=" * 60)

    def get_model(self):
        """获取模型"""
        if self.args.model_name == 'alexnet':
            dropout_rate = getattr(self.args, 'dropout_rate', 0.5)
            model = AlexNet(self.args.in_channels, self.args.num_classes, 
                           input_size=self.args.input_size, dropout_rate=dropout_rate)
        elif self.args.model_name in ['resnet', 'resnet18']:
            model = resnet18(num_classes=self.args.num_classes, 
                           in_channels=self.args.in_channels, 
                           input_size=self.args.input_size)
        else:
            raise ValueError(f"不支持的模型: {self.args.model_name}")
        
        logging.info(f"✅ 模型 {self.args.model_name} 初始化完成")
        return model

    def get_trainer(self, model, device):
        """获取训练器 - 使用基础训练器，不包含水印功能"""
        trainer = TrainerPrivate(model, device, self.dp, self.sigma, None, args=self.args)
        logging.info("✅ 基准训练器初始化完成")
        return trainer

    def train(self):
        """执行联邦学习训练"""
        logging.info("🎯 开始基准联邦学习训练...")
        
        # 获取数据
        train_data, test_data, user_groups = get_data(self.args.dataset, self.data_root, self.iid)
        logging.info(f"📊 数据加载完成: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")
        
        # 获取全局模型
        global_model = self.get_model()
        
        # 初始化训练器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = self.get_trainer(global_model, device)
        
        # 训练历史记录
        train_losses = []
        train_accs = []
        train_sample_accs = []
        val_losses = []
        val_accs = []
        val_sample_accs = []
        val_aucs = []
        
        # 用于保存最佳模型
        best_model_state = None
        best_auc = 0.0
        best_acc = 0.0
        best_epoch = 0
        
        # Early Stopping 配置
        patience = getattr(self.args, 'patience', 10)
        early_stop_counter = 0
        
        # 决定模型选择依据：ChestMNIST 按 AUC，其他（如 CIFAR-10/100）按准确率
        dataset_key = (self.args.dataset or '').lower()
        select_by_auc = (dataset_key == 'chestmnist')
        
        # 联邦学习主循环
        for epoch in range(self.args.epochs):
            logging.info(f"\n🔄 轮次 {epoch + 1}/{self.args.epochs}")
            
            # 选择参与训练的客户端
            m = max(int(self.args.frac * len(user_groups)), 1)
            idxs_users = np.random.choice(range(len(user_groups)), m, replace=False)
            
            # 全局模型参数
            global_weights = global_model.state_dict()
            
            # 客户端本地训练
            local_weights = []
            local_losses = []
            
            for idx in idxs_users:
                # 创建客户端数据
                local_data = DatasetSplit(train_data, user_groups[idx])
                local_loader = DataLoader(local_data, batch_size=self.args.batch_size, shuffle=True)
                
                # 本地训练（传递全局轮次信息用于学习率调度）
                w, loss, acc = trainer.local_update(
                    local_loader, 
                    self.args.local_ep, 
                    self.args.lr, 
                    idx,
                    current_epoch=epoch,
                    total_epochs=self.args.epochs
                )
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
            
            # 联邦平均
            global_weights = self.federated_averaging(global_weights, local_weights)
            global_model.load_state_dict(global_weights)
            
            # 计算平均训练损失
            avg_train_loss = np.mean(local_losses)
            train_losses.append(avg_train_loss)
            
            # 在训练集上评估（计算训练准确率）
            # 使用随机抽样的训练数据子集以提高效率
            train_eval_indices = np.random.choice(len(train_data), min(len(test_data), len(train_data)), replace=False)
            train_eval_data = torch.utils.data.Subset(train_data, train_eval_indices)
            train_eval_loader = DataLoader(train_eval_data, batch_size=self.args.batch_size, shuffle=False)
            train_loss, train_acc, train_auc, train_sample_acc = trainer.test(train_eval_loader)
            train_accs.append(train_acc)
            train_sample_accs.append(train_sample_acc)
            
            # 测试全局模型
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            test_loss, test_acc, test_auc, test_sample_acc = trainer.test(test_loader)
            
            val_losses.append(test_loss)
            val_accs.append(test_acc)  # 标签级准确率
            val_sample_accs.append(test_sample_acc)  # 样本级准确率
            val_aucs.append(test_auc)
            
            # 保存最佳模型和Early Stopping判断
            if select_by_auc:
                # ChestMNIST：基于AUC
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_acc = test_acc  # 同时记录准确率
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(global_model.state_dict())
                    early_stop_counter = 0
                    logging.info(f"⭐ 新的最佳模型! AUC: {best_auc:.4f}")
                else:
                    early_stop_counter += 1
            else:
                # CIFAR-10/100：基于准确率
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_auc = test_auc  # 同时记录AUC
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(global_model.state_dict())
                    early_stop_counter = 0
                    logging.info(f"⭐ 新的最佳模型! 准确率: {best_acc:.4f}%")
                else:
                    early_stop_counter += 1
            
            # 记录训练进度
            logging.info(f"📊 训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}% (样本级: {train_sample_acc:.4f}%)")
            logging.info(f"📊 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}% (样本级: {test_sample_acc:.4f}%)")
            logging.info(f"📊 测试AUC: {test_auc:.4f}")
            if select_by_auc:
                logging.info(f"🏆 当前最佳AUC: {best_auc:.4f} (轮次 {best_epoch + 1})")
            else:
                logging.info(f"🏆 当前最佳准确率: {best_acc:.4f}% (轮次 {best_epoch + 1})")
            
            # Early Stopping检查
            if patience > 0 and early_stop_counter >= patience:
                if select_by_auc:
                    logging.info(f"🛑 Early stopping 触发于轮次 {epoch + 1}。最佳验证AUC: {best_auc:.4f}")
                else:
                    logging.info(f"🛑 Early stopping 触发于轮次 {epoch + 1}。最佳验证准确率: {best_acc:.4f}%")
                break
            
            # 清理内存
            del local_weights, local_losses
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 保存训练结果
        self.save_results(train_losses, train_accs, train_sample_accs, 
                         val_losses, val_accs, val_sample_accs, val_aucs,
                         best_model_state, best_epoch, best_auc)
        
        logging.info("🎉 基准联邦学习训练完成!")
        return global_model

    def federated_averaging(self, global_weights, local_weights):
        """联邦平均算法"""
        # 计算平均权重
        avg_weights = copy.deepcopy(global_weights)
        
        # 确定目标设备（使用全局权重的设备）
        device = next(iter(global_weights.values())).device
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for local_weight in local_weights:
                # 将本地权重移到正确的设备
                avg_weights[key] += local_weight[key].to(device)
            avg_weights[key] = avg_weights[key] / len(local_weights)
        
        return avg_weights

    def save_results(self, train_losses, train_accs, train_sample_accs, 
                    val_losses, val_accs, val_sample_accs, val_aucs,
                    best_model_state, best_epoch, best_auc):
        """保存训练结果和最佳模型"""
        # 创建结果目录
        save_dir = os.path.join(self.args.save_model_dir, self.args.model_name, self.args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建结果数据
        results = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'train_acc_label': train_accs,
            'train_acc_sample': train_sample_accs,
            'val_loss': val_losses,
            'val_acc_label': val_accs,
            'val_acc_sample': val_sample_accs,
            'val_auc': val_aucs
        }
        
        # 保存为CSV
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        csv_filename = f'{timestamp}_baseline_{self.args.dataset}_{self.args.model_name}_results.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        logging.info(f"📁 训练结果已保存: {csv_path}")
        
        # 保存最佳模型权重
        if best_model_state is not None:
            model_filename = f'{timestamp}_baseline_{self.args.dataset}_{self.args.model_name}_best_auc_{best_auc:.4f}.pth'
            model_path = os.path.join(save_dir, model_filename)
            torch.save(best_model_state, model_path)
            
            logging.info(f"🏆 最佳模型已保存 (AUC: {best_auc:.4f}, 轮次: {best_epoch + 1})")
            logging.info(f"📁 模型路径: {model_path}")
        else:
            logging.warning("⚠️ 没有找到最佳模型状态，跳过模型保存")


def main():
    """主函数"""
    # 获取基准测试配置（所有参数都在配置函数中定义）
    args = get_baseline_config()
    
    # 配置日志系统
    setup_logging(args.log_file)
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    logging.info("=" * 80)
    logging.info("🚀 启动基准联邦学习训练脚本 (Baseline Mode)")
    logging.info("=" * 80)
    logging.info("")
    logging.info("📋 实验配置参数:")
    logging.info("-" * 80)
    logging.info(f"  🔹 数据集配置:")
    logging.info(f"     - 数据集: {args.dataset}")
    logging.info(f"     - 类别数: {args.num_classes}")
    logging.info(f"     - 输入通道: {args.in_channels}")
    logging.info(f"     - 数据根目录: {args.data_root}")
    logging.info("")
    logging.info(f"  🔹 模型配置:")
    logging.info(f"     - 模型架构: {args.model_name}")
    logging.info(f"     - Dropout率: {args.dropout_rate}")
    logging.info("")
    logging.info(f"  🔹 联邦学习参数:")
    logging.info(f"     - 全局训练轮次: {args.epochs}")
    logging.info(f"     - 本地训练轮次: {args.local_ep}")
    logging.info(f"     - 参与比例: {args.frac * 100:.0f}%")
    logging.info(f"     - 数据分布: {'IID' if args.iid else 'Non-IID'}")
    logging.info(f"     - Early Stopping耐心值: {args.patience} {'(已启用)' if args.patience > 0 else '(已禁用)'}")
    logging.info("")
    logging.info(f"  🔹 优化器参数:")
    logging.info(f"     - 优化器: {args.optim.upper()}")
    logging.info(f"     - 学习率: {args.lr}")
    logging.info(f"     - 权重衰减: {args.wd}")
    logging.info(f"     - 批次大小: {args.batch_size}")
    logging.info(f"     - 学习率调度器: {'启用' if args.use_lr_scheduler else '禁用'}")
    logging.info("")
    logging.info(f"  🔹 隐私保护:")
    logging.info(f"     - 差分隐私: {'✅ 启用' if args.dp else '❌ 禁用'}")
    if args.dp:
        logging.info(f"     - 噪声参数 σ: {args.sigma}")
    logging.info("")
    logging.info(f"  🔹 水印状态:")
    logging.info(f"     - 水印嵌入: {'启用' if args.enable_watermark else '❌ 禁用 (基准模式)'}")
    logging.info("")
    logging.info(f"  🔹 保存路径:")
    logging.info(f"     - 模型保存: {args.save_model_dir}")
    logging.info(f"     - 日志文件: {args.log_file}")
    logging.info("-" * 80)
    logging.info("")
    
    # 创建实验实例
    experiment = BaselineFederatedLearning(args)
    
    # 开始训练
    start_time = time.time()
    try:
        logging.info("🏁 开始训练...")
        logging.info("")
        model = experiment.train()
        end_time = time.time()
        
        # 计算训练时间
        total_seconds = end_time - start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        logging.info("")
        logging.info("=" * 80)
        logging.info("🎉 基准联邦学习训练成功完成!")
        logging.info(f"⏱️  总训练时间: {hours}小时 {minutes}分钟 {seconds}秒 ({total_seconds:.2f}秒)")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.warning("\n⚠️  训练被用户中断")
        raise
    except Exception as e:
        logging.error(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

