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
import pandas as pd

set_seed()

# 配置 logging
args = parser_args()
log_file_name = args.log_file.replace('.log', '_baseline.log')  # 基准模式使用不同的日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',  # 日期格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')  # 追加模式
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
        logging.info(f"👥 客户端数量: {args.client_num}")
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
        train_data, test_data, user_groups = get_data(self.args.dataset, self.data_root, self.iid, self.client_num)
        logging.info(f"📊 数据加载完成: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")
        
        # 获取全局模型
        global_model = self.get_model()
        
        # 初始化训练器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = self.get_trainer(global_model, device)
        
        # 训练历史记录
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        
        # 联邦学习主循环
        for epoch in range(self.args.epochs):
            logging.info(f"\n🔄 轮次 {epoch + 1}/{self.args.epochs}")
            
            # 选择参与训练的客户端
            m = max(int(self.args.frac * self.args.client_num), 1)
            idxs_users = np.random.choice(range(self.args.client_num), m, replace=False)
            
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
                w, loss = trainer.local_update(
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
            
            # 测试全局模型
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            test_loss, test_acc, test_auc, test_sample_acc = trainer.test(test_loader)
            
            val_losses.append(test_loss)
            val_accs.append(test_acc)  # 标签级准确率
            val_aucs.append(test_auc)
            
            # 记录训练进度
            logging.info(f"📊 训练损失: {avg_train_loss:.4f}")
            logging.info(f"📊 测试损失: {test_loss:.4f}")
            logging.info(f"📊 测试准确率: {test_acc:.2f}%")
            logging.info(f"📊 测试AUC: {test_auc:.4f}")
            
            # 清理内存
            del local_weights, local_losses
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 保存训练结果
        self.save_results(train_losses, train_accs, val_losses, val_accs, val_aucs)
        
        logging.info("🎉 基准联邦学习训练完成!")
        return global_model

    def federated_averaging(self, global_weights, local_weights):
        """联邦平均算法"""
        # 计算平均权重
        avg_weights = copy.deepcopy(global_weights)
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for local_weight in local_weights:
                avg_weights[key] += local_weight[key]
            avg_weights[key] = avg_weights[key] / len(local_weights)
        
        return avg_weights

    def save_results(self, train_losses, train_accs, val_losses, val_accs, val_aucs):
        """保存训练结果"""
        # 创建结果目录
        save_dir = os.path.join(self.args.save_model_dir, self.args.model_name, self.args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建结果数据
        results = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'val_auc': val_aucs
        }
        
        # 保存为CSV
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        csv_filename = f'{timestamp}_baseline_{self.args.dataset}_{self.args.model_name}_results.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        logging.info(f"📁 训练结果已保存: {csv_path}")
        
        # 保存最佳模型
        best_epoch = np.argmax(val_aucs)
        best_auc = val_aucs[best_epoch]
        
        model_filename = f'{timestamp}_baseline_{self.args.dataset}_{self.args.model_name}_best_auc_{best_auc:.4f}.pth'
        model_path = os.path.join(save_dir, model_filename)
        
        # 这里可以保存模型权重
        logging.info(f"🏆 最佳模型 (AUC: {best_auc:.4f}, 轮次: {best_epoch + 1})")
        logging.info(f"📁 模型路径: {model_path}")


def main():
    """主函数"""
    args = parser_args()
    
    # 强制设置为基准模式
    args.enable_watermark = False
    args.watermark_mode = 'baseline'
    
    logging.info("🚀 启动基准联邦学习训练脚本")
    logging.info(f"📋 配置参数:")
    logging.info(f"  - 数据集: {args.dataset}")
    logging.info(f"  - 模型: {args.model_name}")
    logging.info(f"  - 客户端数: {args.client_num}")
    logging.info(f"  - 训练轮次: {args.epochs}")
    logging.info(f"  - 学习率: {args.lr}")
    logging.info(f"  - 本地轮次: {args.local_ep}")
    logging.info(f"  - 批次大小: {args.batch_size}")
    logging.info(f"  - 差分隐私: {'启用' if args.dp else '禁用'}")
    if args.dp:
        logging.info(f"  - 噪声参数: {args.sigma}")
    
    # 创建实验实例
    experiment = BaselineFederatedLearning(args)
    
    # 开始训练
    start_time = time.time()
    try:
        model = experiment.train()
        end_time = time.time()
        
        logging.info("=" * 60)
        logging.info("🎉 基准联邦学习训练成功完成!")
        logging.info(f"⏱️ 总训练时间: {end_time - start_time:.2f} 秒")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"❌ 训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()

