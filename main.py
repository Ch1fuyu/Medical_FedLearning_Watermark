import os
import sys
import time
from datetime import datetime
import logging
import gc

# 强制 UTF-8 编码以支持 emoji 输出（解决 Windows GBK 问题）
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.globals import set_seed
from models.alexnet import AlexNet
from models.resnet import resnet18
from utils.args import parser_args
from utils.base import Experiment
from utils.dataset import get_data, get_data_no_fl
from utils.trainer_private_enhanced import TrainerPrivateEnhanced
import pandas as pd

set_seed()

# 配置 logging
args = parser_args()
log_file_name = args.log_file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
    ]
)


class StandaloneTrainer(Experiment):
    """单机训练类，包含水印嵌入和 MultiLoss 正则项"""
    
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self.dp = args.dp
        self.sigma = args.sigma
        
        logging.info('--------------------------------Start--------------------------------------')
        logging.info(args)
        logging.info('==> Preparing data...')
        
        # 数据集配置
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels
            
        # 确保数据根目录存在
        os.makedirs(self.data_root, exist_ok=True)

        # 使用完整训练集
        self.train_set, self.test_set = get_data_no_fl(
            dataset_name=self.dataset,
            data_root=self.data_root,
        )
        logging.info('==> Training model...')
        
        self.logs = {
            'best_train_acc': -np.inf, 
            'best_train_loss': -np.inf,
            'val_acc': [], 
            'val_loss': [],
            'best_model_acc': -np.inf, 
            'best_model_loss': -np.inf,
            'best_model_auc': -np.inf,
            'best_model': [],
            'highest_acc_ever': -np.inf,
            'highest_auc_ever': -np.inf,
            'acc_when_highest_auc': -np.inf,
            'auc_when_highest_acc': -np.inf,
        }

        self.construct_model()

        # 使用增强训练器（水印嵌入 + MultiLoss）
        logging.info('==> 使用增强水印系统（密钥矩阵 + MultiLoss 正则项）')
        self.trainer = TrainerPrivateEnhanced(self.model, self.device, self.dp, self.sigma, self.args)
        self.tester = self.trainer.tester

    def construct_model(self):
        if self.model_name == 'resnet':
            model = resnet18(
                num_classes=self.num_classes, 
                in_channels=self.in_channels, 
                input_size=self.args.input_size
            )
        else:
            dropout_rate = getattr(self.args, 'dropout_rate', 0.5)
            model = AlexNet(
                self.in_channels, 
                self.num_classes, 
                input_size=self.args.input_size, 
                dropout_rate=dropout_rate
            )
        self.model = model.to(self.device)

    def _cleanup_memory(self):
        """清理内存和GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _optimize_model_storage(self, model_state):
        """优化模型状态存储"""
        optimized_state = {}
        for key, value in model_state.items():
            optimized_state[key] = value.detach().cpu()
        return optimized_state

    def training(self):
        start = time.time()
        
        # 创建数据加载器
        train_ldr = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=False
        )
        val_ldr = DataLoader(
            self.test_set, 
            batch_size=self.batch_size * 2, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=False
        )

        # 决定模型选择依据
        dataset_key = (self.dataset or '').lower()
        select_by_auc = (dataset_key == 'chestmnist')

        # 早停配置
        use_early_stopping = self.args.patience > 0
        early_stop_counter = 0
        best_val_acc = -np.inf
        best_val_auc = -np.inf

        # 统计记录
        stats_rows = []

        for epoch in range(self.epochs):
            logging.info('Epoch: %d / %d' % (epoch + 1, self.epochs))
            
            # 本地更新
            model_state, train_loss, train_acc = self.trainer.local_update(
                dataloader=train_ldr,
                local_ep=1,
                lr=self.lr,
                current_epoch=epoch,
                total_epochs=self.epochs
            )
            
            # 更新模型状态
            self.model.load_state_dict(model_state)

            # 梯度统计（每10轮打印一次）
            if epoch >= 0 and hasattr(self.trainer, 'multi_loss') and (epoch + 1) % 10 == 0:
                try:
                    stats = self.trainer.get_gradient_stats()
                    if stats:
                        logging.info(
                            f'轮次{epoch+1}梯度统计 - GM:{stats.get("prevGM", 0):.6f} '
                            f'GH:{stats.get("prevGH", 0):.6f} Ratio:{stats.get("prevRatio", 1):.6f}'
                        )
                except Exception:
                    pass

            # 每轮训练后评估
            if (epoch + 1) % 1 == 0:
                train_metrics = self.trainer.test(train_ldr)
                val_metrics = self.trainer.test(val_ldr)

                loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean = train_metrics
                loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean = val_metrics

                self.logs['val_acc'].append(acc_val_label_mean)
                self.logs['val_loss'].append(loss_val_mean)

                # 更新历史最高值
                if self.logs['highest_acc_ever'] < acc_val_label_mean:
                    self.logs['highest_acc_ever'] = acc_val_label_mean
                    self.logs['auc_when_highest_acc'] = auc_val
                    
                if self.logs['highest_auc_ever'] < auc_val:
                    self.logs['highest_auc_ever'] = auc_val
                    self.logs['acc_when_highest_auc'] = acc_val_label_mean

                # 模型选择标准
                if select_by_auc:
                    if self.logs['best_model_auc'] < auc_val:
                        self.logs['best_model_acc'] = acc_val_label_mean
                        self.logs['best_model_loss'] = loss_val_mean
                        self.logs['best_model_auc'] = auc_val
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

                # 打印训练和验证指标
                logging.info(
                    f"轮次{epoch+1} | Train Loss:{loss_train_mean:.4f} Acc:{acc_train_label_mean:.4f} "
                    f"AUC:{auc_train:.4f} | Val Loss:{loss_val_mean:.4f} Acc:{acc_val_label_mean:.4f} "
                    f"AUC:{auc_val:.4f} | Best Acc:{self.logs['highest_acc_ever']:.4f} "
                    f"Best AUC:{self.logs['highest_auc_ever']:.4f}"
                )
                
                # MultiLoss 统计信息（每10轮打印）
                if hasattr(self.trainer, 'multi_loss') and (epoch + 1) % 10 == 0:
                    stats = self.trainer.multi_loss.get_stats()
                    logging.info(
                        f"MultiLoss - GM:{stats['prevGM']:.6f} GH:{stats['prevGH']:.6f} "
                        f"Ratio:{stats['prevRatio']:.6f}"
                    )
                
                # 记录统计数据
                stats_row = {
                    'round': epoch + 1,
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
                
                # 添加 MultiLoss 统计信息
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
                        'main_loss': float(multi_loss_stats['main_loss']),
                        'reg1_value': float(multi_loss_stats['reg1_value']),
                        'reg2_value': float(multi_loss_stats['reg2_value']),
                        'reg3_value': float(multi_loss_stats['reg3_value']),
                    })
                
                stats_rows.append(stats_row)
                
                # 早停检查
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
                
                # 清理内存
                self._cleanup_memory()
                
                # 清理临时变量
                del train_metrics, val_metrics
                del loss_train_mean, acc_train_label_mean, auc_train, acc_train_sample_mean
                del loss_val_mean, acc_val_label_mean, auc_val, acc_val_sample_mean

        logging.info('='*60 + ' 训练结果 ' + '='*60)
        logging.info(
            f'最佳模型 | Loss:{self.logs["best_model_loss"]:.4f} '
            f'Acc:{self.logs["best_model_acc"]:.4f} AUC:{self.logs["best_model_auc"]:.4f}'
        )
        logging.info(
            f'历史最高 | Acc:{self.logs["highest_acc_ever"]:.4f}(AUC:{self.logs["auc_when_highest_acc"]:.4f}) | '
            f'AUC:{self.logs["highest_auc_ever"]:.4f}(Acc:{self.logs["acc_when_highest_auc"]:.4f})'
        )
        end = time.time()
        logging.info(f'训练耗时: {(end - start) / 60:.1f} 分钟')
        
        # 清理内存
        self._cleanup_memory()
        logging.info('🧹 训练完成，已清理内存缓存')

        # 导出 Excel
        try:
            os.makedirs(self.args.save_excel_dir, exist_ok=True)
            columns = [
                'round', 'train_loss', 'val_loss', 'train_acc_label', 'train_auc',
                'val_acc_label', 'val_auc', 'best_val_acc_so_far', 'best_val_auc_so_far',
                'train_acc_sample', 'val_acc_sample', 'prevGM', 'prevGH', 'prevRatio',
                'current_grad_M', 'current_grad_H', 'current_var_M', 'current_var_H',
                'main_loss', 'reg1_value', 'reg2_value', 'reg3_value'
            ]
            df = pd.DataFrame(stats_rows, columns=columns)
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            excel_path = f'{self.args.save_excel_dir}/metrics_{self.model_name}_{self.dataset}_{now}.xlsx'
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logging.info(f'Excel metrics saved to: {excel_path}')
        except Exception as e:
            logging.warning(f'Failed to export Excel metrics: {e}')

        # 返回指标信息
        best_metric_value = self.logs['best_model_auc'] if select_by_auc else self.logs['best_model_acc']
        best_metric_name = 'auc' if select_by_auc else 'acc'
        return self.logs, best_metric_value, best_metric_name


def main():
    logs = {
        'net_info': None,
        'arguments': {
            'lr_inner': args.lr,
            'wd': args.wd,
            'optim': args.optim,
            'model_name': args.model_name,
            'dataset': args.dataset,
            'num_classes': args.num_classes,
            'epochs': args.epochs,
            'console_log': os.path.basename(log_file_name),
        }
    }
    
    logging.info('==> 开始单机训练')
    trainer = StandaloneTrainer(args)
    logg, best_metric_value, best_metric_name = trainer.training()
    
    logs['net_info'] = logg
    logs['test_auc'] = {'value': logg.get('best_model_auc', best_metric_value if best_metric_name == 'auc' else 0.0)}
    if best_metric_name == 'acc':
        logs['test_acc'] = {'value': best_metric_value}

    save_dir = os.path.join(args.save_model_dir, args.model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M")
    
    # 文件命名
    watermark_suffix = f"wm_{args.watermark_mode}" if hasattr(args, 'watermark_mode') and args.watermark_mode else "wm_basic"
    file_name = '{}_Dp_{}_{}_ep_{{:.4f}}.pkl'.format(
        formatted_now, args.sigma, watermark_suffix,
        args.epochs
    )
    file_name = file_name.format(best_metric_value)
    torch.save(logs, os.path.join(save_dir, file_name))
    logging.info(f"训练日志已保存: {file_name}")
    logging.info('-------------------------------Finish--------------------------------------')

    return


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main()
