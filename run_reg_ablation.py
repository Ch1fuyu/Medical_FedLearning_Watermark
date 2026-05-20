"""
运行正则项消融实验的脚本
实验1: 只使用 reg3
实验2: 使用所有 reg (reg1+reg2+reg3)
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
log_dir = './logs_reg_ablation'
os.makedirs(log_dir, exist_ok=True)

# 实验配置
EXPERIMENTS = [
    {
        'name': 'reg3_only',
        'reg_config': {'reg1': False, 'reg2': False, 'reg3': True},
        'description': '只使用 reg3 (自适应权重)'
    },
    {
        'name': 'all_reg',
        'reg_config': {'reg1': True, 'reg2': True, 'reg3': True},
        'description': '使用所有正则项 (reg1+reg2+reg3)'
    },
]

# 基础参数（可以根据需要修改）
BASE_ARGS = {
    'dataset': 'chestmnist',
    'model_name': 'alexnet',
    'epochs': 100,
    'client_num': 10,
    'local_ep': 5,
    'lr': 0.01,
    'momentum': 0.9,
    'optim': 'adam',
    'wd': 0.0001,
    'batch_size': 32,
    'iid': 1,
    'unequal': 0,
    'num_classes': 14,
    'dp': 0.0,
    'sigma': 0.0,
    'watermark_mode': 'basic',
    'save_excel_dir': './save/excel_results',
    'key_matrix_dir': './save/key_matrix',
    'encoder_path': './save/autoencoder/autoencoder_chestmnist_alexnet.pth',
    'save_dir': './save',
    'use_multiloss': True,
    'multiloss_alpha_early': 0.001,
    'multiloss_alpha_late': 0.1,
    'use_lr_scheduler': True,
    'select_by_auc': True,
}


def run_experiment(exp_config, exp_index):
    """运行单个实验"""
    from utils.args import Args
    from reg_ablation_experiment import RegAblationExperiment
    
    exp_name = exp_config['name']
    reg_config = exp_config['reg_config']
    
    print("\n" + "="*70)
    print(f"实验 {exp_index}/2: {exp_name}")
    print(f"描述: {exp_config['description']}")
    print(f"正则项配置: {reg_config}")
    print("="*70)
    
    # 创建实验目录
    exp_log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(exp_log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 创建参数对象
    args = Args()
    for key, value in BASE_ARGS.items():
        setattr(args, key, value)
    
    # 设置正则项配置（通过修改类的配置）
    RegAblationExperiment.REG_CONFIG = reg_config
    
    # 运行实验
    try:
        exp = RegAblationExperiment(args)
        results = exp.run()
        logging.info(f"实验 {exp_name} 完成!")
        return True, results
    except Exception as e:
        logging.error(f"实验 {exp_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """主函数"""
    print("\n" + "="*70)
    print("开始正则项消融实验")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        success, result = run_experiment(exp_config, i)
        results[exp_config['name']] = {
            'success': success,
            'result': result
        }
        
        if not success:
            print(f"\n警告: 实验 {exp_config['name']} 失败，继续下一个实验...")
    
    # 汇总结果
    print("\n" + "="*70)
    print("实验汇总")
    print("="*70)
    
    for exp_name, exp_result in results.items():
        status = "成功" if exp_result['success'] else "失败"
        print(f"  {exp_name}: {status}")
    
    print("\n所有实验完成!")
    print(f"日志文件保存在: {os.path.abspath(log_dir)}")


if __name__ == '__main__':
    main()
