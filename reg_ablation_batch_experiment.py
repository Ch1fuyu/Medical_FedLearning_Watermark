import copy
import itertools
import logging
import os
import sys
from datetime import datetime

from utils.args import parser_args
import reg_ablation_experiment as reg_exp


DATASETS = ['chestmnist', 'cifar10', 'cifar100']
MODELS = ['alexnet', 'resnet']
CLIENT_COUNTS = [5, 10, 20]


def _build_combinations():
    return list(itertools.product(DATASETS, MODELS, CLIENT_COUNTS))


def _clone_args(base_args, dataset, model_name, client_num):
    run_args = copy.deepcopy(base_args)
    run_args.dataset = dataset
    run_args.model_name = model_name
    run_args.client_num = client_num

    from utils.key_matrix_utils import get_key_matrix_path

    run_args.key_matrix_path = get_key_matrix_path(run_args.key_matrix_dir, run_args.model_name, run_args.client_num)
    return run_args


def _refresh_dataset_dependent_fields(run_args):
    dataset_presets = {
        'chestmnist': {
            'task_type': 'multilabel',
            'num_classes': 14,
            'in_channels': 1,
            'input_size': 28,
            'normalize_mean': [0.5],
            'normalize_std': [0.5],
            'default_batch_size': 128,
            'metrics': ['loss', 'acc_label', 'acc_sample', 'auc'],
        },
        'cifar10': {
            'task_type': 'multiclass',
            'num_classes': 10,
            'in_channels': 3,
            'input_size': 32,
            'normalize_mean': [0.4914, 0.4822, 0.4465],
            'normalize_std': [0.2470, 0.2435, 0.2616],
            'default_batch_size': 128,
            'metrics': ['loss', 'top1'],
        },
        'cifar100': {
            'task_type': 'multiclass',
            'num_classes': 100,
            'in_channels': 3,
            'input_size': 32,
            'normalize_mean': [0.5071, 0.4867, 0.4408],
            'normalize_std': [0.2675, 0.2565, 0.2761],
            'default_batch_size': 128,
            'metrics': ['loss', 'top1'],
        },
    }
    model_presets = {
        'resnet': {'default_variant': 'cifar'},
        'alexnet': {'default_variant': 'imagenet'},
    }

    ds_cfg = dataset_presets[run_args.dataset]
    run_args.num_classes = ds_cfg['num_classes']
    run_args.in_channels = ds_cfg['in_channels']
    run_args.task_type = ds_cfg['task_type']
    run_args.input_size = ds_cfg['input_size']
    run_args.normalize_mean = ds_cfg['normalize_mean']
    run_args.normalize_std = ds_cfg['normalize_std']
    run_args.metrics = ds_cfg['metrics']
    run_args.model_variant = model_presets[run_args.model_name]['default_variant']


def _configure_logging(base_log_file):
    log_file_name = base_log_file.replace('.logs', '_reg_ablation_batch.logs')
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H-%M-%S'
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    return log_file_name


def main():
    args = parser_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    batch_log_file = _configure_logging(args.log_file)
    combinations = _build_combinations()

    logging.info('=' * 72)
    logging.info('启动 reg_ablation_experiment 批量实验')
    logging.info(f'总组合数: {len(combinations)}')
    logging.info(f'组合列表: {combinations}')
    logging.info(f'批量日志文件: {batch_log_file}')
    logging.info('=' * 72)

    results = []
    total = len(combinations)

    for idx, (dataset, model_name, client_num) in enumerate(combinations, start=1):
        run_args = _clone_args(args, dataset, model_name, client_num)
        _refresh_dataset_dependent_fields(run_args)

        logging.info('-' * 72)
        logging.info(
            f'开始实验 {idx}/{total}: dataset={dataset}, model={model_name}, client_num={client_num}'
        )
        logging.info('-' * 72)

        start_time = datetime.now()
        status = 'success'
        error_message = ''

        try:
            reg_exp.main(run_args)
        except Exception as exc:
            status = 'failed'
            error_message = str(exc)
            logging.exception(
                f'实验失败: dataset={dataset}, model={model_name}, client_num={client_num}'
            )

        end_time = datetime.now()
        results.append({
            'dataset': dataset,
            'model_name': model_name,
            'client_num': client_num,
            'status': status,
            'error': error_message,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': (end_time - start_time).total_seconds(),
        })

    logging.info('=' * 72)
    logging.info('批量实验结束，结果汇总如下：')
    for item in results:
        logging.info(item)
    logging.info('=' * 72)


if __name__ == '__main__':
    main()
