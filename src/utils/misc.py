import os

import numpy as np
import torch

EPSILON = 1E-10


def init_env(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = not cfg.not_cuda_benchmark
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus_str
    cfg.device = torch.device('cuda' if cfg.gpus[0] >= 0 else 'cpu')

    return cfg


def load_dataset(dataset_name):
    if dataset_name.lower() == 'kitti':
        from datasets.kitti import KITTI as Dataset
    elif dataset_name.lower() == 'coco':
        from datasets.coco import COCO as Dataset
    else:
        raise ValueError('invalid dataset name.')
    return Dataset


class MetricLogger(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + EPSILON)
