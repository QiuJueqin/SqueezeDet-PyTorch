import argparse
import os


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('mode',
                                 help='train | eval | demo')
        self.parser.add_argument('--dataset', default='kitti',
                                 help='coco | kitti')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pre-trained model')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='0: show nothing\n'
                                      '1: visualize pre-processed image and boxes\n'
                                      '2: visualize detections.')
        self.parser.add_argument('--exp_id', default='default')

        # model
        self.parser.add_argument('--arch', default='squeezedet',
                                 help='model architecture: squeezedet | squeezedetplus')
        self.parser.add_argument('--dropout_prob', type=float, default=0.5,
                                 help='probability of dropout.')

        # train
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--momentum', type=float, default=0.9,
                                 help='momentum of SGD.')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001,
                                 help='weight decay of SGD.')
        self.parser.add_argument('--grad_norm', type=float, default=5.,
                                 help='max norm of the gradients.')
        self.parser.add_argument('--num_epochs', type=int, default=300,
                                 help='total training epochs.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--batch_size', type=int, default=20,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--save_intervals', type=int, default=1,
                                 help='number of epochs to save model.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--no_eval', action='store_true',
                                 help='bypass mAP evaluation during training.')
        self.parser.add_argument('--print_interval', type=int, default=10,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--flip_prob', type=float, default=0.5,
                                 help='probability of horizontal flip during training.')
        self.parser.add_argument('--drift_prob', type=float, default=1.,
                                 help='probability of drifting image during training.')
        self.parser.add_argument('--forbid_resize', action='store_true',
                                 help='disable image resizing during training, '
                                      'use crop/pad instead.')
        self.parser.add_argument('--class_loss_weight', type=float, default=1.,
                                 help='weight of classification loss.')
        self.parser.add_argument('--positive_score_loss_weight', type=float, default=3.75,
                                 help='positive weight of score prediction loss.')
        self.parser.add_argument('--negative_score_loss_weight', type=float, default=100.,
                                 help='negative weight of score prediction loss.')
        self.parser.add_argument('--bbox_loss_weight', type=float, default=6.,
                                 help='weight of boxes regression loss.')

        # inference
        self.parser.add_argument('--nms_thresh', type=float, default=0.4,
                                 help='discards all overlapping boxes with IoU < nms_thresh.')
        self.parser.add_argument('--score_thresh', type=float, default=0.3,
                                 help='discards all boxes with scores smaller than score_thresh.')
        self.parser.add_argument('--keep_top_k', type=int, default=64,
                                 help='keep top k detections before nms.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=42,
                                 help='random seed')

    def parse(self, args=''):
        if args == '':
            cfg = self.parser.parse_args()
        else:
            cfg = self.parser.parse_args(args)

        cfg.gpus_str = cfg.gpus
        cfg.gpus = [int(gpu) for gpu in cfg.gpus.split(',')]
        cfg.gpus = [i for i in range(len(cfg.gpus))] if cfg.gpus[0] >= 0 else [-1]

        if cfg.mode != 'train' and len(cfg.gpus) > 1:
            print('Only single GPU is supported in {} mode.'.format(cfg.mode))
            cfg.gpus = [cfg.gpus[0]]
            cfg.master_batch_size = -1

        if cfg.master_batch_size == -1:
            cfg.master_batch_size = cfg.batch_size // len(cfg.gpus)
        rest_batch_size = (cfg.batch_size - cfg.master_batch_size)
        cfg.chunk_sizes = [cfg.master_batch_size]
        for i in range(len(cfg.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(cfg.gpus) - 1)
            if i < rest_batch_size % (len(cfg.gpus) - 1):
                slave_chunk_size += 1
            cfg.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', cfg.chunk_sizes)

        cfg.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cfg.data_dir = os.path.join(cfg.root_dir, 'data')
        cfg.exp_dir = os.path.join(cfg.root_dir, 'exp')
        cfg.save_dir = os.path.join(cfg.exp_dir, cfg.exp_id)
        cfg.debug_dir = os.path.join(cfg.save_dir, 'debug')
        print('The results will be saved to ', cfg.save_dir)
        return cfg

    @staticmethod
    def update_dataset_info(cfg, dataset):
        cfg.input_size = dataset.input_size
        cfg.rgb_mean = dataset.rgb_mean
        cfg.rgb_std = dataset.rgb_std
        cfg.class_names = dataset.class_names
        cfg.num_classes = dataset.num_classes
        cfg.anchors = dataset.anchors
        cfg.anchors_per_grid = dataset.anchors_per_grid
        cfg.num_anchors = dataset.num_anchors
        return cfg

    @staticmethod
    def print(cfg):
        names = list(dir(cfg))
        for name in sorted(names):
            if not name.startswith('_'):
                print('{:<30} {}'.format(name, getattr(cfg, name)))
