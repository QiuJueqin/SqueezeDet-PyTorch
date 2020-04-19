from utils.config import Config
from utils.misc import init_env


cmd = ('eval '
       # '--load_model ../models/imagenet/squeezenet1_1-f364aa15.pth '
       '--load_model ../models/squeezedet_kitti_epoch290.pth '
       # '--load_model /home/qiujueqin/PycharmProjects/squeezedet/exp/tmp/model_35.pth '
       '--batch_size 20 '
       '--num_workers 2 '
       # '--val_intervals 10 '
       '--exp_id tmp').split(' ')

cfg = Config().parse()
init_env(cfg)

if cfg.mode == 'train':
    from train import train
    train(cfg)
elif cfg.mode == 'eval':
    from eval import eval
    eval(cfg)
elif cfg.mode == 'demo':
    from demo import demo
    demo(cfg)
else:
    raise ValueError('Mode {} is invalid.'.format(cfg.mode))
