from utils.config import Config
from utils.misc import init_env


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
