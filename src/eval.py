import torch
import torch.utils.data

from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.config import Config
from utils.model import load_model
from utils.misc import load_dataset


def eval(cfg):
    dataset = load_dataset(cfg.dataset)('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)
    Config().print(cfg)

    aps = eval_dataset(dataset, cfg.load_model, cfg)
    for k, v in aps.items():
        print('{:<20} {:.3f}'.format(k, v))

    torch.cuda.empty_cache()


def eval_dataset(dataset, model, cfg):
    # model = SqueezeDet(cfg)
    # model = load_model(model, model_path, cfg)
    detect = model.detect
    model.detect = True
    detector = Detector(model, cfg)

    results = detector.detect_dataset(dataset, cfg)
    dataset.save_results(results)
    aps = dataset.evaluate()
    model.detect = detect
    return aps
