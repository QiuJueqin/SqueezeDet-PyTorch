import torch
import torch.utils.data
import copy
import numpy as np
from engine.detector import Detector
from model.squeezedet import SqueezeDetWithLoss
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
    if cfg.mode=='eval':
        model = SqueezeDetWithLoss(cfg)
        if cfg.qat:

            model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            fused_model = copy.deepcopy(model)
            fused_model.fuse_model()
            assert model_equivalence(model_1=model, model_2=fused_model, device='cpu', rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,cfg.input_size[0],cfg.input_size[1])), "Fused model is not equivalent to the original model!"
            model = torch.quantization.prepare_qat(fused_model)

        model = load_model(model, cfg.load_model, cfg)
    detect = model.detect
    model.detect = True
    detector = Detector(model, cfg)

    results = detector.detect_dataset(dataset, cfg)
    dataset.save_results(results)
    aps = dataset.evaluate()
    model.detect = detect
    return aps


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)

        y1 = model_1.base(x).detach().cpu().numpy()
        y2 = model_2.base(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True