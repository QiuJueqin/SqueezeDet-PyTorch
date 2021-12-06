import os
import operator
import copy
import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from engine.trainer import Trainer
from model.squeezedet import SqueezeDetWithLoss
from utils.config import Config
from utils.model import load_model, load_official_model, save_model
from utils.logger import Logger
from utils.misc import load_dataset
from eval import eval_dataset


def train(cfg):
    Dataset = load_dataset(cfg.dataset)
    train_dataset = Dataset('train', cfg)
    val_dataset = Dataset('val', cfg)
    cfg = Config().update_dataset_info(cfg, train_dataset)
    Config().print(cfg)
    logger = Logger(cfg)

    model = SqueezeDetWithLoss(cfg)
    if cfg.load_model != '':
        if cfg.load_model.endswith('f364aa15.pth') or cfg.load_model.endswith('a815701f.pth'):
            model = load_official_model(model, cfg.load_model, cfg)
        else:
            model = load_model(model, cfg.load_model, cfg)
    # qat specific
    if cfg.qat:
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        fused_model = copy.deepcopy(model)
        fused_model.fuse_model()
        assert model_equivalence(model_1=model, model_2=fused_model, device='cpu', rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,cfg.input_size[0],cfg.input_size[1])), "Fused model is not equivalent to the original model!"
        model = torch.quantization.prepare_qat(fused_model)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=cfg.lr,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=cfg.weight_decay)

    lr_scheduler = StepLR(optimizer, 150, gamma=0.5)

    trainer = Trainer(model, optimizer, lr_scheduler, cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               shuffle=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)

    metrics = trainer.metrics if cfg.no_eval else trainer.metrics + ['mAP']
    best = 1E9 if cfg.no_eval else 0
    better_than = operator.lt if cfg.no_eval else operator.gt

    for epoch in range(1, cfg.num_epochs + 1):
        # qat specific
        if cfg.qat:
            if epoch == cfg.num_epochs // 3:
                print("freeze quantizer parameters")
                model.apply(torch.quantization.disable_observer)
            elif epoch == cfg.num_epochs // 3 * 2:
                print("freeze batch norm mean and variance estimates")
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        train_stats = trainer.train_epoch(epoch, train_loader, cfg=cfg)
        logger.update(train_stats, phase='train', epoch=epoch, cfg=cfg)

        save_path = os.path.join(cfg.save_dir, 'model_last.pth')
        save_model(model, save_path, epoch)

        if epoch % cfg.save_intervals == 0:
            save_path = os.path.join(cfg.save_dir, 'model_{}.pth'.format(epoch))
            save_model(model, save_path, epoch)

        if cfg.val_intervals > 0 and epoch % cfg.val_intervals == 0:
            val_stats = trainer.val_epoch(epoch, val_loader, cfg=cfg)
            logger.update(val_stats, phase='val', epoch=epoch, cfg=cfg)

            if not cfg.no_eval:
                aps = eval_dataset(val_dataset, model, cfg)
                logger.update(aps, phase='val', epoch=epoch, cfg=cfg)

            value = val_stats['loss'] if cfg.no_eval else aps['mAP']
            if better_than(value, best):
                best = value
                save_path = os.path.join(cfg.save_dir, 'model_best.pth')
                save_model(model, save_path, epoch)

        logger.plot(metrics)
        logger.print_bests(metrics, cfg)

    torch.cuda.empty_cache()

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