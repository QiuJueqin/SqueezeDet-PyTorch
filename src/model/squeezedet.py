import torch
import torch.nn as nn

from model.modules import deltas_to_boxes, compute_overlaps, safe_softmax

EPSILON = 1E-10


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        x = torch.cat([
            self.activation(self.expand1x1(x)),
            self.activation(self.expand3x3(x))
        ], dim=1)
        return x


class SqueezeDetBase(nn.Module):
    def __init__(self, cfg):
        super(SqueezeDetBase, self).__init__()
        self.num_classes = cfg.num_classes
        self.num_anchors = cfg.num_anchors

        if cfg.arch == 'squeezedet':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                Fire(512, 96, 384, 384),
                Fire(768, 96, 384, 384)
            )
        elif cfg.arch == 'squeezedetplus':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 96, 64, 64),
                Fire(128, 96, 64, 64),
                Fire(128, 192, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 192, 128, 128),
                Fire(256, 288, 192, 192),
                Fire(384, 288, 192, 192),
                Fire(384, 384, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 384, 256, 256),
                Fire(512, 384, 256, 256),
                Fire(512, 384, 256, 256),
            )
        else:
            raise ValueError('Invalid architecture.')

        self.dropout = nn.Dropout(cfg.dropout_prob, inplace=True) \
            if cfg.dropout_prob > 0 else None
        self.convdet = nn.Conv2d(768 if cfg.arch == 'squeezedet' else 512,
                                 cfg.anchors_per_grid * (cfg.num_classes + 5),
                                 kernel_size=3, padding=1)

        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.convdet(x)

        x = x.permute(0, 2, 3, 1).contiguous()

        return x.view(-1, self.num_anchors, self.num_classes + 5)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.convdet:
                    nn.init.normal_(m.weight, mean=0.0, std=0.002)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PredictionResolver(nn.Module):
    def __init__(self, cfg, log_softmax=False):
        super(PredictionResolver, self).__init__()
        self.log_softmax = log_softmax
        self.input_size = cfg.input_size
        self.num_classes = cfg.num_classes
        self.anchors = torch.from_numpy(cfg.anchors).unsqueeze(0).float()
        self.anchors_per_grid = cfg.anchors_per_grid

    def forward(self, pred):
        pred_class_probs = safe_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)
        pred_log_class_probs = None if not self.log_softmax else \
            torch.log_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)

        pred_scores = torch.sigmoid(pred[..., self.num_classes:self.num_classes + 1].contiguous())

        pred_deltas = pred[..., self.num_classes + 1:].contiguous()
        pred_boxes = deltas_to_boxes(pred_deltas, self.anchors.to(pred_deltas.device),
                                     input_size=self.input_size)

        return pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.resolver = PredictionResolver(cfg, log_softmax=True)
        self.num_anchors = cfg.num_anchors
        self.class_loss_weight = cfg.class_loss_weight
        self.positive_score_loss_weight = cfg.positive_score_loss_weight
        self.negative_score_loss_weight = cfg.negative_score_loss_weight
        self.bbox_loss_weight = cfg.bbox_loss_weight

    def forward(self, pred, gt):
        # slice gt tensor
        anchor_masks = gt[..., :1]
        gt_boxes = gt[..., 1:5]  # xyxy format
        gt_deltas = gt[..., 5:9]
        gt_class_logits = gt[..., 9:]

        # resolver predictions
        pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes = self.resolver(pred)

        num_objects = torch.sum(anchor_masks, dim=[1, 2])
        overlaps = compute_overlaps(gt_boxes, pred_boxes) * anchor_masks

        class_loss = torch.sum(
            self.class_loss_weight * anchor_masks * gt_class_logits * (-pred_log_class_probs),
            dim=[1, 2],
        ) / num_objects

        positive_score_loss = torch.sum(
            self.positive_score_loss_weight * anchor_masks * (overlaps - pred_scores) ** 2,
            dim=[1, 2]
        ) / num_objects

        negative_score_loss = torch.sum(
            self.negative_score_loss_weight * (1 - anchor_masks) * (overlaps - pred_scores) ** 2,
            dim=[1, 2]
        ) / (self.num_anchors - num_objects)

        bbox_loss = torch.sum(
            self.bbox_loss_weight * anchor_masks * (pred_deltas - gt_deltas) ** 2,
            dim=[1, 2],
        ) / num_objects

        loss = class_loss + positive_score_loss + negative_score_loss + bbox_loss
        loss_stat = {
            'loss': loss,
            'class_loss': class_loss,
            'score_loss': positive_score_loss + negative_score_loss,
            'bbox_loss': bbox_loss
        }

        return loss, loss_stat


class SqueezeDetWithLoss(nn.Module):
    """ Model for training """
    def __init__(self, cfg):
        super(SqueezeDetWithLoss, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.loss = Loss(cfg)

    def forward(self, batch):
        pred = self.base(batch['image'])
        loss, loss_stats = self.loss(pred, batch['gt'])
        return loss, loss_stats


class SqueezeDet(nn.Module):
    """ Model for inference """
    def __init__(self, cfg):
        super(SqueezeDet, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.resolver = PredictionResolver(cfg, log_softmax=False)

    def forward(self, batch):
        pred = self.base(batch['image'])
        pred_class_probs, _, pred_scores, _, pred_boxes = self.resolver(pred)
        pred_class_probs *= pred_scores
        pred_class_ids = torch.argmax(pred_class_probs, dim=2)
        pred_scores = torch.max(pred_class_probs, dim=2)[0]
        det = {'class_ids': pred_class_ids,
               'scores': pred_scores,
               'boxes': pred_boxes}
        return det
