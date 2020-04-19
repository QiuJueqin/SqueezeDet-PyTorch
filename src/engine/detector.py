import os
import time

import numpy as np
import torch
import torch.utils.data
from torchvision.ops import nms

from utils.image import image_postprocess
from utils.boxes import boxes_postprocess, visualize_boxes
from utils.misc import MetricLogger


class Detector(object):
    def __init__(self, model, cfg):
        self.model = model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg

    @torch.no_grad()
    def detect(self, batch):
        dets = self.model(batch)

        results = []
        batch_size = dets['class_ids'].shape[0]
        for b in range(batch_size):
            image_meta = {k: v[b].cpu().numpy() if not isinstance(v, list) else v[b]
                          for k, v in batch['image_meta'].items()}

            det = {k: v[b] for k, v in dets.items()}
            det = self.filter(det)

            if det is None:
                results.append({'image_meta': image_meta})
                continue

            det = {k: v.cpu().numpy() for k, v in det.items()}
            det['boxes'] = boxes_postprocess(det['boxes'], image_meta)
            det['image_meta'] = image_meta
            results.append(det)

            if self.cfg.debug == 2:
                image = image_postprocess(batch['image'][b].cpu().numpy().transpose(1, 2, 0), image_meta)
                save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
                visualize_boxes(image, det['class_ids'], det['boxes'], det['scores'],
                                class_names=self.cfg.class_names,
                                save_path=save_path,
                                show=self.cfg.mode == 'demo')

        return results

    def detect_dataset(self, dataset):
        start_time = time.time()

        data_loader = torch.utils.data.DataLoader(DataWrapper(dataset),
                                                  batch_size=self.cfg.batch_size,
                                                  num_workers=self.cfg.num_workers,
                                                  pin_memory=True)
        num_iters = len(data_loader)
        data_timer, net_timer = MetricLogger(), MetricLogger()
        end = time.time()

        results = []
        for iter_id, batch in enumerate(data_loader):
            for k in batch:
                if 'image_meta' not in k:
                    batch[k] = batch[k].to(device=self.cfg.device, non_blocking=True)
            data_timer.update(time.time() - end)
            end = time.time()

            results.extend(self.detect(batch))

            net_timer.update(time.time() - end)
            end = time.time()
            if iter_id % self.cfg.print_interval == 0:
                print('eval: [{0}/{1}] | data {2:.3f}s | net {3:.3f}s'.format(
                    iter_id, num_iters, data_timer.val, net_timer.val))

        total_time = time.time() - start_time
        tpi = total_time / len(dataset)
        print('Elapsed {:.2f}min ({:.1f}ms/image, {:.1f}frames/s)'.format(
            total_time / 60., tpi * 1000., 1 / tpi))
        print('-' * 80)

        return results

    def filter(self, det):
        orders = torch.argsort(det['scores'], descending=True)[:self.cfg.keep_top_k]
        class_ids = det['class_ids'][orders]
        scores = det['scores'][orders]
        boxes = det['boxes'][orders, :]

        # class-wise nms
        filtered_class_ids, filtered_scores, filtered_boxes = [], [], []
        for cls_id in range(self.cfg.num_classes):
            idx_cur_class = (class_ids == cls_id)
            if torch.sum(idx_cur_class) == 0:
                continue

            class_ids_cur_class = class_ids[idx_cur_class]
            scores_cur_class = scores[idx_cur_class]
            boxes_cur_class = boxes[idx_cur_class, :]

            keeps = nms(boxes_cur_class, scores_cur_class, self.cfg.nms_thresh)

            filtered_class_ids.append(class_ids_cur_class[keeps])
            filtered_scores.append(scores_cur_class[keeps])
            filtered_boxes.append(boxes_cur_class[keeps, :])

        filtered_class_ids = torch.cat(filtered_class_ids)
        filtered_scores = torch.cat(filtered_scores)
        filtered_boxes = torch.cat(filtered_boxes, dim=0)

        keeps = filtered_scores > self.cfg.score_thresh
        if torch.sum(keeps) == 0:
            det = None
        else:
            det = {'class_ids': filtered_class_ids[keeps],
                   'scores': filtered_scores[keeps],
                   'boxes': filtered_boxes[keeps, :]}

        return det


class DataWrapper(torch.utils.data.Dataset):
    """ A wrapper of Dataset class that bypasses loading annotations """

    def __init__(self, dataset):
        super(DataWrapper, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        image, image_id = self.dataset.load_image(index)
        image_meta = {'index': index,
                      'image_id': image_id,
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, image_meta, _ = self.dataset.preprocess(image, image_meta)

        batch = {'image': image.transpose(2, 0, 1),
                 'image_meta': image_meta}
        return batch

    def __len__(self):
        return len(self.dataset)
