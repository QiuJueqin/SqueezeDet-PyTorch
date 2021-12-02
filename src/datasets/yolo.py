import os
import subprocess

import numpy as np
import skimage.io

from datasets.base import BaseDataset
from utils.boxes import generate_anchors
from PIL import Image
from torchvision.datasets.folder import default_loader


class YOLO(BaseDataset):
    def __init__(self, phase, cfg):
        super(YOLO, self).__init__(phase, cfg)

        self.input_size = (128, 128)  # (height, width), both dividable by 16
        self.class_names = ('0')
        # real_filtered mean and std
        # self.rgb_mean = np.array([94.87347, 96.89165, 94.70493], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([53.869507, 53.936283, 55.2807], dtype=np.float32).reshape(1, 1, 3)
        
        # real_filtered plus all_sites_seatbelt mean and std
        # self.rgb_mean = np.array([104.90631, 105.41336, 104.70162], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([50.69564, 49.60443, 50.158844], dtype=np.float32).reshape(1, 1, 3)

        self.rgb_mean = np.array([97.631615, 98.70732, 98.41285], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([52.766678, 52.63513, 52.348827], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}

        self.data_dir = os.path.join(cfg.data_dir, 'lpr_crop/merged_data')
        self.sample_ids, self.sample_set_path = self.get_sample_ids()
        self.grid_size = tuple(x // 16 for x in self.input_size)  # anchors grid 
        # self.anchors_seed = np.array([[ 29, 17], [46, 32], [69, 52],
        #                                 [109, 68], [84, 127], [155, 106], 
        #                                 [255, 145], [183, 215], [371, 221]], dtype=np.float32) ## real_filtered anchors
        
        # self.anchors_seed = np.array( [[ 32, 20], [ 61, 42], [ 59, 97],
        #                                 [103, 66], [122, 114], [183, 96],
        #                                 [160, 152], [211, 201], [343, 205]], dtype=np.float32) ## real_filtered plus all_sites_seatbelt anchors

        self.anchors_seed = np.array([[6, 5], [12, 10], [18, 10], [18, 18], [20, 24], [30, 15]], dtype=np.float32)

        self.anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]

        self.results_dir = os.path.join(cfg.save_dir, 'results')

    def get_sample_ids(self):
        sample_set_name = 'train.txt' if self.phase == 'train' \
            else 'val.txt' if self.phase == 'val' \
            else 'trainval.txt' if self.phase == 'trainval' \
            else None

        sample_ids_path = os.path.join(self.data_dir, sample_set_name)
        with open(sample_ids_path, 'r') as fp:
            sample_ids = fp.readlines()
        sample_ids = tuple(x.strip() for x in sample_ids)

        return sample_ids, sample_ids_path

    def load_image(self, index):
        image_id = self.sample_ids[index]
        image_path = os.path.join(self.data_dir, 'images', image_id + '.png')
        image = default_loader(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image).astype(np.float32)
        # image = skimage.io.imread(image_path).astype(np.float32)
        return image, image_id

    def load_annotations(self, index):
        ann_id = self.sample_ids[index]
        ann_path = os.path.join(self.data_dir, 'labels', ann_id + '.txt')
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()

        annotations = [ann.strip().split(' ') for ann in annotations]
        class_ids, boxes = [], []
        for ann in annotations:
            if ann[0] not in self.class_names:
                continue
            class_ids.append(self.class_ids_dict[ann[0]])
            box = [float(x) for x in ann[4:8]]
            # if box[2] <= 0:
            #     box[2] = 0.00001
            # if box[3] <= 0:
            #     box[3] = 0.00001
            boxes.append(box)

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes):
            return class_ids, boxes
        boxes = None
        return class_ids, boxes

    # ========================================
    #                evaluation
    # ========================================

    def save_results(self, results):
        txt_dir = os.path.join(self.results_dir, 'data')
        os.makedirs(txt_dir, exist_ok=True)

        for res in results:
            txt_path = os.path.join(txt_dir, res['image_meta']['image_id'] + '.txt')
            if 'class_ids' not in res:
                with open(txt_path, 'w') as fp:
                    fp.write('')
                continue

            num_boxes = len(res['class_ids'])
            with open(txt_path, 'w') as fp:
                for i in range(num_boxes):
                    class_name = self.class_names[res['class_ids'][i]].lower()
                    score = res['scores'][i]
                    bbox = res['boxes'][i, :]
                    line = '{} -1 -1 0 {:.2f} {:.2f} {:.2f} {:.2f} 0 0 0 0 0 0 0 {:.3f}\n'.format(
                            class_name, *bbox, score)
                    fp.write(line)

    def evaluate(self):
        kitti_eval_tool_path = os.path.join(self.cfg.root_dir, 'src/utils/kitti-eval/cpp/evaluate_object')
        cmd = '{} {} {} {} {}'.format(kitti_eval_tool_path,
                                      os.path.join(self.data_dir),
                                      self.sample_set_path,
                                      self.results_dir,
                                      len(self.sample_ids))

        status = subprocess.call(cmd, shell=True)

        aps = {}
        for class_name in self.class_names:
            map_path = os.path.join(self.results_dir, 'stats_{}_ap.txt'.format(class_name.lower()))
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    lines = f.readlines()
                _aps = [float(line.split('=')[1].strip()) for line in lines]
            else:
                _aps = [0., 0., 0.]

            aps[class_name + '_easy'] = _aps[0]
            aps[class_name + '_moderate'] = _aps[1]
            aps[class_name + '_hard'] = _aps[2]

        aps['mAP'] = sum(aps.values()) / len(aps)

        return aps

