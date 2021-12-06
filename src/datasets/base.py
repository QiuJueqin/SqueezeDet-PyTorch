import os

import numpy as np
import torch.utils.data

from utils.image import whiten, drift, flip, resize, crop_or_pad
from utils.boxes import compute_deltas, visualize_boxes
import imgaug as ia
import imgaug.augmenters as iaa
import random
import torchvision.transforms as transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image,ImageDraw
import cv2


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, cfg):
        super(BaseDataset, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.seq = iaa.Sequential([
            iaa.SomeOf((0, 2),[
                iaa.Flipud(0.5),
                iaa.Fliplr(0.1),
            ]),
            # Perspective/Affine
            iaa.Sometimes(
                p=0.3,
                then_list=iaa.OneOf([
                        iaa.Affine(
                                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                rotate=(-20, 20),
                                shear=(-15, 15),
                                order=[0, 1],
                                cval=(0, 255),
                                mode=ia.ALL
                            ),
                        iaa.CropAndPad(percent=(-0.3, 0.3), pad_mode=ia.ALL),
                        iaa.PerspectiveTransform(scale=(0.02, 0.15)),
                        # iaa.PiecewiseAffine(scale=(0.01, 0.1)),
                    ])
            ),
            iaa.Sometimes(
                0.1,
                iaa.OneOf([
                    iaa.Rain(speed=(0.1, 0.2)),
                    iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))
                ])
            ),
            iaa.Sometimes(
                p=0.05,
                then_list=iaa.OneOf([
                # iaa.Cartoon(blur_ksize=3, segmentation_size=1.0, saturation=2.0, edge_prevalence=1.0),
                iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0)),
                iaa.Solarize(p=1),
                iaa.pillike.Posterize(nb_bits=(1, 8)),
                # iaa.RandAugment(m=(1, 5)),
                ])
            ),
            iaa.Sometimes(
                p=0.15,
                then_list=iaa.OneOf([
                    ## Smoothing
                    iaa.OneOf([
                        iaa.pillike.FilterSmooth(),
                        iaa.pillike.FilterSmoothMore()
                    ]),
                    ## Blurring
                    iaa.OneOf([
                        iaa.imgcorruptlike.DefocusBlur(severity=1),
                        iaa.imgcorruptlike.ZoomBlur(severity=1),
                        iaa.MotionBlur(k=(3, 15), angle=(0, 360),direction=(-1.0, 1.0)),
                        iaa.imgcorruptlike.MotionBlur(severity=1),
                        iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                    ]),
                    ## Edge Enhancement
                    iaa.OneOf([
                        iaa.pillike.FilterEdgeEnhance(),
                        iaa.pillike.FilterEdgeEnhanceMore(),
                        iaa.pillike.FilterContour(),
                        iaa.pillike.FilterDetail(),            
                    ])
                ])
            ),
        ],random_order=True)

    def __getitem__(self, index):
        image, image_id = self.load_image(index)
        gt_class_ids, gt_boxes = self.load_annotations(index)

        image_meta = {'index': index,
                      'image_id': image_id,
                      'orig_size': np.array(image.shape, dtype=np.int32)}
        
        image, image_visualize, image_meta, gt_boxes, gt_class_ids = self.preprocess(image, image_meta, gt_boxes, gt_class_ids)
        gt = self.prepare_annotations(gt_class_ids, gt_boxes)

        inp = {'image': image,
               'image_meta': image_meta,
               'gt': gt}

        if self.cfg.debug == 1:
            # image = image * image_meta['rgb_std'] + image_meta['rgb_mean']

            save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
            visualize_boxes(image_visualize, gt_class_ids, gt_boxes,
                            class_names=self.class_names,
                            save_path=save_path)

        return inp

    def __len__(self):
        return len(self.sample_ids)

    def preprocess(self, image, image_meta, boxes=None, class_ids=None):

        if self.cfg.forbid_resize:
            image, image_meta, boxes = crop_or_pad(image, image_meta, self.input_size, boxes=boxes)
        else:
            image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)    

        if self.phase == "train":
            prob = random.random()
            if boxes is not None:
                if prob < 0.7:
                    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
                    image = transforms.ColorJitter(brightness=(0.6,1.3), contrast=(0.6, 1.3),
                                            saturation=(0.6, 1.3), hue=(-0.3, 0.3)) (image)
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    boxes_aug = []
                    for box, label in zip(boxes, class_ids):
                        boxes_aug.append(BoundingBox(box[0],box[1],box[2],box[3], label=label))
                    boxes_augmented = BoundingBoxesOnImage(boxes_aug,shape=image.shape)
                    image_aug, bbs_aug = self.seq(image = image, bounding_boxes=boxes_augmented)
                    bbs_aug = bbs_aug.remove_out_of_image(fully=True, partly=True).clip_out_of_image()
                    boxes = np.zeros((len(bbs_aug.bounding_boxes),4))
                    class_ids = []
                    for i in range(len(bbs_aug.bounding_boxes)):
                        boxes[i]= [bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].y1,
                                    bbs_aug.bounding_boxes[i].x2,bbs_aug.bounding_boxes[i].y2]
                        class_ids.append(bbs_aug.bounding_boxes[i].label)
                    class_ids = np.array(class_ids, dtype=np.int16)
                    image = image_aug.astype(np.float32)
                    if not len(boxes):
                        boxes = None

        # Trajectory Specific
        # drift_prob = self.cfg.drift_prob if self.phase == 'train' else 0.
        # flip_prob = self.cfg.flip_prob if self.phase == 'train' else 0.
        # image, image_meta = whiten(image, image_meta, mean=self.rgb_mean, std=self.rgb_std)
        # image, image_meta, boxes = drift(image, image_meta, prob=drift_prob, boxes=boxes)
        # image, image_meta, boxes = flip(image, image_meta, prob=flip_prob, boxes=boxes)
        # image_visualize = image

        # LPR Specific
        image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        image_visualize = transforms.Grayscale(num_output_channels=3) (image)
        image = transforms.ToTensor()(image_visualize)
        image_visualize = cv2.cvtColor(np.array(image_visualize), cv2.COLOR_RGB2BGR)

        if boxes is not None:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., image_meta['orig_size'][1] - 1.)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., image_meta['orig_size'][0] - 1.)
            inds = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= 16
            boxes = boxes[inds]
            class_ids = class_ids[inds]
            if not len(boxes):
                boxes = None

        return image, image_visualize, image_meta, boxes, class_ids

    def prepare_annotations(self, class_ids, boxes):
        """
        :param class_ids:
        :param boxes: xyxy format
        :return: np.ndarray(#anchors, #classes + 9)
        """
        gt = np.zeros((self.num_anchors, self.num_classes + 9), dtype=np.float32)
        if boxes is not None:
            deltas, anchor_indices = compute_deltas(boxes, self.anchors)
            gt[anchor_indices, 0] = 1.  # mask
            gt[anchor_indices, 1:5] = boxes
            gt[anchor_indices, 5:9] = deltas
            gt[anchor_indices, 9 + class_ids] = 1.  # class logits

        return gt

    def get_sample_ids(self):
        raise NotImplementedError

    def load_image(self, index):
        raise NotImplementedError

    def load_annotations(self, index):
        raise NotImplementedError

    def save_results(self, results):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
