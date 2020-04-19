"""
Boxes implementations based on Numpy
"""
import os

import numpy as np
import cv2

EPSILON = 1E-10


def xyxy_to_xywh(boxes_xyxy):
    assert np.ndim(boxes_xyxy) == 2
    assert np.all(boxes_xyxy[:, 0] < boxes_xyxy[:, 2])
    assert np.all(boxes_xyxy[:, 1] < boxes_xyxy[:, 3])

    return np.concatenate([
        (boxes_xyxy[:, [0]] + boxes_xyxy[:, [2]]) / 2.,
        (boxes_xyxy[:, [1]] + boxes_xyxy[:, [3]]) / 2.,
        boxes_xyxy[:, [2]] - boxes_xyxy[:, [0]] + 1.,
        boxes_xyxy[:, [3]] - boxes_xyxy[:, [1]] + 1.
    ], axis=1)


def xywh_to_xyxy(boxes_xywh):
    assert np.ndim(boxes_xywh) == 2
    assert np.all(boxes_xywh > 0)

    return np.concatenate([
        boxes_xywh[:, [0]] - 0.5 * (boxes_xywh[:, [2]] - 1),
        boxes_xywh[:, [1]] - 0.5 * (boxes_xywh[:, [3]] - 1),
        boxes_xywh[:, [0]] + 0.5 * (boxes_xywh[:, [2]] - 1),
        boxes_xywh[:, [1]] + 0.5 * (boxes_xywh[:, [3]] - 1)
    ], axis=1)


def generate_anchors(grid_size, input_size, anchors_seed):
    """
    :param grid_size: (grid_height, grid_width), shape of the output of ConvDet layer
    :param input_size: (height, width), shape of input image
    :param anchors_seed: np.ndarray(N, 2), where N is #anchors per grid
    :return: np.ndarray(A, 4), in xyxy format, where A = N * grid_height * grid_width
    """
    assert anchors_seed.shape[1] == 2

    anchors_per_grid = anchors_seed.shape[0]
    grid_height, grid_width = grid_size

    anchors_shape = np.reshape(
        grid_width * grid_height * [anchors_seed],
        (grid_height, grid_width, anchors_per_grid, 2)
    )

    input_height, input_width = input_size
    anchors_center_x, anchors_center_y = np.meshgrid(
        input_width * (1 / (grid_width * 2) + np.linspace(0, 1, grid_width + 1)[:-1]),
        input_height * (1 / (grid_height * 2) + np.linspace(0, 1, grid_height + 1)[:-1])
    )
    anchors_center = np.stack((anchors_center_x, anchors_center_y), axis=2)
    anchors_center = np.repeat(
        np.reshape(
            anchors_center, (grid_height, grid_width, 1, 2)
        ), anchors_per_grid, axis=2
    )
    anchors_xywh = np.concatenate((anchors_center, anchors_shape), axis=3)

    return np.reshape(anchors_xywh, (-1, 4))


def compute_overlaps(boxes, box):
    """
    :param boxes: xyxy format
    :param box: xyxy format
    :return:
    """
    lr = np.maximum(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0)
    tb = np.maximum(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0)
    inter = lr * tb
    union = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + \
            (box[2] - box[0]) * (box[3] - box[1]) - inter
    return inter / (union + EPSILON)


def compute_deltas(boxes_xyxy, anchors_xywh):
    """
    :param boxes_xyxy: xyxy format
    :param anchors_xywh: np.ndarray(A, 4), xywh format
    :return:
    """
    num_anchors = anchors_xywh.shape[0]

    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    anchors_xyxy = xywh_to_xyxy(anchors_xywh)

    deltas, anchor_indices = [], []
    anchor_indices_set = set()

    for i in range(boxes_xyxy.shape[0]):
        overlaps = compute_overlaps(anchors_xyxy, boxes_xyxy[i])

        anchor_idx = num_anchors

        # sort for largest overlaps
        for ov_idx in np.argsort(-overlaps):
            # when overlap is zero break
            if overlaps[ov_idx] <= 0:
                break
            if ov_idx not in anchor_indices_set:
                anchor_indices_set.add(ov_idx)
                anchor_idx = ov_idx
                break

        # if the largest available overlap is 0, choose the anchor box with the one that has the
        # smallest square distance
        if anchor_idx == num_anchors:
            dist = np.sum((boxes_xywh[i] - anchors_xywh) ** 2, axis=1)
            for dist_idx in np.argsort(dist):
                if dist_idx not in anchor_indices_set:
                    anchor_indices_set.add(dist_idx)
                    anchor_idx = dist_idx
                    break

        anchor_indices.append(anchor_idx)

        delta = [(boxes_xywh[i, 0] - anchors_xywh[anchor_idx, 0]) / anchors_xywh[anchor_idx, 2],
                 (boxes_xywh[i, 1] - anchors_xywh[anchor_idx, 1]) / anchors_xywh[anchor_idx, 3],
                 np.log(boxes_xywh[i, 2] / anchors_xywh[anchor_idx, 2]),
                 np.log(boxes_xywh[i, 3] / anchors_xywh[anchor_idx, 3])]

        deltas.append(delta)

    anchor_indices = np.array(anchor_indices, dtype=np.int32)
    deltas = np.array(deltas, dtype=np.float32)

    return deltas, anchor_indices


def boxes_postprocess(boxes, image_meta):
    """
    remap processed boxes back into original image coordinates
    :param boxes: xyxy format
    :param image_meta:
    :return:
    """
    if 'scales' in image_meta:
        boxes[:, [0, 2]] /= image_meta['scales'][1]
        boxes[:, [1, 3]] /= image_meta['scales'][0]

    if 'padding' in image_meta:
        boxes[:, [0, 2]] -= image_meta['padding'][2]
        boxes[:, [1, 3]] -= image_meta['padding'][0]

    if 'crops' in image_meta:
        boxes[:, [0, 2]] += image_meta['crops'][2]
        boxes[:, [1, 3]] += image_meta['crops'][0]

    if 'flipped' in image_meta and image_meta['flipped']:
        image_width = image_meta['drifted_size'][1] if 'drifted_size' in image_meta else \
                      image_meta['orig_size'][1]
        boxes_widths = boxes[:, 2] - boxes[:, 0] + 1.
        boxes[:, 0] = image_width - 1 - boxes[:, 2]
        boxes[:, 2] = boxes[:, 0] + boxes_widths - 1.

    if 'drifts' in image_meta:
        boxes[:, [0, 2]] += image_meta['drifts'][1]
        boxes[:, [1, 3]] += image_meta['drifts'][0]

    return boxes


def visualize_boxes(image, class_ids, boxes, scores=None, class_names=None, save_path=None, show=False):
    image = image.astype(np.uint8)
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_id = class_ids[i]
        bbox = boxes[i].astype(np.uint32).tolist()
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              class_colors[class_id].tolist(), 2)

        class_name = class_names[class_id] if class_names is not None else 'class_{}'.format(class_id)
        text = '{} {:.2f}'.format(class_name, scores[i]) if scores is not None else class_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, fontScale=.5, thickness=1)[0]
        image = cv2.rectangle(image,
                              (bbox[0], bbox[1] - text_size[1] - 8),
                              (bbox[0] + text_size[0] + 8, bbox[1]),
                              class_colors[class_id].tolist(), -1)
        image = cv2.putText(image, text, (bbox[0] + 4, bbox[1] - 4), font,
                            fontScale=.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    if show:
        title = '{} (press any key to continue)'.format(os.path.basename(save_path))
        cv2.imshow(title, image[:, :, ::-1])
        cv2.waitKey()
        cv2.destroyWindow(title)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image[:, :, ::-1])


class_colors = (255. * np.array(
    [0.850, 0.325, 0.098,
     0.466, 0.674, 0.188,
     0.098, 0.325, 0.850,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500]
)).astype(np.uint8).reshape((-1, 3))
