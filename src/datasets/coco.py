from datasets.base import BaseDataset


class COCO(BaseDataset):
    def __init__(self, phase, cfg):
        super(COCO, self).__init__(phase, cfg)

        self.input_size = (512, 512)  # (height, width), both dividable by 16
        self.class_names = (
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        )
        # todo

    def get_sample_ids(self):
        # todo
        pass

    def load_image(self, index):
        # todo
        pass

    def load_annotations(self, index):
        # todo
        pass

    def save_results(self, results):
        # todo
        pass

    def evaluate(self):
        # todo
        pass

