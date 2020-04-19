import tqdm
import numpy as np
import torch.utils.data

from utils.config import Config
from utils.misc import load_dataset


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(DatasetWrapper, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        image, _ = self.dataset.load_image(index)
        return image

    def __len__(self):
        return len(self.dataset)


def compute_dataset_mean_and_std(dataset, max_num_samples=30000, num_workers=4):
    """
    :param dataset: instance of torch.utils.data.Dataset class
    :param max_num_samples: randomly select N samples instead of whole dataset
    :param num_workers:
    """
    max_num_samples = min(max_num_samples, len(dataset))
    dataset.sample_ids = np.random.permutation(dataset.sample_ids)[:max_num_samples]
    dataloader = torch.utils.data.DataLoader(DatasetWrapper(dataset),
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    dataset_mean, dataset_std = [], []
    for image in tqdm.tqdm(dataloader):
        dataset_mean.append(torch.mean(image, dim=[1, 2]))
        dataset_std.append(torch.std(image, dim=[1, 2]))

    dataset_mean = torch.mean(torch.cat(dataset_mean, dim=0), dim=0)
    dataset_std = torch.mean(torch.cat(dataset_std, dim=0), dim=0)

    return (dataset_mean.cpu().numpy(),
            dataset_std.cpu().numpy())


def main():
    cfg = Config().parse('eval --dataset kitti'.split(' '))
    dataset = load_dataset(cfg.dataset)('trainval', cfg)

    mean, std = compute_dataset_mean_and_std(dataset)

    print('Dataset\'s RGB mean: ', mean)
    print('Dataset\'s RGB std: ', std)


if __name__ == '__main__':
    main()
