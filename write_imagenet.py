from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
from PIL import Image

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

class CIFAR_AUG(Dataset):
    """cifar-10.1 dataset"""
    def __init__(self, root_dir, name):
        if name == 'cifar101':
            self.data = np.load(os.path.join(root_dir, 'cifar10.1_v4_data.npy'))
            self.targets = np.load(os.path.join(root_dir, 'cifar10.1_v4_labels.npy'))
        elif name == 'cifar102':
            raw_data = np.load(os.path.join(root_dir, 'cifar102_test.npz'))
            self.data = raw_data['images']
            self.targets = raw_data['labels']
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        return img, target

    def __len__(self):
        return len(self.data)

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet', 'cifar101', 'cifar102'])), 'Which dataset to write', default='imagenet'),
    split=Param(And(str, OneOf(['train', 'test'])), 'Train or test set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length', required=True),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, 'compress probability', default=None)
)

@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
def main(dataset, split, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode,
         compress_probability):
    if max_resolution == -1:
        max_resolution = None
    if dataset == 'cifar':
        my_dataset = CIFAR10(root=data_dir, train=(split == 'train'), download=True)
    elif dataset == 'cifar101':
        assert split == 'test'
        my_dataset = CIFAR_AUG(data_dir, dataset)
    elif dataset == 'cifar102':
        assert split == 'test'
        my_dataset = CIFAR_AUG(data_dir, dataset)
    elif dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
