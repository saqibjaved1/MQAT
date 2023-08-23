import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader, sampler
# from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose
from .dataset import BOP_Dataset, collate_fn
from .transform import *
from .argument import get_args

from .distributed import    DistributedSampler

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)

def get_transforms(cfg):

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3, 3)

    train_trans = Compose(
        [
            Resize(
                cfg['INPUT']['INTERNAL_WIDTH'],
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'],
                cfg['SOLVER']['AUGMENTATION_SCALE'],
                cfg['SOLVER']['AUGMENTATION_ROTATION'],
                cfg['INPUT']['INTERNAL_WIDTH'],
                cfg['INPUT']['INTERNAL_HEIGHT'],
                internal_K),
            Normalize(
                cfg['INPUT']['PIXEL_MEAN'],
                cfg['INPUT']['PIXEL_STD']),
            ToTensor(),
        ]
    )

    valid_trans = Compose(
        [
            Resize(
                cfg['INPUT']['INTERNAL_WIDTH'],
                cfg['INPUT']['INTERNAL_HEIGHT'],
                internal_K),
            Normalize(
                cfg['INPUT']['PIXEL_MEAN'],
                cfg['INPUT']['PIXEL_STD']),
            ToTensor(),
        ]
    )

    return train_trans, valid_trans


def load_data_sets(logbook):
    cfg = get_args()
    train_trans, valid_trans = get_transforms(cfg)

    train_set = BOP_Dataset(
        cfg['DATASETS']['TRAIN'],
        cfg['DATASETS']['MESH_DIR'],
        cfg['DATASETS']['BBOX_FILE'],
        train_trans,
        cfg['DATASETS']['SYMMETRY_TYPES'],
        training=True)
    valid_set = BOP_Dataset(
        cfg['DATASETS']['VALID'],
        cfg['DATASETS']['MESH_DIR'],
        cfg['DATASETS']['BBOX_FILE'],
        valid_trans,
        training=False)

    batch_size_per_gpu = int(logbook.config['data']['bs_train']/logbook.hw_cfg['n_gpus']) if logbook.hw_cfg['n_gpus']>1 else logbook.config['data']['bs_train']
    # print(torch.distributed.init_process_group(backend='gloo', init_method='env://'))
    print(batch_size_per_gpu)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler= data_sampler(train_set, shuffle=True, distributed=logbook.hw_cfg['n_gpus'] > 1),
        num_workers=logbook.config['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=logbook.config['data']['bs_valid'],
        sampler=data_sampler(valid_set, shuffle=False, distributed=logbook.hw_cfg['n_gpus'] > 1),
        num_workers=logbook.config['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    return train_loader, valid_loader
    # return "train_loader", valid_loader
