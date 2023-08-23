import os
import json
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from .boxlist import BoxList
from .poses import PoseAnnot

from .utils import (
    load_bop_meshes,
    load_bbox_3d,
    get_single_bop_annotation,
    load_image_cached,
    load_json_cached,
)

# memory cache, a shared global dict (company with multiple workers in PyTorch)
import multiprocessing
# g_mem_cache = multiprocessing.Manager().dict()
g_mem_cache = None

class BOP_Dataset(Dataset):
    def __init__(self, image_list_file, mesh_dir, bbox_json, transform, symmetry_types=None, training=True):
        # file list and data are typically in the same directory
        dataDir = os.path.split(image_list_file)[0]
        with open(image_list_file, 'r') as f:
            tmp_img_files = f.readlines()
            for i in range(len(tmp_img_files)):
                if tmp_img_files[i].startswith('/'):
                    tmp_img_files[i] = tmp_img_files[i].strip()
                else:
                    tmp_img_files[i] = dataDir + '/' + tmp_img_files[i].strip()
        #
        self.img_files = tmp_img_files
        rawSampleCount = len(self.img_files)

        if training:
            random.shuffle(self.img_files)

        print("Number of samples: %d / %d" % (len(self.img_files), rawSampleCount))
        #
        self.meshes, self.objID_2_clsID= load_bop_meshes(mesh_dir)
        #
        self.bbox_3d = load_bbox_3d(bbox_json)

        self.transformer = transform
        self.symmetry_types = symmetry_types
        self.training = training

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.img_files) - 1)
            item = self.getitem1(index)
        return item

    def getitem1(self, index):
        img_path = self.img_files[index]
        # Load image
        try:
            img = load_image_cached(img_path, g_mem_cache)
            #
            if img is None:
                raise RuntimeError('load image error')
            #
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0)).astype(np.uint8)
            #
            if len(img.shape) == 2:
                # convert gray to 3 channels
                img = cv2.merge([img,img,img]) # three channels by duplication
            # elif img.shape[2] == 3:
            #     # add an alpha channel
            #     img = np.concatenate((img, np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)*255), axis=-1)
            elif img.shape[2] == 4:
                # having alpha
                tmpBack = (img[:,:,3] == 0)
                img[:,:,0:3][tmpBack] = 255 # white background
        except:
            print('image %s not found' % img_path)
            return None

        # Load labels (BOP format)
        height, width, _ = img.shape
        K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation(img_path, self.objID_2_clsID, g_mem_cache)

        # get (raw) image meta info
        meta_info = {
            'path': img_path,
            'K': K,
            'width': width,
            'height': height,
            'class_ids': class_ids,
            'rotations': rotations,
            'translations': translations
        }

        target = PoseAnnot(self.bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)

        # transformation
        img, target = self.transformer(img, target)
        target = target.remove_invalids(min_area = 10)
        if self.training and len(target) == 0:
            # print("WARNING: skipped a sample without any targets")
            return None

        # if False:
        if True:
            # symmetry handling (after all transformations)
            if self.symmetry_types is not None and len(self.symmetry_types) > 0:
                target = target.symmetry_handling(self.symmetry_types)
        else:
            # debug
            cvImg = img.numpy().transpose(1, 2, 0)
            # de-normalize
            cvImg = cvImg * (np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3) * 255)
            cvImg = cvImg + (np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3) * 255)
            #
            cvImg = cv2.cvtColor(cvImg.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cvImg_1 = target.visualize(cvImg)
            cv2.imshow('img_raw', cvImg_1)

            if self.symmetry_types is not None and len(self.symmetry_types) > 0:
                target = target.symmetry_handling(self.symmetry_types)
                cvImg_2 = target.visualize(cvImg)
                cv2.imshow('img_after_symmetry_handling', cvImg_2)

            cv2.waitKey(0)

        return img, target, meta_info

class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        if max_size[1] % stride != 0:
            max_size[1] = (max_size[1] | (stride - 1)) + 1
        if max_size[2] % stride != 0:
            max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)

def collate_fn(size_divisible):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], size_divisible)
        targets = batch[1]
        meta_infos = batch[2]

        return imgs, targets, meta_infos

    return collate_data
