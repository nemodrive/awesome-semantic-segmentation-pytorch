"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .ade import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .sbu_shadow import SBUSegmentation
from .upb_kitti import UPBSegmentation
from .kitti import KittiSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation,
    'sbu': SBUSegmentation,
    'upb': UPBSegmentation,
    'kitti': KittiSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
