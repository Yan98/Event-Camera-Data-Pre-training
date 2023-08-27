#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import torch
import torch.utils.data
from os import path as osp

from utils import get_root_logger, scandir
from torch.utils.data import DataLoader
__all__ = ['create_trainer']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_trainer.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'trainer.{file_name}')
    for file_name in dataset_filenames
]


def create_trainer(name, logger_name,trainer_opt):
    # dynamic instantiation
    for module in _dataset_modules:
        trainer_cls = getattr(module, name, None)
        if trainer_cls is not None:
            break
    if trainer_cls is None:
        raise ValueError(f'Trainer {name} is not found.')
    trainer = trainer_cls(**trainer_opt)    
    logger = get_root_logger(logger_name)
    logger.info(
        f'Trainer {trainer.__class__.__name__} - {name} '
        'is created.')    
    return trainer