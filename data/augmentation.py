#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.transforms as transforms
import torch

class RandAug(object):
    def __init__(self, num_ops = 2, magnitude=20 ):
        self.m = transforms.RandAugment(num_ops= num_ops, magnitude= magnitude, fill=0)
        
    def __call__(self, x):
        dtype = x.dtype
        _,h,w = x.size()
        x = x.view(-1,1,h,w)
        x = (x * 255).to(torch.uint8)
        x = self.m(x).to(dtype) / 255
        x=  x.squeeze()
        return x
    
def get_augmentation(cfg,pop_resize = None):
    cfg = cfg["view_augmentation"]
    name =  cfg["name"]
    if name == "Ours":
        return Ours(cfg, pop_resize)    
    
    elif name == "FineTune":
        return FineTune(cfg, pop_resize)
    else:
        raise SystemExit

         
def FineTune(cfg,pop_resize):
    k = "view1"
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(cfg[k]["crop_min"], 1.), interpolation = 0),
        transforms.RandomHorizontalFlip(),
    ]
    if pop_resize:
        augmentation.pop(0)

    return transforms.Compose(augmentation) 

  
def Ours(cfg,pop_resize):
    augs = []
    for k in ["view1", "view2"]:
        augmentation = [
        transforms.RandomResizedCrop(224, scale=(cfg[k]["crop_min"], 1.), interpolation = 0),
        transforms.RandomHorizontalFlip(),
        ]
        if pop_resize:
            augmentation.pop(0)
        
        augs.append(transforms.Compose(augmentation))
    return augs    
    