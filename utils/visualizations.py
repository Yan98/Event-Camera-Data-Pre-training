#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.utils import save_image

def simple_img(event_tensor):
    H,W=224,224
    event_tensor = torch.clip(event_tensor,0,1) * H
    coords = event_tensor[:, :2].long()
    event_image = torch.zeros([H, W])
    event_image[coords[:,1], coords[:, 0]] = 1.0

    event_image = torch.unsqueeze(event_image, -1)

    event_image = event_image.permute(2, 0, 1)
    event_image = event_image.float()

    return event_image


def event_visualize(imgs, recons, path, value_range = (0,1), convert=False):
    if convert:
        imgs = torch.stack([simple_img(img) for img in imgs])
        recons=torch.stack([simple_img(recon) for recon in recons])
    b, c, h, w = imgs.size()
    imgs =imgs.reshape(-1,1,h,w)
    recons=recons.reshape(-1,1,h,w)
    sample = torch.cat((imgs,recons),-2)
    
    save_image(sample, path, normalize = True, value_range = value_range)