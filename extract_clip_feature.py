#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import clip
from PIL import Image
import os
import glob
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract(args, VAL):
    
    save = args.save_dir
    batch_size = args.batch    
    
    if VAL:
        save +="_val"
        
    os.makedirs(save,exist_ok=True)
    
    def data_generator(root = args.source_dir, index = 0):
        
        subfolder = "val" if VAL else "train"
        
        folders = [folder for folder in os.listdir(os.path.join(root, subfolder)) if folder[0] == "n"]
        folder = folders[index]
        path = os.path.join(root,subfolder,folder)
        imgs = glob.glob(os.path.join(path, "*.JPEG"))
        
        for img in imgs:
            file = img.split(os.sep)[-1][:-5]
            img = preprocess(Image.open(img)).unsqueeze(0).to(device)
            yield img,folder, file
          
    with torch.no_grad():        
        for i in range(1000):
            x = []
            emb = []
            files = []
            import tqdm
            for img, name,file in tqdm.tqdm(data_generator(index = i)):
                x.append(img)
                files.append(file)
                if len(x) == batch_size:
                    x = torch.cat(x)
                    emb.append(model.encode_image(x))
                    x = []
                    
            if len(x) != 0:
                x = torch.cat(x)
                emb.append(model.encode_image(x))
            emb = torch.cat(emb).to("cpu")
            emb = dict(zip(files,emb))
            
            torch.save(emb, os.path.join(save, name + ".pt"))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--batch", default = 128, type = int)
    parser.add_argument("--source_dir", default = None, type = str)
    parser.add_argument("--save_dir", default = None, type = str)
    
    args = parser.parse_args()
    extract(args, False)
    extract(args, True)            
        
    