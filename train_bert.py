#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from trainer import create_trainer
from pytorch_lightning.plugins import DDPPlugin
from data import create_dataset, create_dataloader
from model import create_model
from utils import MessageLogger, get_env_info, get_root_logger
from utils.options import dict2str
import logging
import yaml
from yaml import CLoader as Loader
from copy import deepcopy
from pytorch_lightning.profiler import SimpleProfiler
import os 
from flatten_dict import flatten, unflatten
            
#Speed up training
cudnn.benchmark = True
def init_loggers(opt):
    log_file = opt['logger']["path"]
    logger = get_root_logger(
        logger_name=opt['logger']['name'], log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    return logger


def load(opt):
    opt = yaml.load(open(opt, mode='r'), Loader=Loader)
    base_opt = opt.get("base", None)
    if base_opt == None:
        return opt
    else:
        base_opt = flatten(load(base_opt))
        opt = flatten(opt)
        base_opt.update(opt)
        opt = unflatten(base_opt)
    return opt

def update_opt(opt):

    opt = load(opt)
    name = opt["name"]
    path = opt["exp_path"]
    
    os.makedirs(path,exist_ok=True)
    
    #set logger
    opt["logger"] = {}
    opt["logger"]["name"] = name
    opt["logger"]["path"] = os.path.join(path,"log_" + name)
    
    #checkpoint save path
    opt["train"]["save_path"] = os.path.join(path, "checkpoints")
    
    return opt

class data_prep(pl.LightningDataModule):
    def __init__(self, opt,num_process):
        super().__init__()
        self.opt = deepcopy(opt)
        self.num_process = num_process
    def setup(self, stage: str):   
        opt = self.opt
        dataset_opt = deepcopy(opt["datasets"])
        dataset_opt.update(dataset_opt["train"])
        dataset_opt.update(dataset_opt["slice"])
        dataset_opt['phase'] = 'train'
        del dataset_opt["train"]
        del dataset_opt["slice"]
        
        self.train_opt = dataset_opt
        self.train_dataset = create_dataset(dataset_opt,opt['logger']['name'])
        
        dataset_opt = deepcopy(opt["datasets"])
        dataset_opt.update(dataset_opt["eval"])
        dataset_opt.update(dataset_opt["slice"])
        dataset_opt['phase'] = 'eval'
        del dataset_opt["eval"]
        del dataset_opt["slice"]
        
        self.val_opt = dataset_opt
        self.val_dataset = create_dataset(dataset_opt,opt['logger']['name'])
        
    def train_dataloader(self):
        if hasattr(self, "train_loader"):
            return self.train_loader
        opt = self.opt
        train_loader = create_dataloader(self.train_dataset,self.train_opt,opt['logger']['name'], self.num_process)
        self.train_loader = train_loader
        return train_loader
    
    def val_dataloader(self):
        if hasattr(self, "eval_loader"):
            return self.eval_loader
        opt = self.opt
        eval_loader = create_dataloader(self.val_dataset,self.val_opt,opt['logger']['name'],self.num_process)   
        self.eval_loader = eval_loader
        return eval_loader
    
def main(args):

    num_process = args.gpus * args.num_nodes
    opt = update_opt(args.opt)
    if "torch_home" in opt:
        os.environ['TORCH_HOME'] = opt["torch_home"]
    
    init_loggers(opt)
    msg_logger = MessageLogger(opt)

    model = create_model(opt["network"],opt['logger']['name'])  
    
    model = create_trainer(opt['train']['type'], opt['logger']['name'], {"model": model, "log" : msg_logger, "opt" : opt["train"], "checkpoint": args.checkpoint})
    
    kwargs = {}
    sync_batchnorm = True #toggling sync_batchnorm = False during pre-training lead to better downstream performance 
    msg_logger.raw("sync_batchnorm: " + str(sync_batchnorm))
    check_val_every_n_epoch = opt['train'].get('check_val_every_n_epoch',1)
    plt = pl.Trainer(max_epochs = opt["train"].get("early_stop_epoch", opt["train"]["epoch"]), num_nodes=args.num_nodes, precision = opt.get("precision",32), gpus=args.gpus,strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = False, logger = False, profiler = SimpleProfiler(), sync_batchnorm = sync_batchnorm, replace_sampler_ddp = False, check_val_every_n_epoch = check_val_every_n_epoch, **kwargs)
    plt.fit(model,data_prep(opt,num_process))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--checkpoint", default = None, type = str)
    parser.add_argument('--opt', type=str, default = "config/ours/ablation/ours_moco_scheduler.yml", help='Path to option YAML file.')
    
    args = parser.parse_args()
    main(args)
    
