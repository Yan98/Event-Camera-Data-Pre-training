#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
import datetime
import pytorch_lightning as pl
import os
from utils import accuracy,param_groups_lrd,MetricLogger
import math
import torch.nn as nn
from copy import deepcopy

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        for param in self.parameters():
            param.requires_grad = False  

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
                        
class TrainerModelFinetuing(pl.LightningModule):
    def __init__(self, model, log,opt,checkpoint):
        super().__init__()
        self.model= model
        self.log = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = False
        self.start_time  = None
        self.past_step = 0 
        os.makedirs(opt["save_path"],exist_ok=True)
        
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.model=self.model.get()
        self.max_acc = float("-inf")
        if self.opt.get("use_ema",False):
            self.model_ema = EMA(self.model)
            self.log.raw("Use ema")
  
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(dataset)
        
    @property
    def num_batch_size(self) -> int:
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()
        return dataset.batch_size       
  
    def adjust_learning_rate(self,idx):
        """Decays the learning rate with half-cycle cosine after warmup"""
        optimizer = self.optimizers()
        warmup_epochs = self.opt["warmup_epoch"]
        lr = self.opt["lr"]
        minlr = self.opt.get("min_lr", 0.0)
        epochs = self.opt["epoch"]
        epoch = self.current_epoch % epochs + idx / self.num_training_steps
        
        if epoch < warmup_epochs:
            lr = minlr + (lr - minlr) * epoch / warmup_epochs
        else:
            lr = minlr + (lr - minlr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
            
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
                
    def training_step(self,data,idx):
        
        images =  (data["event"] - 0.5) * 2
        labels = data["label"]
        
        accumulate_grad_batches = self.opt["accumulate_grad_batches"]
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        optimizer = self.optimizers()
        
        if idx % accumulate_grad_batches == 0:
            self.adjust_learning_rate(idx)
        
        pred_labels = self.model(
            images
        )
        loss = self.loss_fun(pred_labels,labels.squeeze())
        
        acc_1, acc_5 = accuracy(pred_labels.cpu(), labels.cpu(), topk=(1, 5))
        
        self.manual_backward(loss / accumulate_grad_batches)
            
        if (idx + 1) % accumulate_grad_batches == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            
        self.produce_log(loss, acc_1, acc_5, idx)
     
    def training_epoch_end(self, training_step_outputs):
       pass
        
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if "max_grad_norm" in self.opt:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt["max_grad_norm"])  
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        if hasattr(self, "model_ema"):
            self.model_ema.update(self.model)
            
    def produce_log(self,loss,acc_1,acc_5,idx):
        
        train_loss = self.all_gather(loss).mean().item()
        acc_1 = self.all_gather(acc_1).mean().item()
        acc_5 = self.all_gather(acc_5).mean().item()
        
        if self.trainer.is_global_zero and loss.device.index == 0 and idx % 100 == 0:
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
            
            self.log(
                {"current_epoch": self.current_epoch,
                 "max_epochs": self.trainer.max_epochs,  
                 "idx": idx,
                 "len_loader":len_loader,
                 "time_left": time_left,
                 "loss": train_loss, 
                 "acc_1": acc_1,
                 "acc_5": acc_5,
                    }
                )
            
            self.log.save_train(
                self.current_epoch,
                idx,
                {"loss": train_loss, 
                  "acc_1": acc_1,
                  "acc_5": acc_5,
                    }
                )
            
    def validation_step(self, data, idx):
        images =  (data["event"] - 0.5) * 2
        labels = data["label"]
        
        if hasattr(self, "model_ema"):
            pred_labels = self.model_ema.module(
            images
            )
        else:
            pred_labels = self.model(
                images
            )
        loss = self.loss_fun(pred_labels,labels.squeeze())
        
        acc_1, acc_5 = accuracy(pred_labels.cpu(), labels.cpu(), topk=(1, 5))
        batch_size = images.size(0)
        self.metric_logger.update(loss=loss.item())
        self.metric_logger.meters['acc_1'].update(acc_1, n=batch_size)
        self.metric_logger.meters['acc_5'].update(acc_5, n=batch_size)
    
    def on_validation_epoch_start(self):
        self.metric_logger = MetricLogger()
        self.log.raw("Initialized metric_logger")
        self.log.raw(f"Sampler epoch: {self.trainer._data_connector._train_dataloader_source.dataloader().sampler.epoch}")
    
    
    def validation_epoch_end(self,outputs):
        self.metric_logger.synchronize_between_processes()
        loss = self.metric_logger.loss.global_avg
        acc_1, acc_5 = self.metric_logger.acc_1.global_avg, self.metric_logger.acc_5.global_avg
        
        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            if self.start_time != None and acc_1 > self.max_acc:
                 self.save()
                 self.max_acc = max(self.max_acc,acc_1)
            
            
            self.log.raw(
                    "[Acc 1 :%f, Acc 5 :%f, Loss: %f] [Max ACC: %f]" %
                    (
                        acc_1,
                        acc_5,
                        loss,
                        self.max_acc
                     )         
                         ) 
            
            self.log.save_eval(
                self.current_epoch,
                {"loss": loss,
                  "acc_1": acc_1,
                  "acc_5" : acc_5,
                  "max_acc": self.max_acc,
                    }
                )
            
            
    def save(self):
        output_path = os.path.join(self.opt["save_path"], "best.pt")
        torch.save(
            {
             "model": self.model.state_dict(),
             "optimizer": self.optimizers().state_dict(),
             }
            , output_path)
        self.log.raw("Model saved")            
            
    def configure_optimizers(self):
        
        target_batch_size = self.opt["target_batch_size"] #1024
        accumulate_grad_batches = max(int(target_batch_size / self.trainer.num_gpus  / self.trainer.num_nodes / self.num_batch_size),1)
        self.opt["accumulate_grad_batches"] = accumulate_grad_batches
        
        
        self.log.raw(f"accumulate_grad_batches: {accumulate_grad_batches}")
        
        self.opt["lr"] = self.opt["base_lr"]  * target_batch_size / 256
        
        param_groups = param_groups_lrd(self.model, self.opt["weight_decay"],
                                        no_weight_decay_list=self.model.no_weight_decay(),
                                        layer_decay=self.opt["layer_decay"]
                                        )
        b1 = self.opt.get("b1",0.9)
        b2=self.opt.get("b2",0.999)
        self.log.raw(f" betas {b1} {b2}")
        optimizer = torch.optim.AdamW(
                            param_groups,
                            lr = self.opt["lr"],
                            betas = (b1, b2),
            )
        
        if self.checkpoint != None:
            pass
        
        return optimizer         