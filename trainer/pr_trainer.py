#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
import datetime
import pytorch_lightning as pl
import os
from utils import accuracy, MetricLogger
import math

class TrainerModelPretraining(pl.LightningModule):
    
    def __init__(self, model, log,opt,checkpoint):
        super().__init__()
        self.model = model
        self.log = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = False
        self.start_time  = None
        os.makedirs(opt["save_path"],exist_ok=True)
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader() 
        return len(dataset)
    
    @property
    def num_batch_size(self) -> int:
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()
        return dataset.batch_size

    def adjust_moco_momentum(self, idx):
        """Adjust moco momentum based on current epoch"""
        if self.opt.get("fix_m", False):
            return self.opt["moco_m"]
        
        epochs = self.opt.get("cycle_epoch", self.opt["epoch"])
        epoch = self.current_epoch % epochs + idx / self.num_training_steps
        
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / epochs)) * (1. - self.opt["moco_m"])
        return m

    def adjust_learning_rate(self,idx):
        """Decays the learning rate with half-cycle cosine after warmup"""
        optimizer = self.optimizers()
        warmup_epochs = self.opt["warmup_epoch"]
        lr = self.opt["learning_rate"]
        minlr = self.opt.get("min_lr", 0.0)
        epochs = self.opt.get("cycle_epoch", self.opt["epoch"])
        epoch = self.current_epoch % epochs + idx / self.num_training_steps
        
        if epoch < warmup_epochs:
            lr = minlr + (lr - minlr) * epoch / warmup_epochs
        else:
            lr = minlr + (lr - minlr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def training_step(self,data,idx):
        
        event1 = (data["event1"] - 0.5)  * 2
        event2 = (data["event2"] - 0.5) * 2
        emb = data["emb"]
        label = data["label"]
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        
        optimizer = self.optimizers()
        self.adjust_learning_rate(idx)
        
        loss_event,loss_image, kl_loss,pred, cls_loss = self.model(
            event1,
            event2,
            emb,
            label,
            self.adjust_moco_momentum(idx),
        )
        #return loss
        self.manual_backward(loss_event + loss_image + kl_loss + cls_loss)
            
        optimizer.step()
        optimizer.zero_grad()
        
        self.produce_log(loss_event,loss_image, kl_loss, pred, label, idx)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if "max_grad_norm" in self.opt:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt["max_grad_norm"]) 

    def training_epoch_end(self, training_step_outputs):
        if self.trainer.is_global_zero and (self.current_epoch + 1) % self.opt["save_every"] == 0:
            self.save()

    
    def produce_log(self,loss_event,loss_image,kl_loss,pred, label,idx):
        
        loss_event = self.all_gather(loss_event).mean().item()
        loss_image = self.all_gather(loss_image).mean().item()
        kl_loss = self.all_gather(kl_loss).mean().item()
        pred_labels = self.all_gather(pred).view(-1,1000)
        labels = self.all_gather(label).view(-1)
        acc_1, acc_5 = accuracy(pred_labels.cpu(), labels.cpu(), topk=(1, 5))
        
        if self.trainer.is_global_zero and idx % 100 == 0:
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
            lr = self.optimizers().param_groups[0]['lr']
            self.log(
                {"current_epoch": self.current_epoch,
                 "max_epochs": self.trainer.max_epochs,  
                 "idx": idx,
                 "len_loader":len_loader,
                 "time_left": time_left,
                 "loss_event": loss_event, 
                 "loss_image": loss_image,
                  "kl_loss": kl_loss,
                  "acc_1": acc_1,
                  "acc_5": acc_5,
                  "lr": lr,
                  "m": self.adjust_moco_momentum(idx)
                    }
                )
            
            self.log.save_train(
                self.current_epoch,
                idx,
                {"loss_event": loss_event, 
                 "loss_image": loss_image,
                  "kl_loss": kl_loss,
                  "acc_1": acc_1,
                  "acc_5": acc_5,
                  "lr": lr,
                  "m": self.adjust_moco_momentum(idx)
                    }
                )
    def on_validation_epoch_start(self):
        self.metric_logger = MetricLogger()
        self.log.raw("Initialized metric_logger")
            
    def validation_step(self, data, idx):
        event1 = (data["event1"] - 0.5)  * 2
        event2 = (data["event2"] - 0.5) * 2
        emb = data["emb"]
        label = data["label"]
        loss_event,loss_image, kl_loss,pred, cls_loss = self.model(
            event1,
            event2,
            emb,
            label,
            self.adjust_moco_momentum(idx),
        )
        
        acc_1, acc_5 = accuracy(pred.cpu(), label.cpu(), topk=(1, 5))
        batch_size = event1.size(0)
        self.metric_logger.update(loss=loss_event.item())
        self.metric_logger.meters['acc_1'].update(acc_1, n=batch_size)
        self.metric_logger.meters['acc_5'].update(acc_5, n=batch_size)
        self.metric_logger.meters['loss_image'].update(loss_image.item(), n=batch_size)
        self.metric_logger.meters['kl_loss'].update(kl_loss.item(), n=batch_size)
        
    def validation_epoch_end(self,outputs):
        self.metric_logger.synchronize_between_processes()
        loss_event = self.metric_logger.loss.global_avg
        acc_1, acc_5 = self.metric_logger.acc_1.global_avg, self.metric_logger.acc_5.global_avg
        loss_image, kl_loss = self.metric_logger.loss_image.global_avg, self.metric_logger.kl_loss.global_avg
        
        if self.trainer.is_global_zero and self.trainer.num_gpus!=0: 
            self.log.raw(
                    "[loss_event :%f, loss_image :%f, kl_loss:%f, acc_1:%f, acc_5:%f]" %
                    (
                        loss_event,
                        loss_image,
                        kl_loss,
                        acc_1,
                        acc_5,
                     )         
                         ) 
            self.log.save_eval(
                self.current_epoch,
                {"loss_event": loss_event,
                  "loss_image": loss_image,
                  "kl_loss": kl_loss,
                  "acc_1": acc_1,
                  "acc_5" : acc_5
                    }
                )

    def save(self):
        output_path = os.path.join(self.opt["save_path"], f"{self.current_epoch + 1}.pt")
        torch.save(
            {
             "checkpoint": self.model.state_dict(),
             "optimizer": self.optimizers().state_dict(),
             }
            , output_path)
        self.log.raw("Model saved")

    def configure_optimizers(self):
        self.opt["learning_rate"] = self.trainer.num_gpus * self.trainer.num_nodes * self.num_batch_size / 256 * self.opt["base_lr"]
        self.log.raw(f"learning_rate: {self.opt['learning_rate']}")
        b1 = self.opt.get("b1",0.9)
        b2=self.opt.get("b2",0.999)
        self.log.raw(f" betas {b1} {b2}")
        
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr = self.opt["learning_rate"],
                betas = (b1, b2),
                weight_decay = self.opt["weight_decay"],
                )

        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            self.model.load_state_dict(checkpoint["checkpoint"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            del checkpoint

        return optimizer                     
                                
        
     
