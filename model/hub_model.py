import torch
from torch import nn
from .ours_model import ours_model_pretrain as ours_pretrain
from .ours_model import ours_model_finetune as ours_finetune
import torch.distributed as dist
from functools import reduce
from operator import mul
import math

class OursModel(nn.Module):
    def __init__(self, pretrain_checkpoint, mask_ratio, num_classes, lambda_event = 1, lambda_img =1, lambda_kl = 2, stop_conv1 = False, base_model = 'vits', emb_dim = 512, use_grad = False, channels = 2, temp_event = 0.2, temp_image = 0.1,  project = False,**kwargs):
        super().__init__()
        
        if pretrain_checkpoint != None and "moco" in pretrain_checkpoint:
            checkpoint = torch.load(pretrain_checkpoint,map_location="cpu")['state_dict']
            del checkpoint['module.base_encoder.patch_embed.proj.weight']
            del checkpoint['module.base_encoder.patch_embed.proj.bias']
            checkpoint['module.base_encoder.cls_token'] = checkpoint['module.base_encoder.cls_token'].repeat(1,2,1)
            checkpoint['module.base_encoder.pos_embed'] = checkpoint['module.base_encoder.pos_embed'][:,1:]
            event_head = {k[len("module.base_encoder.head."):]:v for k,v in checkpoint.items() if k.startswith('module.base_encoder.head')}
            predictor = {k[len("module.predictor."):]:v for k,v in checkpoint.items() if k.startswith('module.predictor')}
            for k in list(checkpoint.keys()):
                if k.startswith('module.base_encoder'): 
                    checkpoint[k[len("module.base_encoder."):]] = checkpoint[k]
                del checkpoint[k]
        elif pretrain_checkpoint != None:
            checkpoint = torch.load(pretrain_checkpoint, map_location='cpu')
            checkpoint = {k:v for k,v in checkpoint['model'].items() if not k.startswith('patch_embed.')}
            checkpoint['pos_embed'] = checkpoint['pos_embed'][:,1:]
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint:
                    del checkpoint[k]
            event_head = {}
            predictor = {}
        else:
            checkpoint = {}
            event_head = {}
            predictor = {}
        if base_model == 'vits':
            model_cls = ours_pretrain.ours_model_small
        elif base_model == 'vitb':
            model_cls= ours_pretrain.ours_model_base
        elif base_model == 'vitl':
            model_cls = ours_pretrain.ours_model_large
        else:
            raise SystemExit
        self.encoder_q = model_cls(mask_ratio, channels,**kwargs)
        self.encoder_k = model_cls(mask_ratio, channels,**kwargs)
        
        missing_key = self.encoder_q.load_state_dict(checkpoint,strict=False)
        print(missing_key)
        
        hidden_dim = self.encoder_q.event_head.weight.shape[1]
        del self.encoder_q.event_head, self.encoder_k.event_head 
        del self.encoder_q.image_head, self.encoder_k.image_head 
        
        self.encoder_q.event_head = self._build_mlp(3, hidden_dim, 4096, 256)
        self.encoder_q.image_head = self._build_mlp(3, hidden_dim, 4096, 256)
        self.encoder_k.event_head = self._build_mlp(3, hidden_dim, 4096, 256)
        self.encoder_k.image_head = self._build_mlp(3, hidden_dim, 4096, 256)
        
        
        self.encoder_q.event_head.load_state_dict(event_head,strict=False)
        self.encoder_q.image_head.load_state_dict(event_head,strict=False)
        
        self.pred1 = self._build_mlp(2, 256, 4096, 256)
        self.pred2 = self._build_mlp(2, 256, 4096, 256)
        
        self.pred1.load_state_dict(predictor,strict=False)
        self.pred2.load_state_dict(predictor,strict=False)
        
        to_event = []
        to_event += [nn.Linear(emb_dim, 256,bias=False)]
        
        self.to_event = nn.Sequential(*to_event)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes)
            )
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  

        if stop_conv1:
            val = math.sqrt(6. / float(3 * reduce(mul, self.encoder_q.patch_embed.patch_size, 1) + self.encoder_q.embed_dim))
            nn.init.uniform_(self.encoder_q.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.encoder_q.patch_embed.proj.bias)
            self.encoder_q.patch_embed.proj.weight.requires_grad = False
            self.encoder_q.patch_embed.proj.bias.requires_grad = False
            print("Stop gradient")

        self.T_event=temp_event
        self.T_image = temp_image
        self.project = project
        self.use_grad = use_grad
        self.lambda_kl, self.lambda_event, self.lambda_img = lambda_kl, lambda_event, lambda_img
        
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True,seq=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        if seq:
            return nn.Sequential(*mlp)
        else:
            return mlp
    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
                    
    def sinkhorn(self, out):
        Q = torch.exp(out).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q = Q / sum_Q.detach()

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q = Q / sum_of_rows.detach()
            Q = Q / K

            # normalize each column: total weight per sample must be 1/B
            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            Q = Q / B

        Q = Q * B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
        

    def get_kl_loss(self,q1, q2, k1):
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        #q2 = concat_all_gather(q2)
        k1 = nn.functional.normalize(k1, dim=1)
        k2=k1
        #k2 = concat_all_gather(k1)
        
        f = nn.LogSoftmax(dim=-1)
        
        q = torch.einsum('nc,mc->nm', [q1, q2]) / self.T_image
        k=  torch.einsum('nc,mc->nm', [k1, k2]) / self.T_image
        return nn.KLDivLoss(reduction="batchmean",log_target=False)(f(q), self.sinkhorn(k)) #f(k))
          
    def contrastive_loss(self, q, k, T, l2_norm = True):
        # normalizePretraining
        if l2_norm:
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T #self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T) #self.T)  
    
    
    def vector_project(self, v1,v2):
        if not self.use_grad:
            v2 = v2.detach()
        v1 = nn.functional.normalize(v1, dim=1)
        v2 = nn.functional.normalize(v2, dim=1)
        return (v1 * v2).sum(1,True) * v2
    
    def forward(self,event_frame1,event_frame2, emb, label, m):
        emb = self.to_event(emb)
        event1,image1, pred = self.encoder_q(event_frame1)
        event1 = self.pred1(event1)
        image1 = self.pred2(image1)
        
        pred = self.head(pred.detach().clone())
        cls_loss = nn.CrossEntropyLoss()(pred,label.squeeze())
        
        with torch.no_grad():  # no gradient to keys
            if self.training:
                self._momentum_update_key_encoder(m)  # update the key encoder
            event2, image2, _  = self.encoder_k(event_frame2)  

        if self.project:
            event1 = self.vector_project(event1,emb)
            event2 = self.vector_project(event2,emb)

        kl_loss = self.get_kl_loss(image1,image1,emb) #* 2   
            
        loss_event= self.contrastive_loss(event1, event2,self.T_event, not self.project) 

        loss_image = self.contrastive_loss(image1, emb,self.T_image)        

        return loss_event * self.lambda_event, loss_image * self.lambda_img, kl_loss * self.lambda_kl, pred, cls_loss  

class OursModelFineTune(nn.Module):
    def __init__(self, pretrain_checkpoint, base_model = 'vits', linear_probing = False, lr_checkpoint = None, **kwargs):
        super().__init__()
        import os
        num_classes = kwargs["num_classes"]
        if base_model == 'vits':
            model_cls = ours_finetune.ours_model_small_ft
        elif base_model == 'vitb':
            model_cls= ours_finetune.ours_model_base_ft
        else:
            raise SystemExit
       
        model = model_cls(linear_probing=linear_probing,**kwargs)
        if os.path.isdir(pretrain_checkpoint):
            pretrain_checkpoint = os.path.join(pretrain_checkpoint, sorted([i for i in os.listdir(pretrain_checkpoint) if i.endswith(".pt")],key=lambda x:int(x.split(".")[0]))[-1])
        
        
        if lr_checkpoint is None:
            checkpoint = torch.load(pretrain_checkpoint, map_location='cpu')
    
            print("Load pre-trained checkpoint from: %s" % pretrain_checkpoint)
            checkpoint = checkpoint['checkpoint']
            if num_classes == 1000 and linear_probing == False:
                head_checkpoint = {k:v for k,v in checkpoint.items() if k.startswith("head.")}
            else:
                head_checkpoint = {}
                print("Init no head")
                #head_checkpoint = {k:v for k,v in checkpoint.items() if k.startswith("head.0.")}
            checkpoint_model = {k[len("encoder_q."):]:v for k,v in checkpoint.items() if k.startswith("encoder_q.")}
            checkpoint_model.update(head_checkpoint)
            checkpoint_model["cls_token"] = checkpoint_model["tokens"]
        else:
            checkpoint = torch.load(lr_checkpoint, map_location='cpu')
            checkpoint_model = checkpoint["model"]
        
        ours_pretrain.interpolate_pos_embed(model, checkpoint_model)
        msg=model.load_state_dict(checkpoint_model,strict=False)
        if linear_probing:
            from timm.models.layers import trunc_normal_
            trunc_normal_(model.head[1].weight, std=0.01)
            torch.nn.init.zeros_(model.head[1].bias)
            print("Init no head")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        print(msg)
        self.model=model
    def forward(self, *args,**kwargs):
         raise SystemExit
    def get(self):
        return self.model

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output 
