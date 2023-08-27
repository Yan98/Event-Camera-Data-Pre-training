from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, mask_ratio, linear_probing=  False, add_layernorm = False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        embed_dim = kwargs["embed_dim"]
        self.cls_token = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features * 2) if add_layernorm else  nn.Identity(), 
            nn.Linear(self.num_features * 2, self.head.weight.size(0))
            )
        self.linear_probing = linear_probing
        #torch.nn.init.zeros_(self.head[1].weight)
        #torch.nn.init.zeros_(self.head[1].bias)
        
    def density_masking(self, x,density):

        if self.mask_ratio  < 1/196:
            return x
            
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        density = -density.flatten(2).squeeze(1)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(density, dim=1)  # ascend: small is keep, large is remove
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked

    def forward_features(self, x):
        density = None
        if self.mask_ratio >= 1/196:
            with torch.no_grad():
                density = nn.AvgPool2d(16,16)(x/2+0.5)
                density = density.mean(1,True)
        
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.density_masking(x, density)

        x = torch.cat((self.cls_token.expand(B, -1, -1) , x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        
        return torch.cat((x[:, 0], x[:, 1]),1) #x[:, 0] * 1/2 + x[:, 1] * 1/2
    
    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x[:,1:])
        return features



def vit_patch16_small(in_chans,mask_ratio,**kwargs):
    model = VisionTransformer(
        mask_ratio = mask_ratio, patch_size=16, embed_dim=384, depth=12, in_chans=in_chans, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_patch16_base(in_chans,mask_ratio,**kwargs):
    model = VisionTransformer(
        mask_ratio = mask_ratio, patch_size=16, embed_dim=768, depth=12, in_chans=in_chans, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_patch16_large(in_chans,mask_ratio,**kwargs):
    model = VisionTransformer(
        mask_ratio = mask_ratio, patch_size=16, embed_dim=1024, depth=24, in_chans=in_chans, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

ours_model_small_ft = vit_patch16_small
ours_model_base_ft = vit_patch16_base
ours_model_large_ft = vit_patch16_large


