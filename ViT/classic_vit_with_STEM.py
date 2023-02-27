
import torch
import torch.nn as nn
import numpy as np
from mulit_head_attention import MultiHeadAttention
from einops import repeat, rearrange
from torch import Tensor

from einops.layers.torch import Rearrange, Reduce

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Replicator(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model;
    def forward(self, x):
        x = self.model(x,x,x)
        return x[0]

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4, forward_drop_p: float = 0., ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ExtractCLS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        #print(x.shape)
        return x[:, 0]
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            #ExtractCLS(),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ParallelScale(nn.Module):
    def __init__(self, model1, embeder1, model2, embeder2) -> None:
        super().__init__()
        self.model1 = model1
        self.embeder1 = embeder1
        self.model2 = model2
        self.embeder2 = embeder2

    def forward(self, x):
        x1 = self.model1(x)
        x1 = self.embeder1(x1)
        y1 = self.model2(x)
        y1 = self.embeder2(y1)

        out = torch.cat([x1,y1], dim=2)
        return out
class Embedder(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, n_patches: int = 16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            #nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            #nn.Linear(patch_size * patch_size * in_channels, emb_size),
            #nn.Linear(emb_size, emb_size//2),
            #nn.Linear(emb_size//2, emb_size//4)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(in_channels+1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        #x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
class ConvolutionalSTEM(nn.Module):
    def __init__(self, channels= [3, 24, 48, 96, 192], emb_size: int = 768):
        super().__init__()

        self.convs =nn.ModuleList() 
        for i, c in enumerate(channels[:-1]):
            self.convs.append(nn.Sequential(
                nn.Conv2d(c, channels[i+1], 3, stride=2),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),

            ))
        self.final = nn.Conv2d(channels[-1], channels[-1], 1)
        self.transform = Rearrange('b c h w -> b c (h w)')

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return self.transform(self.final(x))




class ViTSTEM(nn.Sequential):
    def __init__(self,     
                channels =  [3, 24, 48, 96],
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224, 
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        outsize = (patch_size-12)
        #emb_size = (outsize**2)*out_channels

        super().__init__(
            ConvolutionalSTEM(channels=channels),
            Embedder(channels[-1], outsize, emb_size, n_patches=channels[-1]),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class Printer(nn.Module):
    def forward(self,x):
        print(x.shape)
        return x


if __name__=="__main__":
    x = torch.randn((16, 3, 299, 299))
    model = ViTSTEM(
        channels=[3, 24, 48, 96],
        img_size=299,
        patch_size=299//13, 
        emb_size=1296,
        num_heads=3,
        depth=1,
        n_classes=5, 
        dropout=0.1,
        forward_expansion=1)
    print(model(x).shape)
    #print(TransformerEncoderBlock()(patches_embedded).shape)