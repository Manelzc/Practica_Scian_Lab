
import torch
import torch.nn as nn
import numpy as np
from mulit_head_attention import MultiHeadAttention
from einops import repeat, rearrange
from torch import Tensor

from patch_embedding import PatchEmbedding, ImagePatcher, ImagePatcherOverlapRandom
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
        self.model = model
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

class ConvolutionalStem(nn.Sequential):
    def __init__(self, in_channels:int=3, out_channels:int=12, n_patches:int=13, dilation:int=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0, dilation=dilation),
            nn.GELU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            #nn.Conv2d(out_channels, out_channels*2, 3, stride=1, padding=0, dilation=dilation),
            #nn.GELU(),
            #nn.Conv2d(out_channels*2, out_channels*3, 3, stride=1, padding=0, dilation=dilation),
            #nn.GELU(),
            #nn.Conv2d(out_channels*3, out_channels*4, 3, stride=1, padding=0, dilation=dilation),
            #nn.GELU(),
            #nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, dilation=dilation),
            #nn.GELU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #nn.Conv2d(out_channels, out_channels*2, 5, stride=1, padding=0),
            #nn.GELU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Rearrange('(b i) c s1 s2 -> b (i) (c s1 s2)', i=n_patches)
        )

class Embedder(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, n_patches: int = 16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            #nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
            #nn.Linear(emb_size, emb_size//2),
            #nn.Linear(emb_size//2, emb_size//4)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((n_patches) + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
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




class MultiscaleViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                out_channels: int = 12,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224, 
                depth: int = 12,
                n_classes: int = 1000,
                r=169,
                **kwargs):
        outsize = (patch_size-12)
        #emb_size = (outsize**2)*out_channels
        n_patches = r 
        super().__init__(
            ImagePatcherOverlapRandom(in_channels, patch_size, img_size, r=n_patches),
            #ParallelScale(
                #Printer(),
                ConvolutionalStem(in_channels, out_channels, n_patches=(n_patches) ),
                Embedder(out_channels, outsize, emb_size, n_patches=n_patches),
                #ConvolutionalStem(in_channels, out_channels, n_patches=img_size//patch_size, dilation=2),
                #Embedder(out_channels, 15, emb_size, n_patches=img_size//patch_size),
                #ConvolutionalStem(in_channels, out_channels, n_patches=img_size//patch_size, dilation=3),
                #Embedder(out_channels, 11, emb_size, n_patches=img_size//patch_size)
            #),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class Printer(nn.Module):
    def forward(self,x):
        print(x.shape)
        return x


if __name__=="__main__":
    x = torch.randn((16, 3, 299, 299))
    model = MultiscaleViT(
        in_channels= 3, 
        out_channels=24,
        img_size=299,
        patch_size=299//13, 
        emb_size=768,
        num_heads=16,
        depth=1,
        n_classes=5, 
        dropout=0.1,
        forward_expansion=1)
    model(x)
    #print(TransformerEncoderBlock()(patches_embedded).shape)