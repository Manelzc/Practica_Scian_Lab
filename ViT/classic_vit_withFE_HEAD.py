
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from mulit_head_attention import MultiHeadAttention
import timm

from patch_embedding import PatchEmbedding
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange

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
        
class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000, feature_extractor: nn.Module = None, fe_size : int = 0):
        super().__init__()


        self.reduce = Reduce('b n e -> b e', reduction='mean')
        self.layernorm = nn.LayerNorm(emb_size)
        self.fe = feature_extractor
        self.linear = nn.Linear(emb_size+fe_size, n_classes)

    def forward(self,x):

        original, inputVal = x

        inputVal = self.reduce(inputVal)
        inputVal = self.layernorm(inputVal)

        features = self.fe(original)
        features = features.view(original.shape[0], -1)
        concatted = torch.cat([inputVal, features], dim=1)
        return self.linear(concatted)

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class ViTWithFE(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                feature_extractor: nn.Module = None,
                fe_size : int = 0,
                **kwargs):
        
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.depth = depth
        self.n_classes = n_classes
        self.feature_extractor = feature_extractor
        self.fe_size = fe_size
        self.kwargs = kwargs
        self.patchEmbedding = PatchEmbedding(self.in_channels, self.patch_size, self.emb_size, self.img_size)
        self.transformerEncoder = TransformerEncoder(self.depth, emb_size=self.emb_size, **self.kwargs)
        self.classificationHead = ClassificationHead(self.emb_size, self.n_classes, feature_extractor=self.feature_extractor, fe_size=self.fe_size)
    
    def forward(self, x):
        out = self.patchEmbedding(x)
        out = self.transformerEncoder(out)
        out = (x, out)
        out = self.classificationHead(out)
        return out

if __name__=="__main__":
    x = torch.randn((16, 3, 299, 299))
    m = timm.create_model('inception_v3', pretrained=True, num_classes=0, global_pool='')
    for param in m.parameters():
        param.requires_grad = False

    output_shape = m(x).shape
    outputsize = output_shape[1]*output_shape[2]*output_shape[3]
    print(output_shape)
    patches_embedded = ViTWithFE(patch_size=23, img_size=299, feature_extractor=m, fe_size=outputsize, num_heads=6)(x)
    print(patches_embedded.shape)
    