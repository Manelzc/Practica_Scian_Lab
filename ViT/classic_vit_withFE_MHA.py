
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

class CombineFEwithEmbed(nn.Module):
    def __init__(self, embedder, feature_extractor, n_patches : int = 16, fe_size :int=0):
        super().__init__()
        self.embedder = embedder
        self.feature_extractor = feature_extractor
        self.n_patches = n_patches
        self.fe_size = fe_size
        #assert fe_size % emb_size == 0, "Cant fit features into embedding, sizes doesnt fit"
        self.n_features_per_embbeding = fe_size//n_patches**2 - 1
        print(f"Assigning {fe_size}/{n_patches**2}-1={self.n_features_per_embbeding} elements per embedding")
        self.cls_token = nn.Parameter(torch.randn(1,1, self.n_features_per_embbeding))

        
    def forward(self, x):
        b, _, _, _ = x.shape

        embbedings = self.embedder(x)
        features = self.feature_extractor(x)
        features = features.view(b, -1)
        features = features[:, :self.n_features_per_embbeding*self.n_patches**2]
        features = features.view(b, self.n_patches**2,self.n_features_per_embbeding)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        features = torch.cat([cls_tokens, features], dim=1)


        embbedings = torch.cat([embbedings, features], dim=2)


        return embbedings

class FeatureExtractor(nn.Module):
    def __init__(self, fe : nn.Module = None, fe_size : int = 0, emb_size : int = 768, n_patches : int = 14) -> None:
        super().__init__()
        self.fe = fe
        #print(fe_size, emb_size*n_patches**2)
        self.targetSize= emb_size*(n_patches**2)
        #self.resizer = nn.Linear(fe_size, emb_size*n_patches**2)
        self.norm = nn.LayerNorm(emb_size)
        self.emb_size =emb_size

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

    def forward(self, x):
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = self.fe(x)
        x = x.view(b, -1)
        x = x[:, :self.targetSize]
        x = x.view(b, -1, self.emb_size)
        x = torch.cat([cls_tokens, x], dim=1)
        #x = self.norm(x)
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
    #def forward(self, x):

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ExtractCLS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x[:,0]
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            #ExtractCLS(),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

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
        self.pe = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.fe = FeatureExtractor(feature_extractor, fe_size=fe_size, emb_size=emb_size, n_patches= img_size//patch_size)
        self.transformer_embedding = TransformerEncoder(depth//2, emb_size=emb_size, **kwargs)
        self.transformer_features = TransformerEncoder(depth//2, emb_size=emb_size, **kwargs)

        self.ch = ClassificationHead(emb_size*2, n_classes)
    def forward(self,x):
        embeddings = self.pe(x)
        features = self.fe(x)

        embeddings = self.transformer_embedding(embeddings)
        features = self.transformer_features(features)
        x = torch.cat([embeddings, features], dim=2)
        #print(x.shape)
        x = self.ch(x)
        return x
if __name__=="__main__":
    x = torch.randn((16, 3, 299, 299))
    m = timm.create_model('inception_v3', pretrained=True, num_classes=0, global_pool='')
    for param in m.parameters():
        param.requires_grad = False

    output_shape = m(x).shape
    outputsize = output_shape[1]*output_shape[2]*output_shape[3]
    patches_embedded = ViTWithFE(patch_size=23, img_size=299, feature_extractor=m, fe_size=outputsize, num_heads=6)(x)
    