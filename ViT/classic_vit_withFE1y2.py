
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
    def __init__(self, embedder, feature_extractor, n_patches : int = 16, fe_size :int=0, emb_size=768):
        super().__init__()
        self.embedder = embedder
        self.feature_extractor = feature_extractor
        self.n_patches = n_patches
        self.fe_size = fe_size
        #assert fe_size % emb_size == 0, "Cant fit features into embedding, sizes doesnt fit"
        #self.n_features_per_embbeding = fe_size//n_patches**2
        self.n_features_per_embbeding = emb_size
        print(f"Assigning {fe_size}/{n_patches**2}={self.n_features_per_embbeding} elements per embedding")
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


        return embbedings, features

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
    def __init__(self, emb_size: int = 768, n_classes: int = 1000, fe_size : int = 0):
        super().__init__()


        self.reduce = Reduce('b n e -> b e', reduction='mean')
        #self.reduce2 = Reduce('b n e -> b e', reduction='mean')
        self.layernorm = nn.LayerNorm(emb_size)
        self.layernormfea = nn.LayerNorm(fe_size)
        self.linear = nn.Linear(emb_size+fe_size, n_classes)

    def forward(self,x):

        emb, fea = x

        inputVal = self.reduce(emb)
        inputVal = self.layernorm(inputVal)
        fea = self.reduce(fea)
        fea = self.layernormfea(fea)
        concatted = torch.cat([inputVal, fea], dim=1)
        return self.linear(concatted)

class ViTWithFE(nn.Sequential):
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
        n_patches = img_size//patch_size
        self.patchEmbedding = PatchEmbedding(self.in_channels, self.patch_size, self.emb_size, self.img_size)
        self.combine = CombineFEwithEmbed(embedder= self.patchEmbedding,n_patches=img_size//patch_size,feature_extractor=self.feature_extractor,fe_size=fe_size, emb_size=emb_size)
        
        # self.transformerEncoder = TransformerEncoder(self.depth, emb_size=self.emb_size+fe_size//n_patches**2, **self.kwargs)
        self.transformerEncoder = TransformerEncoder(self.depth, emb_size=self.emb_size*2, **self.kwargs)
        # self.classificationHead = ClassificationHead(self.emb_size+fe_size//n_patches**2, self.n_classes, fe_size=fe_size//n_patches**2)
        self.classificationHead = ClassificationHead(self.emb_size*2, self.n_classes, fe_size=emb_size)
    def forward(self, x):
        emb, fea = self.combine(x)
        out = self.transformerEncoder(emb)
        out = self.classificationHead((out, fea))
        return out
        

if __name__=="__main__":
    x = torch.randn((16, 3, 299, 299))
    m = timm.create_model('inception_v3', pretrained=True, num_classes=0, global_pool='')
    for param in m.parameters():
        param.requires_grad = False

    output_shape = m(x).shape
    outputsize = output_shape[1]*output_shape[2]*output_shape[3]
    print(output_shape)
    model = ViTWithFE(patch_size=23, img_size=299, forward_expansion=2, depth=6, emb_size=512, feature_extractor=m, fe_size=outputsize, num_heads=6)
    #patches_embedded = model(x)
    #print(patches_embedded.shape)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.2f}MB'.format(size_all_mb))