import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        #self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.q = nn.Linear(emb_size, emb_size)
        self.k = nn.Linear(emb_size, emb_size)
        self.v = nn.Linear(emb_size, emb_size)
        #self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qq = self.q(x)
        kk = self.k(x)
        vv = self.v(x)
        #qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        #qkv = torch.cat((qq,kk,vv), dim=2)
        #qkv = rearrange(qkv, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        qq = rearrange(qq, "b n (h d) -> b h n d", h=self.num_heads)
        kk = rearrange(kk, "b n (h d) -> b h n d", h=self.num_heads)
        vv = rearrange(vv, "b n (h d) -> b h n d", h=self.num_heads)
        #queries, keys, values = qkv[0], qkv[1], qkv[2]
        queries, keys, values = qq, kk, vv
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        #att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

if __name__=="__main__":
    x = torch.rand((2, 170, 768))
    mha = MultiHeadAttention()
    x = mha(x)
    print(x.shape)