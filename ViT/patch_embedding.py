import torch
from torch import nn
from torch import Tensor
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import numpy as np
class ImagePatcherOverlap(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, img_size: int = 224, returnGrouped:bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_size = img_size

        self.scale1 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.scale2 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.scale3 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        self.scale4 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        if returnGrouped:
            self.transform = Rearrange('b (l) (c s1 s2) -> b (l) (c s1 s2)', s1=patch_size, s2=patch_size)
        else:
            self.transform = Rearrange('b (l) (c s1 s2) -> (b l) c s1 s2', s1=patch_size, s2=patch_size)

        #self.scale3 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        #self.scale4 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)

        #self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)


    def forward(self, x:Tensor):
        out1 = self.scale1(x)
        start = int(np.ceil(self.patch_size/2.0))
        end = int(self.img_size - np.floor(self.patch_size/2.0))
        out2 = self.scale2(x[:, :, start:end, start:end])
        out3 = self.scale3(x[:, :, :, start:end])
        out4 = self.scale4(x[:, :, :, start:end])

        return self.transform(torch.cat([out1, out2, out3, out4], dim=1))

    def toPatchesTensor(self, x):
        x= self.transform(x)
        return rearrange(x, "b (h w) (c s1 s2) -> b h w c s1 s2", s1=self.patch_size, s2=self.patch_size, w=self.img_size//self.patch_size, h=self.img_size//self.patch_size)

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


class ImagePatcher(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, img_size: int = 224, returnGrouped:bool=True):
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_size = img_size
        super().__init__()

        #self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        if returnGrouped:
            self.transform = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        else:
            self.transform = Rearrange('b c (h s1) (w s2) -> (b h w) c s1 s2', s1=patch_size, s2=patch_size)

    def forward(self, x:Tensor):
        return self.transform(x)

    def toPatchesTensor(self, x):
        x= self.transform(x)
        return rearrange(x, "b (h w) (c s1 s2) -> b h w c s1 s2", s1=self.patch_size, s2=self.patch_size, w=self.img_size//self.patch_size, h=self.img_size//self.patch_size)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b h w e')
            #ImagePatcher(in_channels, patch_size, img_size),
            #nn.Linear(patch_size * patch_size * in_channels, emb_size)

        )
        #self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        #self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
        self.p_enc_2d_model_sum = Summer(PositionalEncoding2D(emb_size))

        self.rearrange2 = Rearrange('b h w e -> b (h w) e')


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # prepend the cls token to the input
        #x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x = self.p_enc_2d_model_sum(x)
        x = self.rearrange2(x)

        #x += self.positions
        return x

class PatchEmbedder(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, r: int = 16):
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.r = r
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

        self.rearrange2 = Rearrange('b h w e -> b (h w) e')


    def forward(self, x: Tensor) -> Tensor:
        tensor = x[0]
        pos = x[1]
        # b h w c s1 s2
        tensor = self.projection(tensor)
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        #x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding

        #x += self.positions
        tensor = tensor + overlap_positional_encoding(tensor.shape[0], self.r, self.emb_size, pos).to(tensor.device)
        return tensor

import torch.nn.functional as F

def random_cropb(input_tensor, crop_size:int):
    batch_size, _, height, width = input_tensor.shape
    
    # Generate random crop coordinates for each image in the batch
    crop_top = torch.randint(0, height - crop_size, (batch_size,))
    crop_left = torch.randint(0, width - crop_size, (batch_size,))

    # Create a grid of coordinates for each image in the batch
    grid = torch.zeros((batch_size, crop_size, crop_size, 2))
    grid[:, :, :, 0] = torch.arange(0, crop_size).view(1, -1, 1).repeat(batch_size, 1, crop_size)
    grid[:, :, :, 1] = torch.arange(0, crop_size).view(1, 1, -1).repeat(batch_size, crop_size, 1)
    # Add the crop coordinates to the grid
    grid[:, :, :, 0] += crop_top.view(-1, 1, 1) 
    grid[:, :, :, 1] += crop_left.view(-1, 1, 1)

    grid[:, :, :, 0] = 2*(grid[:, :, :, 0])/height-1 
    grid[:, :, :, 1] = 2*(grid[:, :, :, 1])/width-1 
    # Perform the random crop on each image in the batch using grid_sample
    cropped_images = F.grid_sample(input_tensor, grid.to(input_tensor.device), align_corners=True)
    return cropped_images, crop_top, crop_left

def overlap_positional_encoding(batch_size, patches, embedded_dim, positions):
    # Create the positional encoding matrix
    position_encoding = torch.zeros(batch_size, patches, embedded_dim)
    # Get the dimensions of the positional encoding matrix
    d_batch, d_patches, d_embed = position_encoding.shape
    # Create the indices for the matrix
    i = torch.arange(d_batch).view(-1, 1, 1).repeat(1, d_patches, d_embed//2)
    j = torch.arange(d_patches).view(1, -1, 1).repeat(d_batch, 1, d_embed//2)
    # Calculate the sin and cos values for the positional encoding
    position_encoding[:, :, ::2] = torch.sin(positions[i, j, 0] / (10000 ** (2 * torch.arange(d_embed // 2) / d_embed)))
    position_encoding[:, :, 1::2] = torch.cos(positions[i, j, 1] / (10000 ** (2 * torch.arange(d_embed // 2) / d_embed)))
    return position_encoding

class ImagePatcherOverlapRandom(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, img_size: int = 224, r=169, returnGrouped:bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.r = r
        if returnGrouped:
            self.transform = Rearrange('b (l) c s1 s2 -> b (l) (c s1 s2)', c=in_channels, s1=patch_size, s2=patch_size)
        else:
            self.transform = Rearrange('b (l) c s1 s2 -> (b l) c s1 s2', c=in_channels, s1=patch_size, s2=patch_size)

        #self.scale3 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)
        #self.scale4 = Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size)

        #self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)


    def forward(self, x:Tensor):
        ogdevice = x.device
        #x = x.to('cpu')
        x = x.unsqueeze(0).expand(self.r, x.shape[0],-1,-1,-1).transpose(0,1)

        #indices = torch.arange(1024).repeat(98)
        #x =x[indices]
        #out = torch.Tensor().to(ogdevice)
        #out = self.crop(x)
        #gsh = random_crop_grid(x, self.gs)
        #out = torch.nn.functional.grid_sample(x, gsh, align_corners=True)
        # crop the images using grid_sample()
        #print(x.shape)
        #x = random_cropb(x[0], (self.patch_size,self.patch_size))
        #x = F.grid_sample(images, corners.to(ogdevice), align_corners=True)
        #x = arandom_crop(x, (self.patch_size, self.patch_size));
        out = torch.Tensor().to(ogdevice)
        pos = torch.Tensor()

        for b in range(x.shape[0]):
            ovr, pot_top, pot_left = random_cropb(x[b], crop_size=self.patch_size)
            out = torch.cat([out, ovr.unsqueeze(0)])
            stack = torch.stack([pot_top, pot_left],dim=-1)
            pos = torch.cat([pos, stack.unsqueeze(0)],dim=0)
        return (self.transform(out), pos)

    def toPatchesTensor(self, x):
        x= self.transform(x)
        return rearrange(x, "b (h w) (c s1 s2) -> b h w c s1 s2", s1=self.patch_size, s2=self.patch_size, w=self.img_size//self.patch_size, h=self.img_size//self.patch_size)





if __name__=="__main__":
    import matplotlib.pyplot as plt
    x = torch.randn((2, 3, 299, 299))
    patches = (ImagePatcherOverlapRandom(patch_size=23, img_size = 299, r=169)(x))

    #patches = (PatchEmbedding()(x))
    print(patches.shape)

    #N_PATCHES = 14
    #gridSpec = dict(top = 1, bottom = 0, right = 1, left = 0, hspace = 1, wspace = 1)
    #patches = patches.reshape((1, N_PATCHES,N_PATCHES,16,16,3))
    #print(patches.shape)

    # plt.imshow(torch.transpose(x[0],0,2))
    # fig, axes = plt.subplots(N_PATCHES, N_PATCHES ,  sharex=True, sharey=True, gridspec_kw=gridSpec)
    # for i in range((N_PATCHES)):
    #     for j in range((N_PATCHES)):

    #         axes[i,j].imshow(patches[0, i, j].detach().numpy(), interpolation=None)
    # plt.show()

