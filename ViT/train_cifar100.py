import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor,Resize,Compose, Normalize
from torchvision import datasets
from classic_vit import ViT
from torchvision.datasets import CIFAR100
#from torchvision.models import vit_b_16, ViT_B_16_Weights

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
N_PATCHES = 16

transform = Compose([
                #Resize([IMAGE_WIDTH, IMAGE_HEIGHT]),
                ToTensor()
            ])

torch.manual_seed(0)


MAX_CHECKPOINTS = 3

assert IMAGE_WIDTH%N_PATCHES==0

PATCH_WIDTH = IMAGE_WIDTH//N_PATCHES
PATCH_HEIGHT = IMAGE_HEIGHT//N_PATCHES


# 20% of data goes to test
TEST_SPLIT = 0.2
BATCH_SIZE = 1024

N_EPOCHS = 10
LR = 0.00001
import os


train_set = CIFAR100(root='./../datasets', train=True, download=True, transform=transform)
test_set = CIFAR100(root='./../datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

print(f"ROI divided into regions of {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS}\nUsing {N_PATCHES}x{N_PATCHES} patches of {PATCH_WIDTH}x{PATCH_HEIGHT}x{IMAGE_CHANNELS}")
model = ViT(
        in_channels= IMAGE_CHANNELS,
        img_size=IMAGE_WIDTH,
        patch_size=IMAGE_WIDTH//N_PATCHES,
        emb_size=768,
        num_heads=12,
        depth=12,
        n_classes=len(train_set.classes),
        dropout=0.1,
        forward_expansion=4
    ).to(device)


# from vit_pytorch import ViT

# model = ViT(
#     image_size = IMAGE_WIDTH,
#     channels=IMAGE_CHANNELS,
#     patch_size = N_PATCHES,
#     num_classes = len(train_set.classes),
#     dim = 768,
#     depth = 12,
#     heads = 12,
#     mlp_dim = 3702,
#     dropout = 0.1,
#     emb_dropout = 0.1
# ).to(device)
# # import timm
# # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100).to(device)
# model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT, progress=True).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.2f}MiB'.format(size_all_mb))

print(model)

# Training loop
MODEL_PATH= "models/bestNormalCifar100.pth"

optimizer = Adam(model.parameters(), lr=LR)#, betas=(0.9, 0.999))
criterion = CrossEntropyLoss()
bestLoss = 999999;

train_losses = []
val_losses = []
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss)

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

# Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        val_losses.append(test_loss)
        if test_loss < bestLoss:
            print("Model Saved!")
            torch.save(model.state_dict(), MODEL_PATH)
            bestLoss = test_loss


# %%
fig, ax = plt.subplots()

ax.plot(np.arange(N_EPOCHS), train_losses)
ax.plot(np.arange(N_EPOCHS), val_losses)
fig.savefig('plots/food101_traditional_losses-'+str(LR)+'lr.png')


