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
from vit_overlap_random import ViT
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16,vit_b_16, ViT_B_16_Weights
from torchvision.transforms.functional import normalize


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
N_PATCHES = 14

transform = Compose([
                Resize([IMAGE_WIDTH, IMAGE_HEIGHT]),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

torch.manual_seed(0)

MODEL_NAME="ViT-S14_pretrained_overlap_lg_IN1K"

MAX_CHECKPOINTS = 3

assert IMAGE_WIDTH%N_PATCHES==0

PATCH_WIDTH = IMAGE_WIDTH//N_PATCHES
PATCH_HEIGHT = IMAGE_HEIGHT//N_PATCHES


BATCH_SIZE = 1024
ACCUMULATE_BATCH_SIZE=1024

N_EPOCHS = 90
LR = 0.001
import os


train_set = ImageFolder(root='./../datasets/ILSVRC/Data/CLS-LOC/train', transform=transform)
test_set = ImageFolder(root='./../datasets/ILSVRC/Data/CLS-LOC/val', transform=transform)
print(len(test_set.classes))
print(len(train_set.classes))
print(train_set.classes == test_set.classes)

train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

print(f"ROI divided into regions of {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS}\nUsing {N_PATCHES}x{N_PATCHES} patches of {PATCH_WIDTH}x{PATCH_HEIGHT}x{IMAGE_CHANNELS}")
model = ViT(
        in_channels= IMAGE_CHANNELS,
        img_size=IMAGE_WIDTH,
        patch_size=IMAGE_WIDTH//N_PATCHES,
        emb_size=384,
        num_heads=6,
        depth=12,
        n_classes=len(train_set.classes),
        dropout=0.0,
        forward_expansion=4,
        r=(N_PATCHES**2)*2
    ).to(device)

#import timm
#model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000).to(device)

model= nn.DataParallel(model)

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
#model = vit_b_16('DEFAULT', progress=True).to(device)

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

accum_iterations = ACCUMULATE_BATCH_SIZE/BATCH_SIZE
print(f"Target BATCH_SIZE of {ACCUMULATE_BATCH_SIZE}: Accumulating in {accum_iterations} batches of {BATCH_SIZE}")
# Training loop
MODEL_PATH= "models/"


from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_warmup as warmup


optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)
criterion = CrossEntropyLoss()
bestLoss = 999999;
bestAcc = 0;

train_losses = []
val_losses = []

import transformers
num_total_steps = (len(train_set) // BATCH_SIZE) * N_EPOCHS
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=10000, 
                                                        num_training_steps=num_total_steps)
#lr_scheduler = CosineAnnealingLR(optimizer, T_max=10000/accum_iterations)
#warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
scaler = torch.cuda.amp.GradScaler()

files = []

for epoch in trange(1, N_EPOCHS+1, desc="Training"):
    train_loss = 0.0
    for bid, batch in enumerate(tqdm(train_loader)):
        x, y = batch
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)
        scaler.scale(loss).backward()


        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        #with warmup_scheduler.dampening():
        #    lr_scheduler.step()

    train_losses.append(train_loss)

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.5f}")

# Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Correct: {correct} Test accuracy: {correct / total * 100:.3f}%")
        if correct/total*100 > bestAcc:
            bestAcc = correct/total*100
        val_losses.append(test_loss)
        if test_loss < bestLoss:
            if len(files)==MAX_CHECKPOINTS:
                os.remove(files[0])
                files = files[1:]
            files.append(MODEL_PATH+MODEL_NAME+"-Epoch:"+str(epoch)+"_Acc:"+str(bestAcc)+".pth")
            print("Model Saved!")
            torch.save(model.state_dict(), files[-1])
            bestLoss = test_loss


arr = np.array(train_losses)
arr2 = np.array(val_losses)
arr.tofile("plots/"+MODEL_NAME+"_train_loss-"+str(LR)+"lr_Acc:"+str(bestAcc)+".csv", sep = ',')  
arr2.tofile("plots/"+MODEL_NAME+"_validation_loss-"+str(LR)+"lr_Acc:"+str(bestAcc)+".csv", sep = ',')  
# %%
fig, ax = plt.subplots()

ax.plot(np.arange(N_EPOCHS), train_losses)
ax.plot(np.arange(N_EPOCHS), val_losses)
fig.savefig('plots/'+MODEL_NAME+'_losses-'+str(LR)+'lr_Acc:'+str(bestAcc)+'.png')


