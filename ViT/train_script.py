# %%
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision import datasets
from collections import Counter

from classic_vit import MyViT
from her2_dataset import HER2Dataset

transform = ToTensor()

torch.manual_seed(0)

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
IMAGE_CHANNELS = 3
N_PATCHES = 13

MAX_CHECKPOINTS = 3

assert IMAGE_WIDTH%N_PATCHES==0

PATCH_WIDTH = IMAGE_WIDTH//N_PATCHES
PATCH_HEIGHT = IMAGE_HEIGHT//N_PATCHES


DATASET_PATH='../datasets/HER2_gastric_5classes'
# 20% of data goes to test
TEST_SPLIT = 0.2
BATCH_SIZE = 128

N_EPOCHS = 5
LR = 0.005


# %%

dataset = HER2Dataset(DATASET_PATH, transform=ToTensor())
train_dataset, test_dataset = dataset.genSplits(TEST_SPLIT)
train_loader, test_loader = dataset.getDataLoaders(BATCH_SIZE)

print(f"Dataset split:\nTrain: {len(train_dataset)}\nTest: {len(test_dataset)}")



# %%
train_count, test_count = dataset.getSampleCountByClass()

fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(10,4))
fig.suptitle("Classes distribution")
ax[0].set_ylabel("Count")
ax[0].bar(list(zip(*train_count))[0], list(zip(*train_count))[1], tick_label=list(zip(*train_count))[2])
ax[0].set_title("Train Split")
ax[1].bar(list(zip(*test_count))[0], list(zip(*test_count))[1], tick_label=list(zip(*test_count))[2])
ax[1].set_title("Test Split")

# %%

data_iter = iter(train_loader)
images, labels = next(data_iter)

class_first_sample = list(map(lambda x: np.where(x == np.array(labels))[0][0], dataset.class_to_idx.values()))


print(images.shape)

fig, axes = plt.subplots(tight_layout=True, ncols=len(class_first_sample))
fig.suptitle("First sample of the dataset by class")

for ii in range(len(class_first_sample)):
    ax = axes[ii]
#     helper.imshow(images[ii], ax=ax, normalize=False)
    ax.set_title(dataset.classes[labels[class_first_sample[ii]]])
    ax.imshow(torch.transpose(images[class_first_sample[ii]], 0, 2))

# %%
data_iter = iter(train_loader)
images, labels = next(data_iter)

fig, axes = plt.subplots(tight_layout=True, figsize=(8,8))
axes.imshow(torch.transpose(images[1], 0, 2), interpolation=None)
axes.vlines(range(0,IMAGE_WIDTH,PATCH_WIDTH), 0, IMAGE_HEIGHT)
axes.hlines(range(0,IMAGE_HEIGHT,PATCH_HEIGHT), 0, IMAGE_WIDTH)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

print(f"ROI divided into regions of {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS}\nUsing {N_PATCHES}x{N_PATCHES} patches of {PATCH_WIDTH}x{PATCH_HEIGHT}x{IMAGE_CHANNELS}")
model = MyViT((IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT), n_patches=N_PATCHES, out_d=len(dataset.classes), trainable_pos_embedding=True, dropout=0.2).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)


# %%
# Training loop
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
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


# %%



