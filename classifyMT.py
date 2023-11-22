# %% 
%load_ext autoreload
%autoreload 2

# %%
import HeronImageLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as F
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import crop


# %%
DATA_DIR = '/data/shared/herons/TinaDubach_data'


# %%
"""
    function that shows all the images in a 16 size dataloader
"""
def show_images(images):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images[:16]), nrow=4
).permute(1, 2, 0))
    plt.show()

# %%
imagePropsList = torch.tensor([])
loader = DataLoader(data, batch_size=64, num_workers=3, shuffle=False) # batch_size=64, num_workers=3
for img, index in tqdm(loader):
    #plt.imshow(  img[0].permute(1, 2, 0) )
    cond = img < 0.5
    isM = cond.sum(dim=(1, 2, 3)).gt(3690)
    props = torch.stack((isM, index), -1)
    imagePropsList = torch.cat((imagePropsList, props))

# %%
df = pd.DataFrame(imagePropsList, columns=["isM", "class"])
df.to_csv("imageProps.csv")
  
    
# %%
"""
 shows a badge of 16 images resized to the M or T in the bottom of the initial image
"""
# data = HeronImageLoader.rawHeronDataset(folder="GBU2", transform=T.Compose([T.ToTensor(), lambda im : F.crop(im, top=im.size(dim=1)-85, left=300, height=70, width=70)]))
# loader = DataLoader(data, batch_size=16, num_workers=4, shuffle=False) # batch_size=64, num_workers=3
# for imgs, lbl, inx, badImage in tqdm(loader):
#     print(f'Labels: {type(lbl)}, Index: {inx}')
#     show_images(imgs)

# %%
"""
 shows a badge of 16 images resized to the M or T in the bottom of the initial image
"""
data = HeronImageLoader.rawHeronDataset(folder="GBU2", transform=T.Compose([T.ToTensor(), lambda im : F.crop(im, top=im.size(dim=1)-85, left=300, height=70, width=70)]))
loader = DataLoader(data, batch_size=16, num_workers=4, shuffle=False) # batch_size=64, num_workers=3
for imgs, lbl, inx, badImage in tqdm(loader):
    
