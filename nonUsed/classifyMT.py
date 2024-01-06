# %% 
# %load_ext autoreload
# %autoreload 2

# %%
import HeronImageLoader
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random


# %%
DATA_DIR = '/data/shared/herons/TinaDubach_data'


# %%
"""
    function that shows all the images in a 16 size dataloader
"""
# def show_images(images, textarr = []):
#     plt.figure(figsize=(10, 6))
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(make_grid((images[:16]), nrow=4
# ).permute(1, 2, 0))
#     ax = np.array(ax)
#     if len(textarr) == 16:
#         for i, axs in enumerate(ax.reshape(-1)):
#             axs.text(0, 0, str(textarr[i])).set_fontsize("large")
#         print(textarr)
#     plt.show()


# %%
"""
decides if the image is grayscale and if there's an M in the bottom of the image
"""
def decideGrayscale(imgBatch: torch.Tensor): #img: 3 x h x w
    _, _, h, w = imgBatch.shape
    
    # sample 10 random pixels
    selectH = random.choices(range(h-200), k=10)
    selectW = random.choices(range(w), k=10)
    sampleBatch = imgBatch[:, :, selectH, selectW]
    stdSum = sampleBatch.std(dim=1).sum(dim=1)
    isGrayscale = stdSum < 0.1
    # print(stdSum)
    # show_images(imgBatch, isGrayscale)
    return isGrayscale

def decideM(imgBatch: torch.Tensor):
    cond = imgBatch < 0.5
    return cond.sum(dim=(1, 2, 3)).gt(3690)

if __name__ == '__main__':
    # %%
    """
    Analize the data for interesting cameras
    """
    cameraDataDF = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
    #cameraDataDF.describe()
    folders = cameraDataDF.groupby(["camera"]).size().sort_values(ascending=False).index
    lastCSVName = "imageProps.csv"
    for folderName in folders[1:10]:
    # %%
        print(f"Working on folder: {folderName}")
        data = HeronImageLoader.rawHeronDataset(folder=folderName)
        # print(data.imagePaths[:10])
        # %%
        """
        loads the images and their properies into an array
        """
        imagePropsList = np.empty([1, 5])
        loader = DataLoader(data, batch_size=64, num_workers=2, shuffle=False) # batch_size=64, num_workers=3
        for rawImg, cropImg, lbl, path, badImage in tqdm(loader):
            grayScale = decideGrayscale(rawImg)
            isM = decideM(cropImg)
            props = np.stack((lbl, path, badImage, isM, grayScale), -1)
            imagePropsList = np.concatenate((imagePropsList, props))
            #print(imagePropsList)


        # %%
        """
            save to csv
        """
        df = pd.DataFrame(imagePropsList, columns=["cam", "ImagePath", "badImage", "motion", "grayscale"])
        try:
            dfOld = pd.read_csv(lastCSVName)
            df = pd.concat([dfOld, df])
            print("successfully concat with old DataFrame")
        finally:
            thisCSVName = f"imageProps{folderName}.csv"
            df.to_csv(thisCSVName)
            print(f"saved work in {thisCSVName}")
            lastCSVName = thisCSVName