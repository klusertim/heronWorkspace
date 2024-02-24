# %% 
# %load_ext autoreload
# %autoreload 2

# %%
from ClassifyMotionGrayDataset import MotionGrayHeronDataset
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
from argparse import ArgumentParser
import os


class ClassifyMotionGray():
    
    # %%
    """
    decides if the image is grayscale and if there's an M in the bottom of the image
    """
    def decideGrayscale(self, imgBatch: torch.Tensor): #img: 3 x h x w
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

    def decideM(self, imgBatch: torch.Tensor):
        cond = imgBatch < 0.5
        return cond.sum(dim=(1, 2, 3)).gt(3690)
    
    def classify(self, cameras: [str]):
        """
        Analize the data for interesting cameras
        """
        cameraDataDF = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
        folders = cameraDataDF.groupby(["camera"]).size().index
        for cam in cameras:
            if cam not in folders:
                raise ValueError(f"camera {cam} not found in data")

        allCamString = "".join(cameras)

        for folderName in cameras:
            csvName = f"/data/tim/heronWorkspace/MotionGrayClassification/classifiedMotionGray{folderName}.csv"
            if os.path.isfile(csvName):
                print(f"{csvName} already exists")
                continue

            print(f"Classifying images of camera: {folderName} into grayscale/rgb and motion/static sensor")
            data = MotionGrayHeronDataset(folder=folderName)
            """
            loads the images and their properies into an array
            """
            imagePropsList = np.array([]).reshape(0,5)
            batch_size = 16
            loader = DataLoader(data, batch_size=batch_size, num_workers=2, shuffle=False) # batch_size=64, num_workers=3
            for rawImg, cropImg, _, path, badImage in tqdm(loader):
                grayScale = self.decideGrayscale(rawImg)
                isM = self.decideM(cropImg)
                cam = [folderName] * len(isM)
                props = np.stack((cam, path, badImage, isM, grayScale), -1)
                imagePropsList = np.concatenate((imagePropsList, props))

            """
                save to csv
            """
            df = pd.DataFrame(imagePropsList, columns=["camera", "ImagePath", "badImage", "motion", "grayscale"])
            try:
                df.to_csv(csvName, index=False)
                print(f"saved {csvName}")
            except:
                print(f"couldn't save {csvName}")
                print(f"try to save {folderName}.csv")
                df.to_csv(f"{folderName}.csv", index=False)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--cameras', type=str, nargs="+", required=True, help="camera name/s")
    args = parser.parse_args()

    cameras = args.cameras

    if cameras[0] == "allTrain":
        cameras = ["SBU2", "SBU3", "GBU1", "GBU4", "KBU2", "NEN1", "NEN2", "PSU1", "PSU2", "PSU3", "SGN1", "SGN2"]
    print(f'cameras: {cameras}')

    ClassifyMotionGray().classify(cameras)  
   