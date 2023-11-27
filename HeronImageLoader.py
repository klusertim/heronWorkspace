# %%
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torch
import glob
import torchvision.transforms as T
import torchvision.transforms.functional as F

# %%
class rawHeronDataset(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    def __init__(self, folder="all", transform=None):
        
        self.folder = folder
        if transform == None:
            self.transform = transform
        self.cropImsize = 85
        self.rawImsize = (2448, 3264) # h x w

        self.imagePaths = self.prepareData()

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_paths[idx])
      
        pathLabel = self.imagePaths[idx] #(imagePaths, classIndex) 
        fileName = os.path.splitext(os.path.basename(pathLabel[0]))[0]   
        try:
            with open(pathLabel[0], "rb") as f:
                img = Image.open(f).convert("RGB")
           
            cropImg = self.transformCrop(img)
            img = self.transformTensor(img)
            return img, cropImg, pathLabel[1], fileName, False
        except OSError:
            # print(f"We had an error loading the image: {pathLabel[0]}")
            return (
                torch.zeros((3, self.rawImsize[0], self.rawImsize[1])),
                torch.zeros((3, self.cropImsize, self.cropImsize)),
                pathLabel[1],
                fileName,
                True
            )
    
    def transformCrop(self, img):
        trsf = T.Compose([T.ToTensor(), lambda im : F.crop(im, top=im.size(dim=1)-self.cropImsize, left=290, height=self.cropImsize, width=self.cropImsize)])
        return trsf(img)
    
    def transformTensor(self, img):
        trsf = T.ToTensor()
        return trsf(img)

    def prepareData(self):
        # Make paths
        files = os.listdir(self.ROOT_DIR)
        folders = [f for f in files if os.path.isdir(self.ROOT_DIR+'/'+f)]
        folders = [(f, i) for i, f in enumerate(folders)]
        # TODO: set index of folders
        if self.folder != "all":
            folders = [(folder, i) for (folder, i) in folders if folder == self.folder]
        
        imagePaths = [(glob.glob(os.path.abspath(os.path.join(self.ROOT_DIR, item[0], "*.JPG"))), item[1]) for item in folders]
        imagePaths = [(filePath, item[1]) for item in imagePaths for filePath in item[0]]
        #print(f'Debug: imagePaths: {imagePaths[100000]}')

        return imagePaths
    
# %%
"""
Try out for test/training set
"""
df1 = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
df2 = pd.read_csv("imageProps.csv", on_bad_lines="warn")
#print(df1.head(10))
#print(df2.head(10))
df = pd.merge(df1, df2, left_on="fotocode", how="right", right_on="ImagePath")
#df = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (~ df["species"].notna())]
df["ImagePath"].unique().size
#df.head(10)
df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & ( df["species"].notna())]["ImagePath"].unique().size

# %%
for i, path in enumerate(df[(df["motion"] == "True") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (~ df["species"].notna())]["ImagePath"]):
    img = Image.open(f'/data/shared/herons/TinaDubach_data/{path[5:9]}/{path}.JPG')
    plt.imshow(img)
    plt.show()
    if i > 50:
        break

# %%
