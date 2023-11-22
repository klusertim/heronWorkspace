from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torch
import glob
import torchvision.transforms as T
import torchvision.transforms.functional as F


class rawHeronDataset(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    def __init__(self, folder="all", transform=None):
        
        self.folder = folder
        if transform == None:
            self.transform = transform
        self.imsize = 85

        self.imagePaths = self.prepareData()

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_paths[idx])
      
        pathLabel = self.imagePaths[idx] #(imagePaths, classIndex)    
        try:
            with open(pathLabel[0], "rb") as f:
                img = Image.open(f).convert("RGB")

            #tensor_image = self.transforms(img)

            if self.transform != None:
                img = self.transform(img)
            return img, pathLabel[1], idx, 0
        except OSError:
            # print(f"We had an error loading the image: {pathLabel[0]}")
            return (
                torch.zeros((3, self.imsize, self.imsize)),
                pathLabel[1],
                idx,
                1
            )
    
    def transform(self, img):
        trsf = T.Compose([T.ToTensor(), lambda im : F.crop(im, top=im.size(dim=1)-self.imsize, left=290, height=self.imsize, width=self.imsize)])
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
        # for item in folders:
        #     print(glob.glob(os.path.abspath(os.path.join(self.ROOT_DIR, item[0], "*.JPG"))))s
        imagePaths = [(filePath, item[1]) for item in imagePaths for filePath in item[0]]
        #print(f'Debug: imagePaths: {imagePaths[100000]}')

        return imagePaths

