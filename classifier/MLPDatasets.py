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
import sys
sys.path.append("/data/tim/heronWorkspace/dataPreprocessing")
from classifyMotionGray import ClassifyMotionGray
import random
from sklearn.model_selection import train_test_split

MEAN = 0.5
STD = 0.5

class MLPDatasetValidated(Dataset): # from validation of SBU4
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    imsize = (2448-100, 3264) # h x w
    def __init__(self, set="train", resize_to = (216, 324), transform=None):

            
        self.set = set
        self.imsize = resize_to

        df = pd.read_csv("datasetValidation.csv")
        unwanted = df.columns[df.columns.str.startswith('Unnamed')]
        df.drop(unwanted, axis=1, inplace=True)
        dfAnomaly = df[(df["ValidationValue"] == 2) | (df["ValidationValue"] == 4)]["ImagePath"]
        dfNormal = df[(df["ValidationValue"] <= 0)].sample(n=len(dfAnomaly))["ImagePath"]

        # print(dfAnomaly.head(10))
        trainSetAnomaly = dfAnomaly.sample(frac=0.8,random_state=200)
        trainSetNormal = dfNormal.sample(frac=0.8,random_state=200)
        trainSet = trainSetAnomaly + trainSetNormal

        testValSetAnomaly = dfAnomaly.drop(trainSetAnomaly.index) 
        testValSetNormal = dfNormal.drop(trainSetNormal.index)
        testSetAnomaly = testValSetAnomaly.sample(frac=0.5,random_state=200) 
        testSetNormal = testValSetNormal.sample(frac=0.5,random_state=200)
        testSet = testSetAnomaly + testSetNormal

        valSetAnomaly = testValSetAnomaly.drop(testSetAnomaly.index)
        valSetNormal = testValSetNormal.drop(testSetNormal.index)

        if set == "test":
            self.imagePaths, self.lbl = testSetAnomaly.to_list() + testSetNormal.to_list(), [1 for _ in range(len(testSet)//2)] + [0 for _ in range(len(testSet)//2)]
        elif set == "val":
            self.imagePaths, self.lbl = valSetAnomaly.to_list() + valSetNormal.to_list(), [1 for _ in range(len(valSetAnomaly))] + [0 for _ in range(len(valSetNormal))]
        elif set == "train":
            self.imagePaths, self.lbl = trainSetAnomaly.to_list() + trainSetNormal.to_list(), [1 for _ in range(len(trainSet)//2)] + [0 for _ in range(len(trainSet)//2)]


    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        
        fotocode = self.imagePaths[idx]
        try:
            with open(f'/data/shared/herons/TinaDubach_data/{fotocode[5:9]}/{fotocode}.JPG', "rb") as f:
                img = Image.open(f).convert("RGB")
                

            img = self.transform(img)
            return img, self.lbl[idx], idx
        except OSError:
            print(f"Error occured loading the image: {fotocode}")
            return (
                torch.zeros((3, self.imsize[0], self.imsize[1])),
                0,
                idx
            )
    
    def transform(self, img):
        trsf = T.Compose([
                T.ToTensor(),
                lambda im : F.crop(im, top=0, left=0, height=2448-190, width=3264),
                T.Resize([self.imsize[0], self.imsize[1]], antialias=True),
                T.Normalize(mean=(MEAN, MEAN, MEAN), std=(STD, STD, STD)),
            ]
        )
        return trsf(img)
    

class MLPDatasetThreeConsecutive(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    imsize = (2448-100, 3264) # h x w
    def __init__(self, set="train", cameras = ["GBU3"],  resize_to = (216, 324), transform=None):           
        self.set = set
        self.imsize = resize_to

        allFeatures = []
        for camera in cameras:
            allFeatures.extend(self.generateFeaturesBalanced(camera))

        trainSet, testSet = train_test_split(allFeatures, test_size=0.2, random_state=1)

        trainSet, valSet = train_test_split(trainSet, test_size=0.25, random_state=1)

        if set == "test":
            self.imagePaths = testSet
        elif set == "val":
            self.imagePaths = valSet
        elif set == "train":
            self.imagePaths = trainSet


    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        
        imgs, lbl = self.imagePaths[idx]
        try:
            prevImg, img, nextImg = [self.loadAndTransform(x) for x in imgs]     
            return [prevImg, img, nextImg], lbl, idx
        except OSError:
            print(f"Error occured loading the image: {img}")
            zero = torch.zeros((3, self.imsize[0], self.imsize[1]))
            return (
                [zero]*3,
                0,
                idx
            )
    
    def transform(self, img):
        trsf = T.Compose([
                T.ToTensor(),
                lambda im : F.crop(im, top=0, left=0, height=2448-190, width=3264),
                T.Resize([self.imsize[0], self.imsize[1]], antialias=True),
                T.Normalize(mean=(MEAN, MEAN, MEAN), std=(STD, STD, STD)),
            ]
        )
        return trsf(img)
    
    ### HELPER FUNCTIONS ###

    def loadAndTransform(self, path):
        with open(f'/data/shared/herons/TinaDubach_data/{path[5:9]}/{path}.JPG', "rb") as f:
            img = Image.open(f).convert("RGB")
            img = self.transform(img)
        return img

    def generateFeaturesBalanced(self, camera : str):
        try:
            df = pd.read_csv(f"/data/tim/heronWorkspace/MotionGrayClassification/classifiedMotionGray{camera}.csv")
            dfPreClassified = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
        except:
            return []
        
        df = df[(df["grayscale"] == False) & (df["badImage"] == False)].sort_values(by=['ImagePath'])

        #TODO: evtl filter out non-unique ImagePaths

        sortedPaths = df["ImagePath"].tolist()
        motions = df["motion"].astype(int).tolist()
        features = []

        for i, row in enumerate(sortedPaths):
            _, _, nrCurr = row.split("_")
            if i > 0 and i < len(sortedPaths)-1:
                _, _, nrPrev = sortedPaths[i-1].split("_")
                _, _, nrNext = sortedPaths[i+1].split("_")
                if int(nrPrev) + 1 == int(nrCurr) and int(nrNext)-1 == int(nrCurr):
                    features.append([(sortedPaths[i-1], row, sortedPaths[i+1]), motions[i]])
        
        featuresNormal = [feature for feature in features if feature[1] == 0]
        featuresMotion = [feature for feature in features if feature[1] == 1]
        minLen = min(len(featuresNormal), len(featuresMotion))

        random.seed(1)
        featuresNormal = random.sample(featuresNormal, minLen)
        featuresMotion = random.sample(featuresMotion, minLen)

        features = featuresNormal + featuresMotion
        
        return features