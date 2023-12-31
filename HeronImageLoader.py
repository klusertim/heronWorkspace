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
from classifyMotionGray import ClassifyMotionGray

class HeronDatasetCAE(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    imsize = (2448-100, 3264) # h x w
    def __init__(self, set="train", resize_to = (216, 324), cameras = None, transform=None):
        
        # load data
        for i, cam in enumerate(cameras):
            try:
                dfNew = pd.read_csv(f"MotionGrayClassification/classifiedMotionGray{cam}.csv")
            except FileNotFoundError:
                print(f"File not found: MotionGrayClassification/classifiedMotionGray{cam}.csv")
                ClassifyMotionGray().classify([cam])
                dfNew = pd.read_csv(f"MotionGrayClassification/classifiedMotionGray{cam}.csv")
                continue
            if i == 0:
                df = dfNew
            else:
                df = pd.concat([df, dfNew])
            
        self.set = set
        self.imsize = resize_to

        df = df[(df["grayscale"] == "False") & (df["motion"] == "False") & (df["badImage"] == "False")]
        df = df["ImagePath"].unique()
        train=df.sample(frac=0.8,random_state=200)
        testAndVal=df.drop(train.index)
        test=testAndVal.sample(frac=0.5,random_state=200)
        val=testAndVal.drop(test.index)

        if set == "test":
            self.imagePaths, self.lbl = test.to_list(), [0 for _ in range(len(test))]
        elif set == "val":
            self.imagePaths, self.lbl = val.to_list(), [0 for _ in range(len(val))]
        elif set == "train":
            self.imagePaths, self.lbl = train.to_list(), [0 for _ in range(len(train))]


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

MEAN = 0.5
STD = 0.5

class HeronDataset(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    imsize = (2448-100, 3264) # h x w
    def __init__(self, set="train", transform=None, resize_to = (216, 324), sorted=False):
        # TODO: train with more than only one camera
        df1 = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
        df2 = pd.read_csv("/data/tim/heronWorkspace/ImageData/imagePropsSBU4.csv", on_bad_lines="warn")
        df = pd.merge(df1, df2, left_on="fotocode", how="right", right_on="ImagePath")

        if sorted:
            df.sort_values(by=["ImagePath"], inplace=True)
            
        self.set = set
        self.imsize = resize_to

        #TODO: this is not random
        if set == "test":
            self.imagePaths, self.lbl = self.prepareTest(df)
        elif set == "val":
            self.imagePaths, self.lbl = self.prepareVal(df)
        elif set == "onlyPos":
            self.imagePaths, self.lbl = self.prepareOnlyPos(df)
        elif set == "train":
            self.imagePaths, self.lbl = self.prepareTrain(df)
        elif set == "trainMLP":
            self.imagePaths, self.lbl = self.prepareTrainMLP(df)
        elif set == "valMLP":
            self.imagePaths, self.lbl = self.prepareValMLP(df)
        elif set == "testMLP":
            self.imagePaths, self.lbl = self.prepareTestMLP(df)


    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        
        fotocode = self.imagePaths[idx]
        try:
            with open(f'/data/shared/herons/TinaDubach_data/{fotocode[5:9]}/{fotocode}.JPG', "rb") as f:
                img = Image.open(f).convert("RGB")
                # tens = T.ToTensor()(img)
                # print(f'before: {img.shape} and after: {tens.size()} conversion')

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
                #T.RandomHorizontalFlip(p=0.5),
                # T.ToPILImage(),
                T.ToTensor(),
                lambda im : F.crop(im, top=0, left=0, height=2448-190, width=3264),
                T.Resize([self.imsize[0], self.imsize[1]], antialias=True),
                T.Normalize(mean=(MEAN, MEAN, MEAN), std=(STD, STD, STD)),
            ]
        )
        return trsf(img)

    """ TODO: at the moment, splits are made only for training with one camera
    splits: 0.8 train, 0.1 val, 0.1 test, where test contains anomalous frames as well
    """
    def prepareTrain(self, df: pd.DataFrame):
        pathList = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & ( df["species"].isna())]["ImagePath"].to_list()
        lenTest = int(len(pathList) * 0.9)
        pathList = pathList[:lenTest]
        return pathList, [0 for _ in range(len(pathList))]

    def prepareVal(self, df: pd.DataFrame):
        pathList = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].isna())]["ImagePath"].to_list()
        pathLen = len(pathList)
        lenTest = int(pathLen * 0.9)
        pathList = pathList[lenTest:lenTest+int(pathLen*0.1)]
        return pathList, [0 for _ in range(len(pathList))]
    
    def prepareTest(self, df: pd.DataFrame):
        pathListNeg = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].isna())]["ImagePath"].to_list()
        # only pos we're sure of
        pathListPos = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].notna())]["ImagePath"].to_list() #TODO: remove false at positive set
        negPathLen = len(pathListNeg)
        lenTest = int(negPathLen * 0.9)
        pathListNeg = pathListNeg[lenTest+int(lenTest*0.1):]
        return (pathListNeg + pathListPos), ([0 for _ in range(len(pathListNeg))] + [1 for _ in range(len(pathListPos))])
    
    def prepareOnlyPos(self, df: pd.DataFrame):
        pathListPos = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].notna())]["ImagePath"].to_list()
        return pathListPos[:50], [1 for _ in range(len(pathListPos))][:50]
    
    """Splits for MLP: 0.1, 0.1, 0.8"""
    def prepareTrainMLP(self, df: pd.DataFrame):
        pathListNeg, pathListPos, balanceLen = self.balanceDatasets(df)

        pathListNeg = pathListNeg[:int(balanceLen*0.8)]
        pathListPos = pathListPos[:int(balanceLen*0.8)]
        return (pathListNeg + pathListPos), ([0 for _ in range(len(pathListNeg))] + [1 for _ in range(len(pathListPos))])

    def prepareTestMLP(self, df: pd.DataFrame):
        pathListNeg, pathListPos, balanceLen = self.balanceDatasets(df)

        lenTrain = int(balanceLen*0.8)
        pathListNeg = pathListNeg[lenTrain : lenTrain + int(balanceLen*0.1)]
        pathListPos = pathListPos[lenTrain : lenTrain + int(balanceLen*0.1)]
        return (pathListNeg + pathListPos), ([0 for _ in range(len(pathListNeg))] + [1 for _ in range(len(pathListPos))])

    def prepareValMLP(self, df: pd.DataFrame):
        pathListNeg, pathListPos, balanceLen = self.balanceDatasets(df)

        lenTrain = int(balanceLen*0.8)
        pathListNeg = pathListNeg[lenTrain + int(balanceLen*0.1) :]
        pathListPos = pathListPos[lenTrain + int(balanceLen*0.1) :]
        return (pathListNeg + pathListPos), ([0 for _ in range(len(pathListNeg))] + [1 for _ in range(len(pathListPos))])
    
    def balanceDatasets(self, df: pd.DataFrame):
        pathListNeg = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].isna())]["ImagePath"].to_list()
        # only pos we're sure of
        pathListPos = df[(df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].notna())]["ImagePath"].to_list()
        balanceLen = min(len(pathListNeg), len(pathListPos))
        return (pathListNeg[:balanceLen], pathListPos[:balanceLen], balanceLen)


class UnNormalize(object):
    def __init__(self, mean = (MEAN, MEAN, MEAN), std = (STD, STD, STD)):
        self.mean = mean
        self.std = std

    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized
            or (B, C, H, W) to be normalized with batch
        Returns:
            Tensor: Normalized image.
        """
        if tensor.dim() == 4:
            for t in tensor:
                self.unNormSingle(t)
            return tensor
        elif tensor.dim() == 3:
            return self.unNormSingle(tensor)
        else:
            raise ValueError("Tensor must be 4-dim or 3-dim")

    def unNormSingle(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# %%
"""
Try out for test/training set
"""
# df1 = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
# df2 = pd.read_csv("imageProps.csv", on_bad_lines="warn")
# #print(df1.head(10))
# #print(df2.head(10))
# df = pd.merge(df1, df2, left_on="fotocode", how="right", right_on="ImagePath")
# #df = df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].isna())]
# df["ImagePath"].unique().size
# #df.head(10)
# df[(df["motion"] == "False") & (df["badImage"] == "False") & (df["grayscale"] == "False") & ( df["species"].notna())]["ImagePath"].unique().size

# # %%
# for i, path in enumerate(df[(df["motion"] == "True") & (df["badImage"] == "False") & (df["grayscale"] == "False") & (df["species"].isna())]["ImagePath"]):
#     img = Image.open(f'/data/shared/herons/TinaDubach_data/{path[5:9]}/{path}.JPG')
#     plt.imshow(img)
#     plt.show()
#     if i > 50:
#         break

# %%
