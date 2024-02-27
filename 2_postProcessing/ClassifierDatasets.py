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
sys.path.append("/data/tim/heronWorkspace/src")
sys.path.append("/data/tim/heronWorkspace/0_preProcessing")
sys.path.append("/data/tim/heronWorkspace/1_AE")
sys.path.append("/data/tim/heronWorkspace/2_postProcessing")
from classifyMotionGray import ClassifyMotionGray
import random
from sklearn.model_selection import train_test_split

MEAN = 0.5
STD = 0.5

class MLPDatasetValidated(Dataset): # from validation of SBU4
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    imsize = (2448-100, 3264) # h x w
    def __init__(self, set="train", resize_to = (216, 324), transform=None):
        """
        validatoinMode: TinaDubach, MotionSensor, Manually
        """
            
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
    

class DatasetThreeConsecutive(Dataset):
    ROOT_DIR = '/data/shared/herons/TinaDubach_data'

    def __init__(self,
                set="train",
                cameras = ["GBU3"],
                resize_to = (216, 324),
                lblValidationMode = "Manual",
                balanced = True,
                anomalyObviousness = "obvious",
                distinctCAETraining = False,
                colorMode = "RGB",
                random_state = 1,
                transforms=None):
        """
        set: train, val, test, all
        lblValidatoinMode: TinaDubach, MotionSensor, Manual
        cameras: list of cameras to use
        resize_to: (h, w) to resize the images to
        balanced: if true, the dataset will be balanced between positive and negative frames - always balanced on camera basis
        anomalyObviousness: obvious, notObvious, all (does only come into play if lblValidationMode is Manual)
        random_state: random seed for reproducibility
        transform: torchvision transform to apply to the images - None is default transform
        distinctCAETraining: if true, the dataset will be distinct from the one used for training the CAE
        colorMode: RGB, grayscale, mix
        """           
        self.set = set
        self.imsize = resize_to
        self.lblValidationMode = lblValidationMode
        self.balanced = balanced
        self.anomalyObviousness = anomalyObviousness
        self.transforms = transforms
        self.distinctCAETraining = distinctCAETraining
        self.colorMode = colorMode
        self.random_state = random_state

        allFeatures = []
        for camera in cameras:
            allFeatures.extend(self.generateFeatures(camera))

        if set == "all":
            self.imagePaths = allFeatures
            return
        
        trainSet, testSet = train_test_split(allFeatures, test_size=0.2, random_state=random_state)

        trainSet, valSet = train_test_split(trainSet, test_size=0.25, random_state=random_state)

        if set == "test":
            self.imagePaths = testSet
        elif set == "val":
            self.imagePaths = valSet
        elif set == "train":
            self.imagePaths = trainSet
        elif set == "paper":
            self.imagePaths = self.generatePaperSet()
        else:
            raise ValueError(f"set: {set} not implemented")


    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        
        imgs, lbl = self.imagePaths[idx]
        camera = imgs[1].split("_")[1]
        try:
            prevImg, img, nextImg = [self.loadAndTransform(x) for x in imgs]     
            return [prevImg, img, nextImg], lbl, camera, imgs[1]
        except OSError:
            print(f"Error occured loading the image: {img}")
            zero = torch.zeros((3, self.imsize[0], self.imsize[1]))
            return (
                [zero]*3,
                0,
                camera,
                imgs[1]
            )
    
    def transform(self, img):
        trsf = T.Compose([
               T.ToTensor(),
                T.Resize([self.imsize[0], self.imsize[1]], antialias=True),
                lambda im : F.crop(im, top=0, left=0, height=self.imsize[0]-20, width=self.imsize[1]),
                T.Normalize(mean=(MEAN, MEAN, MEAN), std=(STD, STD, STD)),
            ]
        )
        return trsf(img)
    
    ### HELPER FUNCTIONS ###
    def generatePaperSet(self):
        # 2-3 interesting images
        # intImg1 = ["2017_SBU3_04030785", "2017_SBU3_04030786", "2017_SBU3_04030787"]
        # intImg2 = ["2017_SBU3_04080237", "2017_SBU3_04080238", "2017_SBU3_04080239"]
        # intImg3 = ["2017_SBU3_04140814", "2017_SBU3_04140815", "2017_SBU3_04140816"]
        # intImg4 = ["2017_SBU3_04160054", "2017_SBU3_04160055", "2017_SBU3_04160056"]
        # intImg5 = ["2017_SBU3_04280215", "2017_SBU3_04280216", "2017_SBU3_04280217"]
        # intImg6 = ["2017_SBU3_07050557", "2017_SBU3_07050558", "2017_SBU3_07050559"]
        # intImg7 = ["2017_SBU3_05270186", "2017_SBU3_05270187", "2017_SBU3_05270188"]

        # intImg1 = ["2017_SBU3_04030785", "2017_SBU3_04030786", "2017_SBU3_04030787"]
        # intImg2 = ["2017_SBU3_04080237", "2017_SBU3_04080238", "2017_SBU3_04080239"]
        # imArr = ['2017_NEN1_03200419',
        # '2017_NEN1_04050024',
        # '2017_NEN1_03220623',
        # '2017_NEN1_03200432',
        # '2017_NEN1_07170789',
        # '2017_NEN1_03280207',
        # '2017_NEN1_01310504',
        # '2017_NEN1_07140519',
        # '2017_NEN1_02260016',
        # '2017_NEN1_02110554']

        imArr = [
            '2017_NEN1_03200432',
            '2017_NEN1_02110554',
            "2017_SBU3_04080238",
            "2017_SBU3_07050558"
        ]

        consImArr = []

        for path in imArr:
            year, camera, nr = path.split("_")
            consImArr.append(([f"{year}_{camera}_{str(int(nr)-1).zfill(8)}", path, f"{year}_{camera}_{str(int(nr)+1).zfill(8)}"], 1))

        return consImArr
        

        # return [(intImg1, 1), (intImg2, 0), (intImg3, 0), (intImg4, 0), (intImg5, 0), (intImg6, 0), (intImg7, 1)]
        

    def loadAndTransform(self, path):
        with open(f'/data/shared/herons/TinaDubach_data/{path[5:9]}/{path}.JPG', "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            else:
                img = self.transform(img)
        return img

    def generateFeatures(self, camera : str):
        try:
            dfMotionGrayClassified = pd.read_csv(f"/data/tim/heronWorkspace/MotionGrayClassification/classifiedMotionGray{camera}.csv")
            dfPreClassified = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
        except:
            return []
        
        dfMotionGrayClassified = dfMotionGrayClassified[(~dfMotionGrayClassified["badImage"])]

        # distinct for training
        if self.distinctCAETraining:
            dfMotionGrayClassified = dfMotionGrayClassified[dfMotionGrayClassified["motion"]]
        
        #color mode
        if self.colorMode == "grayscale":
            dfMotionGrayClassified = dfMotionGrayClassified[dfMotionGrayClassified["grayscale"]]
        elif self.colorMode == "RGB":
            dfMotionGrayClassified = dfMotionGrayClassified[~dfMotionGrayClassified["grayscale"]]
        elif self.colorMode == "mix":
            pass
        else:
            raise ValueError(f"colorMode: {self.colorMode} not implemented")
        
        # merge dataframes and drop duplicates
        # IMPORTANT: sort the data here already

        dfFeatures = pd.merge(dfMotionGrayClassified, dfPreClassified, left_on="ImagePath", right_on="fotocode", how="left")
        dfFeatures = dfFeatures.drop_duplicates(subset = ["ImagePath"], keep="first")

        try:
            manuallyClassified = pd.read_csv(f"/data/tim/heronWorkspace/manuallyClassified/manuallyClassified{camera}.csv")
            dfFeatures = pd.merge(dfFeatures, manuallyClassified, on="ImagePath", how="left")
        except:
            # if not manual mode, then we dont care
            if self.lblValidationMode == "Manual":
                raise FileNotFoundError(f"/data/tim/heronWorkspace/manuallyClassified/manuallyClassified{camera}.csv not found\nyou must manually classify some images first to use the manual validation mode")
            
        dfFeatures = dfFeatures.sort_values(by=["ImagePath"])
        dfFeatures = dfFeatures.reset_index()

        # label generation depending on validation mode
        labels = []
        if self.lblValidationMode == "TinaDubach":
            indexSet = dfFeatures.index # all Images
            labels = dfFeatures["species"].notna().astype(int).tolist()

        elif self.lblValidationMode == "MotionSensor":
            indexSet = dfFeatures.index # all Images
            labels = dfFeatures["motion"].astype(int).tolist()

        elif self.lblValidationMode == "Manual":
            #load manually generated labels

            # label generation depending on obviousness
            if self.anomalyObviousness == "obvious":
                indexSet = dfFeatures[(dfFeatures["ValidationValue"] == 0) | (dfFeatures["ValidationValue"] == 2) | (dfFeatures["ValidationValue"] == 4)].index
            elif self.anomalyObviousness == "notObvious":
                indexSet = dfFeatures[(dfFeatures["ValidationValue"] == 0) | (dfFeatures["ValidationValue"] == 1) | (dfFeatures["ValidationValue"] == 3)].index
            elif self.anomalyObviousness == "all":
                indexSet = dfFeatures[dfFeatures["ValidationValue"] >= 0].index
            else:
                raise ValueError(f"anomalyObviousness: {self.anomalyObviousness} not implemented")

            labels = dfFeatures.loc[indexSet]["ValidationValue"].astype(int).tolist()
            labels = [1 if x > 0 else x for x in labels]
        else:
            raise ValueError(f'Label val mode: {self.lblValidationMode} not implemented')
        
        features = []
        dfPaths = dfFeatures["ImagePath"]
        for i, index in enumerate(indexSet):
            _, _, nrCurr = dfPaths.loc[index].split("_")
            if index > 0 and index < len(dfPaths)-1:
                _, _, nrPrev = dfPaths.loc[index-1].split("_")
                _, _, nrNext = dfPaths.loc[index+1].split("_")
                if int(nrPrev) + 1 == int(nrCurr) and int(nrNext)-1 == int(nrCurr):
                    if labels[i] != -1:
                        features.append([(dfPaths.loc[index - 1], dfPaths.loc[index], dfPaths.loc[index + 1]), labels[i]])

        # # generate features - always theree consecutive images
        # sortedPaths = dfFeatures["ImagePath"].tolist()
        # features = []
        # for i, row in enumerate(sortedPaths):
        #     _, _, nrCurr = row.split("_")
        #     if i > 0 and i < len(sortedPaths)-1:
        #         _, _, nrPrev = sortedPaths[i-1].split("_")
        #         _, _, nrNext = sortedPaths[i+1].split("_")
        #         if int(nrPrev) + 1 == int(nrCurr) and int(nrNext)-1 == int(nrCurr):
        #             if labels[i] != -1:
        #                 features.append([(sortedPaths[i-1], row, sortedPaths[i+1]), labels[i]])

        # balance dataset
        if self.balanced:
            featuresNormal = [feature for feature in features if feature[1] == 0]
            featuresMotion = [feature for feature in features if feature[1] == 1]
            minLen = min(len(featuresNormal), len(featuresMotion))

            random.seed(self.random_state)
            featuresNormal = random.sample(featuresNormal, minLen)
            featuresMotion = random.sample(featuresMotion, minLen)

            features = featuresNormal + featuresMotion
        return features
    


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

