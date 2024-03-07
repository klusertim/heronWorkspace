import sys
sys.path.append("/data/tim/heronWorkspace/src")
sys.path.append("/data/tim/heronWorkspace/0_preProcessing")
sys.path.append("/data/tim/heronWorkspace/1_AE")
sys.path.append("/data/tim/heronWorkspace/2_postProcessing")
sys.path.append("/data/tim/heronWorkspace/")

import torch
from ClassifierDatasets import DatasetThreeConsecutive, UnNormalize
from torch.utils.data import DataLoader
import numpy as np
from models import CAEV1
from AEHeronModel import CAEHeron
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F


class CheckPoints:
    bestSBU3 = "/data/tim/heronWorkspace/logs/BasicCAE1SBU3/version_0/checkpoints/epoch=48-step=9457.ckpt"
    worseSBU3 = "/data/tim/heronWorkspace/logs/BasicCAE1/version_12/checkpoints/epoch=24-step=4825.ckpt"
    bestGlobal = "/data/tim/heronWorkspace/logs/BasicCAE1Global/version_0/checkpoints/epoch=48-step=66591.ckpt"
    worseGlobal = "/data/tim/heronWorkspace/logs/BasicCAE1/version_10/checkpoints/epoch=13-step=19026-v1.ckpt"


class MinFilter:
    def __init__(self, kernelSize: int):
        self.kernelSize = kernelSize
        
    def __call__(self, tensor: torch.Tensor ) -> torch.Tensor:
        # Unfold the tensor into sliding local blocks
        unfolded = tensor.unfold(0, self.kernelSize, 1)
        unfolded = unfolded.unfold(1, self.kernelSize, 1)
        # Compute the minimum in each of these blocks
        return unfolded.min(dim=-1)[0].min(dim=-1)[0]
    
class PostProcess: 
    @staticmethod
    def computeSum(params: dict, loaderParams: dict, checkPoint = None):
        unnorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if checkPoint is None:
            checkPoint = CheckPoints.bestGlobal
        caeLoaded = CAEHeron.load_from_checkpoint(checkPoint, model = CAEV1)
        caeLoaded.freeze()
    
        
        dataset = DatasetThreeConsecutive(cameras=params["cameras"], resize_to=CAEV1.imsize, **loaderParams)
        print(f'Length of dataset: {len(dataset)}')
        print(params)
        dataLoader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

        blur = GaussianBlur(kernel_size=params["gaussianFilterSize"], sigma=params["gaussianFilterSigma"]) #TODO: make this a parameter
        lossFn = F.mse_loss if params["lossFn"] == "MSE" else F.l1_loss
        sumVals = []
        lblVals = []

        for (imArr, lblArr, camera, ImagePath) in dataLoader:

            isTrainingCamera = camera in caeLoaded.hparams.cameras
            prevImg = imArr[0] #alwasy #batch_size images
            currImg = imArr[1]
            nextImg = imArr[2]

            prevPred, currPred, nextPred = [unnorm(caeLoaded(x.to(caeLoaded.device))) for x in [prevImg, currImg, nextImg]]
            prevImg, currImg, nextImg = [unnorm(x) for x in [prevImg, currImg, nextImg]]


            prevImgBlurred, currImgBlurred, nextImgBlurred = [blur.forward(x).to(prevPred.device) for x in [prevImg, currImg, nextImg]]
        
            
            prevImd, currImd, nextImd = [torch.sum(lossFn(imgBlurred, pred, reduction='none'), dim=1) for imgBlurred, pred in zip([prevImgBlurred, currImgBlurred, nextImgBlurred], [prevPred, currPred, nextPred])]

            prevToCurrImd = torch.clamp(torch.sub(currImd, prevImd), min= 0)
            nextToCurrImd = torch.clamp(torch.sub(currImd, nextImd), min= 0)

            prevNextCurrImd = torch.div(torch.add(prevToCurrImd, nextToCurrImd), 2)

            minFilter = MinFilter(kernelSize=params["minFilterKernelSize"])
            prevNextCurrImdMin = torch.stack([minFilter(x) for x in prevNextCurrImd]) #TODO: evtl make this as before
            

            prevNextCurrImdMinThresh = torch.where(prevNextCurrImdMin < params["zeroThreshold"], torch.zeros_like(prevNextCurrImdMin), prevNextCurrImdMin)
            # sumZeroThreshold = torch.sum(torch.where(prevNextCurrImdMin < zeroThreshold, torch.zeros_like(prevNextCurrImdMin), prevNextCurrImdMin), dim=(1, 2)).numpy(force=True)
            # sumZeroThresholdArr = np.array(sumZeroThresholdArr) # shape: (len(zeroThresholdArr), batch_size)

            
            sumPrevNextCurrImdMin = torch.sum(prevNextCurrImdMinThresh, dim=(1, 2))
            # predictions = (sumPrevNextCurrImdMin> params["sumThreshold"]).to(torch.int)

            sumVals = np.concatenate((sumVals, sumPrevNextCurrImdMin.to("cpu").detach().numpy()))
            lblVals = np.concatenate((lblVals, lblArr.to("cpu").detach().numpy()))
        
            # imagePaths += list(ImagePath)
        
        
        return np.array(sumVals), np.array(lblVals)



