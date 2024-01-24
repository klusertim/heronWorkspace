from ClassifierDatasets import DatasetThreeConsecutive
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform, uniform
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

random_state = 1
distributions = dict(
    cameras = [["NEN1", "SBU3"]],
    balanced = [True, False],
    distinctCAETraining = [True, False],
    filter = "MinFilter", #["MinFilter", "GaussianFilter"]
    zeroThreshold = uniform(0.15, 0.25), # loguniform(0.15, 0.4), #threshold for zeroing out the image
    sumThreshold = uniform(0.5, 20)
)

sampler = ParameterSampler(distributions, n_iter=30, random_state=random_state),

loaderParams = dict(
    lblValidationMode = "Manual",
    balanced = True,
    anomalyObviousness = "obvious",
    distinctCAETraining = False,
    colorMode = "RGB",
    random_state = 1,
    filterParams = [3],
    set = "all"
)
caePath = '/data/tim/heronWorkspace/logs/CAEV1/version_3/checkpoints/epoch=49-step=19350.ckpt',


for params in sampler:
    print(params["balanced"])
    dataset = DatasetThreeConsecutive(cameras=params["cameras"], **loaderParams)
    # loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
