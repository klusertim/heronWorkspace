import sys
sys.path.append("/data/tim/heronWorkspace/src")

from heronWorkspace.AE.AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelCheckpoint
from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
from models import CAESmallBottleneck, CAEBigBottleneck, MLPBasic


caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAESmallBottleneck/version_0/checkpoints/epoch=146-step=34692.ckpt", imsize=(216, 324), model = CAESmallBottleneck(), batch_size = 16, learning_rate=0.017378008287493765)
trainer = pl.Trainer(devices=1, num_nodes=1)
trainer.test(caeLoaded)