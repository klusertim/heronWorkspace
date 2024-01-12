import sys
sys.path.append("/data/tim/heronWorkspace/src")

from heronWorkspace.AE.AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
from heronWorkspace.classifier.MLPV1 import MLP
from models import MLPBasic, CAEBigBottleneck, CAESmallBottleneckWithLinear
from argparse import ArgumentParser

# load cae
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAESmallBottleneckLinearLayer/version_0/checkpoints/epoch=23-step=5664.ckpt", model=CAESmallBottleneckWithLinear(), imsize=(216, 324))
caeLoaded.freeze()

# arguments
parser = ArgumentParser()
# parser.add_argument('--resize_Y', type=float, default=0)
parser = MLP.add_model_specific_args(parser)
args = parser.parse_args()

print(args)

mlp = MLP(mlpModel=MLPBasic(), cae=caeLoaded, batch_size=16, num_workers_loader=4, resize_Y=args.resize_Y)

# summary(mlp, [3], device="cuda")

# Find learning rate
trainer = pl.Trainer( accelerator='cuda', max_epochs=25) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(mlp)
fig = lr_finder.plot(show=True, suggest=True, )
fig.savefig('lr_finder.jpg')

# Train

mlp = MLP(learning_rate=lr_finder.suggestion(), mlpModel=MLPBasic(), cae=caeLoaded, batch_size=16, num_workers_loader=4, resize_Y=args.resize_Y)

logger=CSVLogger(save_dir="logs/", name="basicMLPV1Resized")
# earlyStopping = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=5, verbose=False, mode="min")
callbacks = [ModelCheckpoint(monitor="val_acc", save_top_k=4, mode="min"), ModelCheckpoint(monitor="val_acc", every_n_epochs=20, mode="min")]
trainer = pl.Trainer(callbacks=callbacks, logger=logger, accelerator='cuda', max_epochs=25, log_every_n_steps=1) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
trainer.fit(mlp)

# Plot
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train_acc", "val_acc"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="Accuracy"
)
plt.savefig(f"{trainer.logger.log_dir}/accPlotTraining.jpg")

trainer.test(mlp)