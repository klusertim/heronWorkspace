import sys
sys.path.append("/data/tim/heronWorkspace/src")
sys.path.append("/data/tim/heronWorkspace/0_preProcessing")
sys.path.append("/data/tim/heronWorkspace/1_AE")
sys.path.append("/data/tim/heronWorkspace/2_postProcessing")

from AEHeronModel import CAEHeron
from lightning.pytorch.callbacks import ModelCheckpoint
from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
from models import CAEV1
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform, uniform
from datetime import datetime

distributions = dict(
    learning_rate = loguniform(0.001, 0.1),
    weight_decay = loguniform(1e-8, 1e-4),
    batch_size = [8, 16], # 32 probably too much
    cameras = [["GBU1", "GBU4", "KBU2", "PSU1", "PSU2", "PSU3", "SBU3", "SGN1"]],#[["NEN1", "SBU3", "SBU2", "SBU4", "NEN3"]], #["NEN1", "SBU3", "SBU2"], [["SBU3"]],
    bottleneck = [32, 64, 128, 256],
    ldim = [4, 8, 16],
    gammaScheduler = uniform(0.1, 0.9)
)


sampler = ParameterSampler(distributions, n_iter=10, random_state=1)

params = list(sampler)[3]

# print(params)

now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
print("Begin training at: ", now)

cae = CAEHeron(learning_rate=params["learning_rate"],
                   weight_decay=params["weight_decay"],
                   batch_size=params["batch_size"],
                   cameras=params["cameras"],
                   bottleneck=params["bottleneck"],
                   ldim = params["ldim"],
                   gammaScheduler=params["gammaScheduler"],
                   model=CAEV1)

logger=CSVLogger(save_dir="logs/", name="BasicCAE1Global")
callbacks = [ModelCheckpoint(monitor="val_loss", save_top_k=2, mode="min"), ModelCheckpoint(every_n_epochs=7)]
trainer = pl.Trainer(callbacks=callbacks, logger=logger, accelerator='cuda', max_epochs=50, log_every_n_steps=2) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
trainer.fit(cae)

now = datetime.now()
print("End training at: ", now)

try: 
    # Plot
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    plt.savefig(f"{trainer.logger.log_dir}/loss_plot.jpg")
except:
    print("Could not plot loss")

# trainer = pl.Trainer(callbacks=[], logger=None, accelerator='cuda', max_epochs=1, devices=1, num_nodes=1)
# trainer.test(cae)