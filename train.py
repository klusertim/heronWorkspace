import sys
sys.path.append("/data/tim/heronWorkspace/src")

from AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelCheckpoint
from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt

cae = AEHeronModel(batch_size=16)
summary(cae, (3, 215, 323), device="cpu")

logger=CSVLogger(save_dir="logs/", name="basicCAEBigBottleneck")
callbacks = [ModelCheckpoint(monitor="val_loss", save_top_k=4, mode="min")]
trainer = pl.Trainer(callbacks=callbacks, logger=logger, accelerator='cuda', max_epochs=130, log_every_n_steps=3) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
trainer.fit(cae)

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
plt.savefig("suggest_loss.jpg")