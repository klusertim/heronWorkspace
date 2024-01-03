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
from models import CAESmallBottleneck, CAESmallBottleneckWithLinear


# cae = AEHeronModel(batch_size=16, model=CAESmallBottleneck(), imsize=(216, 324), num_workers_loader=4)
cae = AEHeronModel(batch_size=16, num_workers_loader=4, model=CAESmallBottleneckWithLinear(), imsize=(216, 324))

summary(cae, (3, 216, 324), device="cpu")

# Find learning rate
trainer = pl.Trainer( accelerator='cuda', max_epochs=150) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(cae)
fig = lr_finder.plot(show=True, suggest=True, )
fig.savefig('lr_finder.jpg')
print(lr_finder.suggestion())

# Train
# cae = AEHeronModel(batch_size=16, learning_rate=lr_finder.suggestion(), model=CAESmallBottleneck(), imsize=(216, 324), num_workers_loader=4)
cae = AEHeronModel(batch_size=16, learning_rate=lr_finder.suggestion(), num_workers_loader=4, model=CAESmallBottleneckWithLinear(), imsize=(216, 324))

# logger=CSVLogger(save_dir="logs/", name="basicCAESmallBottleneck")
logger=CSVLogger(save_dir="logs/", name="basicCAESmallBottleneckLinearLayer")
callbacks = [ModelCheckpoint(monitor="val_loss", save_top_k=4, mode="min"), ModelCheckpoint(every_n_epochs=40, mode="min")]
trainer = pl.Trainer(callbacks=callbacks, logger=logger, accelerator='cuda', max_epochs=25, log_every_n_steps=1) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
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