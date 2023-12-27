# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("/data/tim/heronWorkspace/src")

from AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelSummary
from torchsummary import summary
import HeronImageLoader
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger



# %%
# Model
cae = AEHeronModel(batch_size=16, num_workers_loader=4)
summary(cae, (3, 215, 323), device="cpu")

# %%
# Find learning rate
trainer = pl.Trainer( accelerator='cuda', max_epochs=5, devices=[1]) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(cae)
lr_finder.plot(show=True, suggest=True)
print(lr_finder.suggestion())
# %%
# Train
trainer = pl.Trainer( accelerator='cuda', max_epochs=1, logger=CSVLogger(save_dir="logs/", name="my-model"), log_every_n_steps=1) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
trainer.fit(cae)

# %%
### Plot

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

plt.savefig("suggest_loss.pdf")

# df_metrics[["train_acc", "val_acc"]].plot(
#     grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
# )

# plt.savefig("suggest_acc.pdf")

plt.show()

# %%
# predict some images
# cae = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/lightning_logs/version_18/checkpoints/epoch=0-step=236.ckpt")
# trainer = pl.Trainer()
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/lightning_logs/version_32/checkpoints/epoch=0-step=2.ckpt")
trainer.predict(caeLoaded)

# %%
# play around with different loader settings
data = HeronImageLoader.HeronDataset()
loader = DataLoader(data, batch_size=2, num_workers=1, shuffle=False) # batch_size=64, num_workers=3
unnorm = HeronImageLoader.UnNormalize()
for i, (imArr, _, _) in enumerate(loader):
    # print(imArr[0])
    for j in range(len(imArr)):
        print("Normalized:")
        print(f'Mean: ' + str(imArr[j].mean(axis=(1, 2))))
        plt.imshow(imArr[j].permute(1, 2, 0))
        plt.show()
        unnormArr = unnorm(imArr)

        print("UnNormalized:")
        plt.imshow(unnormArr[j].permute(1, 2, 0))
        plt.show()
    if i >0 :
        break

# %%
metrics = pd.read_csv("/data/tim/heronWorkspace/logs/basicCAE/version_0/metrics.csv")

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

# %%
# basic model with 10 epochs and big bottleneck
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
trainer = pl.Trainer()
trainer.predict(caeLoaded)


# %%
# basic model with 10 epochs and big bottleneck

caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAEBigBottleneck/version_0/checkpoints/epoch=149-step=35400.ckpt")
dataLoader = DataLoader(HeronImageLoader.HeronDataset(set="onlyPos", resize_to=(215, 323)), batch_size=16, shuffle=False, num_workers=4)
trainer = pl.Trainer()
res = trainer.predict(caeLoaded, dataloaders=dataLoader)


# %%
# basic model with 150 epochs and big bottleneck
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
dataLoader = DataLoader(HeronImageLoader.HeronDataset(set="onlyPos", resize_to=(215, 323)), batch_size=16, shuffle=False, num_workers=4)
trainer = pl.Trainer()
res = trainer.predict(caeLoaded, dataloaders=dataLoader)
