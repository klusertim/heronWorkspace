# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("/data/tim/heronWorkspace/src")

from AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelSummary
from torchsummary import summary
import HeronImageLoader
from torch.utils.data import DataLoader, BatchSampler
from matplotlib import pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
from MLPV1 import MLP
from models import MLPBasic, CAEBigBottleneck
import numpy as np
import torch.nn.functional as F
import torch




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

metrics = pd.read_csv(f"/data/tim/heronWorkspace/logs/basicMLPV1/version_0/metrics.csv")

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

plt.savefig("loss_over_epochs.jpg")

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
# basic model with 150 epochs and big bottleneck

caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAEBigBottleneck/version_0/checkpoints/epoch=149-step=35400.ckpt")
dataLoader = DataLoader(HeronImageLoader.HeronDataset(set="onlyPos", resize_to=(215, 323)), batch_size=16, shuffle=False, num_workers=4)
trainer = pl.Trainer()
res = trainer.predict(caeLoaded, dataloaders=dataLoader)


# %%
# basic model with 10 epochs and big bottleneck
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
dataLoader = DataLoader(HeronImageLoader.HeronDataset(set="onlyPos", resize_to=(215, 323)), batch_size=16, shuffle=False, num_workers=4)
trainer = pl.Trainer()
res = trainer.predict(caeLoaded, dataloaders=dataLoader)


# %%
dataset = HeronImageLoader.HeronDataset(set="testMLP", resize_to=(215, 323))
print(len(dataset))

# %%
# train mlp
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
caeLoaded.freeze()
mlp = MLP(mlpModel=MLPBasic(), cae=caeLoaded, batch_size=16, num_workers_loader=4)
trainer = pl.Trainer(max_epochs=1, accelerator='cuda', log_every_n_steps=1)
trainer.fit(mlp)


# %%
# test mlp
trainer = pl.Trainer(max_epochs=1, accelerator='cuda', log_every_n_steps=1)
caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
caeLoaded.freeze()
mlpLoaded = MLP.load_from_checkpoint("/data/tim/heronWorkspace/lightning_logs/version_58/checkpoints/epoch=0-step=155.ckpt", cae=caeLoaded, mlpModel=MLPBasic())
trainer.predict(mlpLoaded)

# %%
df1 = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
df2 = pd.read_csv("/data/tim/heronWorkspace/ImageData/imagePropsSBU4.csv", on_bad_lines="warn")
df = pd.merge(df1, df2, left_on="fotocode", how="right", right_on="ImagePath")
df.sort_values(by=["ImagePath"], inplace=True)
df.head(10)

# %%
# test distance from last prediction to current

def heatMap(before: torch.Tensor, after: torch.Tensor, stepY, stepX):
    heatMap = []
    for i in range(0, before.shape[-2]-stepY+1, stepY):
        row = []
        for j in range(0, before.shape[-1]-stepX+1, stepX):
            row.append(F.mse_loss(before[:, i:i+stepY, j:j+stepX], after[:, i:i+stepY, j:j+stepX]).item())
        heatMap.append(row)
    return torch.tensor(heatMap).type_as(before)

caeLoaded = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/logs/basicCAE/version_0/checkpoints/epoch=9-step=630.ckpt")
caeLoaded.freeze()
dataLoader = DataLoader(HeronImageLoader.HeronDataset(set="test", resize_to=(215, 323), sorted=True), batch_size=1, shuffle=False, num_workers=4)
print(len(dataLoader.dataset.imagePaths))
unnorm = HeronImageLoader.UnNormalize()

stepY = 5
stepX = 5

lastImd = np.zeros((int(215/stepY), int(323/stepX)))
for i, img in enumerate(list(dataLoader)[200:]):
    # print(img[0].size())
    # plt.imshow(unnorm(img[0][0]).permute(1, 2, 0))
    img = img[0].to(caeLoaded.device)
    pred = caeLoaded(img)
    # img = unnorm(img[0][0]).permute(1, 2, 0).numpy()
    # pred = unnorm(pred[0].cpu()).permute(1, 2, 0).numpy()
    
    img, pred = [unnorm(x) for x in [img[0], pred[0]]]
    imd = heatMap(img, pred, stepY, stepX)

    img, pred = [x.permute(1, 2, 0).cpu().numpy() for x in [img, pred]]
    imd = imd.cpu().numpy()
    # imd = 0.0 + np.sum(img - pred, axis=2)**2
    # imd = np.linalg.norm(im - x, axis=2)

    # imd = imd / (np.max(imd) - np.min(imd))
    # imd = (imd - np.min(imd)) / (np.max(imd) - np.min(imd))

    f, a = plt.subplots(1,5, figsize=(50,10))
    # f.suptitle(fi)

   
    a[0].imshow(img)
    a[1].imshow(pred)
    ma = a[2].imshow(np.abs(imd), cmap="hot", interpolation='none')
    a[3].imshow(np.abs(imd - lastImd), cmap="hot", interpolation='none')

    diff = imd - lastImd
    a[4].imshow(np.where(diff < 0, 0, diff), cmap="hot", interpolation='none')

    plt.show()

    lastImd = imd
    
    if (i > 100):
        break

# %%
