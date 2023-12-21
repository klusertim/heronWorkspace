# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("/data/tim/heronWorkspace/src")

# %%
from AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelSummary
from torchsummary import summary
import HeronImageLoader
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lightning.pytorch as pl

# %%
# Training
cae = AEHeronModel(batch_size=32, num_workers_loader=1)
summary(cae, (3, 324, 216), device="cpu")

# %%
trainer = pl.Trainer(accelerator='cuda', max_epochs=1, devices=[1]) # devices is the index of the gpu
trainer.fit(cae)


# %%
# predict some images
model = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/lightning_logs/version_18/checkpoints/epoch=0-step=236.ckpt")
trainer = pl.Trainer()
trainer.predict(model)

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
