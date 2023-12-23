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

from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


# %%
# Training
cae = AEHeronModel(batch_size=32, num_workers_loader=1)
summary(cae, (3, 215, 323), device="cpu")

# %%
trainer = pl.Trainer( accelerator='cuda', max_epochs=1) # devices is the index of the gpu, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))],
trainer.fit(cae)


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
