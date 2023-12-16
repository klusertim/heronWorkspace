# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("/data/tim/heronWorkspace/src")

# %%
import lightning.pytorch as pl
from AEHeronModelV1 import AEHeronModel
from lightning.pytorch.callbacks import ModelSummary
from torchsummary import summary

# %%
cae = AEHeronModel()
summary(cae, (3, 324, 216), device="cpu")

# %%
trainer = pl.Trainer(callbacks=[ModelSummary(max_depth=1)], accelerator='cuda', max_epochs=1) 
trainer.fit(cae)
# %%
11 * 3 - 2 * 1 + 1 * (5-1) +  1
35 * 3 - 2 * 1 + 1 * (5-1) +  1
107 * 3 - 2 * 1 + 1 * (5-1) +  1

8 * 3 - 2 * 2 + 1 * (4-1) + 1
23 * 3 - 2 * 1 + 1 * (5-1) +  1
71 * 3 - 2 * 1 + 1 * (5-1) +  1
# %%

model = AEHeronModel.load_from_checkpoint("/data/tim/heronWorkspace/lightning_logs/version_14/checkpoints/epoch=0-step=236.ckpt")
trainer = pl.Trainer()
trainer.predict(model)
# %%
