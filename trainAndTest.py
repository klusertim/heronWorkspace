# %%
import pytorch_lightning as pl
from AEHeronModelV1 import AEHeronModel

# %%
cae = AEHeronModel()
trainer = pl.Trainer(gpus=1, accelerator='dp', max_epochs=5) 
trainer.fit(cae)
# %%
