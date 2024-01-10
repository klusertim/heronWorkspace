# %%

# import sys
# sys.path.append("/data/tim/heronWorkspace/src")

#! export PYTHONPATH="/data/tim/heronWorkspace/src"
#%%
from models import CAEBigBottleneck

# %%
import lightning as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from HeronImageLoader import HeronDataset, HeronDatasetCAE, UnNormalize
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from models import CAEV1






class CAEHeron(pl.LightningModule):
    
    def __init__(self,
        learning_rate=0.008317637711026709,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        model=CAEV1(),
        cameras = ["GBU3"], 
        transforms=None):
        
        super(CAEHeron, self).__init__() # changed from super(AEHeronModel, self).__init__(), seems to be jupyter issue

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.cameras = cameras
        self.transforms = transforms
        
        self.model = model

        # dataset specific attributes
        self.imsize = model.imsize 
        self.dims = (3, self.imsize[0], self.imsize[1])

        self.save_hyperparameters(ignore=["model"])

    
    def forward(self, x):
        x = self.model.encoder(x)
        x = self.model.decoder(x)
        return x
    
    def configure_optimizers(self):
        "optimiser config plus lr scheduler callback"
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
       
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 100], gamma=0.1 
        )
        return [optimizer], [lr_scheduler]
       
    def training_step(self, batch, batch_idx):
        "training iteration per batch"

        x, y, _ = batch
        output = self(x)
        
        loss = self.loss_fn(output, x)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True) 
        return loss

    def validation_step(self, batch, batch_idx, print_log="val"):
        "validation iteration per batch"
        x, y, _ = batch

        output = self(x)
       
        loss = self.loss_fn(output, x)

        self.log(f"{print_log}_loss", loss, prog_bar=True, sync_dist=True)  
        return loss

    def test_step(self, batch, batch_idx, print_log="tst"):
        "test iteration per batch"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log="tst")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch

        preds = self(x)

        self.predictBatchVisual(x, preds)
        
        return preds  
    
    # DATALOADERS

    def train_dataloader(self):
        return DataLoader(HeronDatasetCAE(set="train", resize_to=self.imsize, cameras=self.cameras, transforms=self.transforms), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers_loader)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(HeronDatasetCAE(set="val", resize_to=self.imsize, cameras=self.cameras, transforms=self.transforms), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    def test_dataloader(self):
        return  DataLoader(HeronDatasetCAE(set="test", resize_to=self.imsize, cameras=self.cameras, transforms=self.transforms), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    def predict_dataloader(self):
        return  DataLoader(HeronDatasetCAE(set="test", resize_to=self.imsize, cameras=self.cameras, transforms=self.transforms), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    # HELPER FUNCTIONS

    def loss_fn(self, output, target):
        loss = F.mse_loss(output, target, reduction='none')
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def heatMap(self, before:torch.Tensor, after:torch.Tensor):
        # heatmap for batch of images
            heatMaps = []
            for batchIndex in range(len(before)):
                stepY = 5
                stepX = 5
                heatMap = []
                for i in range(0, before.shape[-2]-stepY+1, stepY):
                    row = []
                    for j in range(0, before.shape[-1]-stepX+1, stepX):
                        row.append(F.mse_loss(before[batchIndex, :, i:i+stepY, j:j+stepX], after[batchIndex, :, i:i+stepY, j:j+stepX]).item())
                    heatMap.append(row)
                heatMaps.append(heatMap)
            return heatMaps
    
    def msePerImage(self, before:torch.Tensor, after:torch.Tensor):
        lossArr = []
        for (x, y) in zip(before, after):
            lossArr.append(F.mse_loss(x, y).item())
        return lossArr
    
    def predictBatchVisual(self, before:torch.Tensor, after:torch.Tensor):
        unNorm = UnNormalize()
        fig, ax = plt.subplots(3, len(before), figsize=(len(before)*10, 3*10))

        lossArr = self.msePerImage(before, after)
        heatMaps = self.heatMap(before, after)
        for i in range(len(before)):
            ax[0, i].imshow(unNorm(before[i]).cpu().permute(1, 2, 0))
            ax[1, i].imshow(unNorm(after[i]).cpu().permute(1, 2, 0))
            ax[2, i].imshow(heatMaps[i], cmap='hot', interpolation='nearest')
            ax[2, i].text(0.5,-0.2, f'mse loss: {lossArr[i]:.4f}', size=20, ha="center", transform=ax[2, i].transAxes)
        
        plt.axis('off')
        plt.show()


        