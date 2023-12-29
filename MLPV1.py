import lightning as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from HeronImageLoader import HeronDataset, UnNormalize
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.classification import Accuracy
from argparse import ArgumentParser

class MLP(pl.LightningModule):
    
    def __init__(self, learning_rate=0.008317637711026709,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        mlpModel=None, 
        cae=None, resize_Y=0):
        super(MLP, self).__init__() # changed from super(AEHeronModel, self).__init__(), seems to be jupyter issue

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=["mlpModel", "cae", "model"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.resize_Y = resize_Y
        
        self.model = mlpModel
        self.cae = cae
        self.cae.eval()
        self.imsize = cae.imsize

        self.accuracy = Accuracy(task="binary")
        # dataset specific attributes
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--resize_Y', type=float, default=0, help="resize Y axis of image: 0 means no resize, 0.5 means half the size (always to the bottom)")
        return parser
    

    def forward(self, x):
        x = self.model.fw(x)
        return x
    
    def configure_optimizers(self):
        "optimiser config plus lr scheduler callback"
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
       
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30], gamma=0.1  # 10, 20,
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        "training iteration per batch"
        x, y, _ = batch
        output = self.cae(x)

        errorVals = self.computeErrorVals(x, output)

        pred = self(errorVals).squeeze()
        y = y.type_as(pred)
        self.accuracy(pred, y)
        loss = F.binary_cross_entropy(pred, y.type_as(pred)) #TODO: evtl change reduction, without logits because sigmoid already applied
        self.log("train_loss", loss, prog_bar=True, sync_dist=True) 
        self.log(f"train_acc", self.accuracy, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, print_log="val"):
        "validation iteration per batch"
        x, y, _ = batch
        output = self.cae(x) #TODO add accuracy metric

        errorVals = self.computeErrorVals(x, output)

        pred = self(errorVals).squeeze()
        y = y.type_as(pred)
        self.accuracy(pred, y)

        # print(f"pred: {pred}, y: {y}")
        loss = F.binary_cross_entropy(pred, y)

        self.log(f"{print_log}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{print_log}_acc", self.accuracy, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, print_log="tst"):
        "test iteration per batch"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log="tst")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch
        output = self.cae(x)

        errorVals = self.computeErrorVals(x, output)

        pred = self(errorVals).squeeze()

        self.predictBatchVisual(x, output, errorVals, pred, y)

        # return pred>0.5 
    

    
    # def prepare_data(self) -> None:
    #     self.train_dataset = HeronDataset(set="train", resize_to=self.imsize)
    #     self.val_dataset = HeronDataset(set="val", resize_to=self.imsize)
    #     self.test_dataset = HeronDataset(set="test", resize_to=self.imsize)
    
    def train_dataloader(self):
        return DataLoader(HeronDataset(set="trainMLP", resize_to=self.imsize), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers_loader)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(HeronDataset(set="valMLP", resize_to=self.imsize), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    def test_dataloader(self):
        return  DataLoader(HeronDataset(set="testMLP", resize_to=self.imsize), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    def predict_dataloader(self):
        return  DataLoader(HeronDataset(set="testMLP", resize_to=self.imsize), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers_loader)
    
    # HELPER FUNCTIONS
    def computeErrorVals(self, input: torch.Tensor, output: torch.Tensor):
        input = input[:, :, int(input.shape[2] * self.resize_Y) : , : ]
        output = output[:, :, int(output.shape[2] * self.resize_Y) : , : ]


        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(input.device)
        ssimArr = ssim(input, output)

        mse = F.mse_loss(input, output, reduction="none").mean(dim=(1, 2, 3))
        mae = F.l1_loss(input, output, reduction="none").mean(dim=(1, 2, 3))

        return torch.stack((mse, mae, ssimArr), dim=1).type_as(input)
    
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
    
    def predictBatchVisual(self, before:torch.Tensor, after:torch.Tensor, errorVals:torch.Tensor, pred:torch.Tensor, true:torch.Tensor):
        unNorm = UnNormalize()
        fig, ax = plt.subplots(3, len(before), figsize=(len(before)*10, 3*10))
        fig.tight_layout()

        errorVals = errorVals.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        heatMaps = self.heatMap(before, after)
        for i in range(len(before)):
            ax[0, i].imshow(unNorm(before[i]).cpu().permute(1, 2, 0))
            ax[1, i].imshow(unNorm(after[i]).cpu().permute(1, 2, 0))
            ax[2, i].imshow(heatMaps[i], cmap='hot', interpolation='nearest')
            ax[2, i].text(0.5,-0.5, f'mse: {errorVals[i, 0]:.4f}\nmae: {errorVals[i, 1]:.4f}\nssim: {errorVals[i, 2]:.4f}\nprediction: {pred[i]:.4f}\ntrue: {true[i]}', size=20, ha="center", transform=ax[2, i].transAxes)
        
        for a0 in ax:
            for a1 in a0:
                a1.set_xticks([])
                a1.set_yticks([])

        plt.axis('off')
        plt.show()
