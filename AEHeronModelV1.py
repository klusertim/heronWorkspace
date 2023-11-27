# %%
from models import CAE

# %%
import lightning as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from HeronImageLoader import HeronDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


class AEHeronModel(pl.LightningModule):
    
    def __init__(self, learning_rate=1e-4,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4):
        super(AEHeronModel, super).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        
        self.model = CAE()

        # dataset specific attributes
        self.imsize = (324, 316)
        self.dims = (3, self.imsize[0], self.imsize[1])
    
    def forward(self, x):
        x = self.model.encoder(x)
        print(f"enc {x.shape}")
        # x = self.model.bottleneck(x)
        x = self.model.decoder(x)
        print(f"bot {x.shape}")

        return x
    
    def configure_optimizers(self):
        "optimiser config plus lr scheduler callback"
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
       
       # why do we need a scheduler?
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30], gamma=0.1  # 10, 20,
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        "training iteration per batch"

        x, y, _ = batch
        # output, mu, log_var = self(x)
        output = self(x)
        
        loss = F.mse_loss(output, x)

        self.log("trn_loss", loss, prog_bar=True) 

    def validation_step(self, batch, batch_idx, print_log="val"):
        "validation iteration per batch"
        x, y, _ = batch

        output = self(x)
       
        loss = F.mse_loss(output, x)

        self.log(f"{print_log}_loss", loss, prog_bar=True)  
        return loss

    def test_step(self, batch, batch_idx, print_log="tst"):
        "test iteration per batch"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log="tst")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch
        if len(x.shape) > 4:
            x = x[0, ...]
            preds = self(x)
            # # print(logits.shape)
            # pp = torch.softmax(logits, dim=1)
            # max_pos = torch.argmax(pp[:, 1])
            # probs = pp[max_pos, :]
            # preds = torch.argmax(probs)
        else:
            preds = self(x)
            # print(logits.shape)
            # probs = torch.softmax(logits, dim=1)
            # preds = torch.argmax(probs, dim=1)

        return preds  
    
    def prepare_data(self) -> None:
        self.train_dataset = HeronDataset(set="train", resize_to=self.imsize)
        self.val_dataset = HeronDataset(set="val", resize_to=self.imsize)
        self.test_dataset = HeronDataset(set="test", resize_to=self.imsize)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers_loader)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return  DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers_loader)
    
    