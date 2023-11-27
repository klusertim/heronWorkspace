
class AEHeronModelV1(pl.LightningModule):
    def __init__(self, hparams):
        super(AEHeronModelV1, self).__init__()
        self.hparams = hparams
        self.encoder = EncoderV1(hparams)
        self.decoder = DecoderV1(hparams)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.hparams.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.hparams.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.hparams.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4