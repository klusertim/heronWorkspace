import torch.nn as nn

# model from michele, modified to work with non-cropped images
class CAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=2048, ldim=8):
        super().__init__()

        self.encoder = nn.Sequential(         

            #324 x 216 - almost equal to spanish paper
            nn.Conv2d(image_channels, ldim, 5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(ldim),
            nn.LeakyReLU(True),

            #108 x 72
            nn.Conv2d(ldim, ldim * 2, 5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(2*ldim),
            nn.LeakyReLU(True),

            #36 x 24
            nn.Conv2d(ldim * 2, ldim * 4, 5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(4*ldim),
            nn.LeakyReLU(True),

            #12 x 8
            nn.Flatten(),

            #TODO: try lower dim here

            #12 x 8 x ldim * 4
        )

        # self.bottleneck = nn.Sequential(
        #     nn.Linear(8 * 6 * ldim * 2, h_dim),
        # )

        nonlin = nn.ReLU
        self.decoder = nn.Sequential(
            
            nn.Unflatten(1, (ldim * 4, 12, 8)),
            # 12 x 8

            nn.ConvTranspose2d(ldim * 4, ldim * 2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ldim * 2),
            nonlin(True),

            #36 x 24
            nn.ConvTranspose2d(ldim * 2, ldim, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ldim * 2),
            nonlin(True),

            #108 x 72
            nn.ConvTranspose2d(ldim, image_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ldim * 2),
            nonlin(True),

            #324 x 216
            nn.ConvTranspose2d(ldim * 2, ldim, 3, stride=2, padding=1, bias=False),
            
            nn.Tanh()

        )