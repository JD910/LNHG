import torch
from torch import nn


class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
       
        layers = [
                    nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(out_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(int(in_channels/2+in_channels), out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        x = x[:, :, 0:1, :, :]
        x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class UNet(nn.Module):
    def __init__(self, init_channels=1, class_nums=1, batch_norm=True, sample=True):
        super(UNet, self).__init__()
        self.down1 = pub(init_channels, 64, batch_norm)
        self.down2 = unet3dDown(64, 128, batch_norm)
        self.down3 = unet3dDown(128, 256, batch_norm)
        self.down4 = unet3dDown(256, 512, batch_norm)
        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.con_last(x)
        return self.sigmoid(x)