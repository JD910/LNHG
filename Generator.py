"""How to construct the generator network"""
import torch.nn as nn

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        # torch.nn.init.
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )

        self.deConv1_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.deConv2_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.deConv3_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.deConv4_1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)

        self.deConv4 = nn.ReLU()

    def forward(self, input):

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x = self.deConv1_1(conv4)
        x = x + conv3  
        deConv1 = self.deConv1(x)

        x = self.deConv2_1(deConv1)
        x += conv2
        deConv2 = self.deConv2(x)

        x = self.deConv3_1(deConv2)
        x += conv1
        deConv3 = self.deConv3(x)

        x = self.deConv4_1(deConv3)
        x += input
        output = self.deConv4(x)

        return output
