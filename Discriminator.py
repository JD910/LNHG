"""How to construct the discriminator network"""

import torch.nn as nn

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.fc = nn.Linear(128 * 1 * 256 * 256, 1) 
        # self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.fc(x) 
        # output = self.fc2(x)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features
    

