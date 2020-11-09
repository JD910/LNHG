"""VGG19 net is used to evaluate the perceptual loss of generated images"""
from torchvision import models
import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)

        self.feature = vgg19.features
        #self.feature = vgg19
    '''
    input: N*1*D()*H*W
    output: N*C*H*W
    '''

    def forward(self, inputs):
        # VGG19: means:103.939, 116.779, 123.68

        input = inputs
        input /= 16
        depth = input.size()[2]
        result = []
        for i in range(depth):
            x = torch.cat(
                (input[:, :, i, :, :] - 103.939, input[:, :, i, :, :] - 116.779, input[:, :, i, :, :] - 123.68), 1)
            result.append(self.feature(x))

        output = torch.cat(result, dim=1)

        # output = self.feature(input)

        return output
