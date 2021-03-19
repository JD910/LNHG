import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import cv2
import skimage

def Train_Gene(self, Seg_img, Ori_img, batch_index, train = True):
    
    z = Variable(Ori_img)
    real_img = Variable(Seg_img, requires_grad=False) 
    z = Variable(fake_img)

    if self.gpu:
        z = z.cuda()
        real_img = real_img.cuda()
        
    self.G_optimizer.zero_grad()
    criterion_mse = nn.MSELoss()
    criterion_vgg= nn.MSELoss()
    fake_img = self.generator(z).cuda()
    mse_loss = criterion_mse(fake_img, real_img)

    if train:
        (self.lambda_mse * mse_loss).backward(retain_graph = True)

    feature_fake_vgg = self.vgg19(fake_img)

    feature_real_vgg = Variable(self.vgg19(real_img).data, requires_grad=False).cuda()

    vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)
    fake_validity = self.discriminator(fake_img)
    g_loss = self.lambda_vgg * vgg_loss + self.lambda_g_fake * torch.mean(-fake_validity)

    if train:
        g_loss.backward()
        self.G_optimizer.step()

    return g_loss.data.item()

