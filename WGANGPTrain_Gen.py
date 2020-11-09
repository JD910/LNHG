import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import cv2
import skimage

def WGANGPTrain_Gen(self, free_img, noised_img,  batch_index, train=True):
    z = Variable(noised_img)
    real_img = Variable(free_img, requires_grad=False) 

    if self.gpu:
        z = z.cuda()
        real_img = real_img.cuda()

    self.G_optimizer.zero_grad()
   
    #self.D_optimizer.zero_grad()
    #self.vgg19.zero_grad()

    #for p in self.discriminator.parameters():
    #    p.requires_grad = False  # to avoid computation

    criterion_mse = nn.MSELoss()
    criterion_vgg= nn.MSELoss()
    GIoU_loss = torch.tensor(0).cuda()
    fake_img = self.generator(z).cuda()
    mse_loss = criterion_mse(fake_img, real_img)

    #fake_img_samples  =  fake_img.mul(0.5).add(0.5)
    fake_img_samples  =  fake_img

    if train:
        (self.lambda_mse * mse_loss).backward(retain_graph=True)
    
    feature_fake_vgg = self.vgg19(fake_img)

    feature_real_vgg = Variable(self.vgg19(real_img).data, requires_grad=False).cuda()

    vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)


    fake_validity = self.discriminator(fake_img / 4096) 
    giou_batch = torch.zeros(1, fake_img.shape[0])
    Threshold_batch = torch.zeros(1, fake_img.shape[0])
    GIoU_loss_batch = torch.zeros(1, fake_img.shape[0])
    Dice_batch = torch.zeros(1, fake_img.shape[0])

    if(batch_index > 60):
        GIoU_loss_batch, giou_batch, Threshold_batch, Dice_batch = self.Train_GIoU(fake_img, free_img)
        GIoU_loss_batch = GIoU_loss_batch.view(1,-1)
        giou_batch = giou_batch.view(1,-1)
        Threshold_batch = Threshold_batch.view(1,-1)
        Dice_batch = Dice_batch.view(1,-1)
        GIoU_loss = GIoU_loss_batch.mean()
        Dice_mean = Dice_batch.mean()

    if(batch_index < 80):
        g_loss =  self.lambda_vgg * vgg_loss + self.lambda_g_fake * torch.mean(-fake_validity)
    else:

        g_loss =  Variable(self.lambda_giou * torch.mean(GIoU_loss_batch) + self.lambda_g_fake * torch.mean(-fake_validity),requires_grad = True)

    if train:
        g_loss.backward()
        self.G_optimizer.step()

    return g_loss.data.item(), self.lambda_giou * GIoU_loss.data.item(), \
        self.lambda_g_fake * torch.mean(-fake_validity).data.item(), self.lambda_vgg * vgg_loss.data.item(),\
             fake_img_samples, Dice_mean, \
                 Threshold_batch,Dice_batch