import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
from VGG19 import *
import cv2
import skimage.measure as measure

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
    
class Train_Loss(nn.Module):

    def __init__(self):
        super(Train_Loss, self).__init__()
        self.criterion_loss = nn.MSELoss()
        self.vgg = VGG19().to("cuda:0")
        self.giouloss = GIoU().to("cuda:0")

    def forward(self, fake_img, real_img, fake_validity):
        
        Mse_loss = self.criterion_loss(fake_img.float(), real_img.float())
        VGG19_loss = VGG_Loss(self,fake_img, real_img)
        GIoU_loss = self.giouloss(fake_img,real_img)
        G_Loss = GIoU_loss.mean() + Mse_loss + VGG19_loss
        return G_Loss

class GIoU(nn.Module):
    def __init__(self):
        super(GIoU, self).__init__()

    
    def forward(self,fake_img,real_img):
        loss = giou_loss(self,fake_img,real_img)
        return loss

def giou_loss(self,fake,real):
    
    giou_batch = torch.zeros(1, fake.shape[0], requires_grad=True)
    Threshold_batch = torch.zeros(1, fake.shape[0])
    C_domain_batch = torch.zeros(1, fake.shape[0])

    for i in range(fake.shape[0]): 
        fake_img_i = fake[i].clone()
        real_img_i = real[i].clone()
        Box_gt = torch.zeros([1, 4])

        fake_GIoU = fake_img_i.squeeze()
        fake_GIoU[:, :] = (fake_GIoU[:, :] - torch.min(fake_GIoU[:, :])) / \
                            (torch.max(fake_GIoU[:, :]) -
                            torch.min(fake_GIoU[:, :]))
        real_GIoU = real_img_i.squeeze()
        points = torch.nonzero(real_GIoU)
        Box_gt = torch.Tensor([min(points[:, 0]), min(
            points[:, 1]), max(points[:, 0]), max(points[:, 1])]).cuda()

        Threshold = 'Please select your best Threshold for your own dataset' ## the rage of 0.15~0.20 is best based on our training
        MaxGIoU = -1.0
        MaxDice = -1.0
        while (Threshold != 0):

            MaskImg = torch.where(fake_GIoU > Threshold, torch.full_like(
                    fake_GIoU, 1), torch.full_like(fake_GIoU, 0))
            max_giou, max_dice = self.Max_GIoU_calculate(
                    i,MaskImg,Box_gt,Threshold,MaxGIoU,Threshold_batch,giou_batch,\
                        C_domain_batch,real_GIoU,MaxDice)
            MaxGIoU = max_giou
            MaxDice =max_dice

    loss_giou = 1 - giou_batch
    return loss_giou.cuda()

def Max_GIoU_calculate(self,i,MaskImg,Box_gt,Threshold,MaxGIoU,Threshold_batch,giou_batch,C_domain_batch,real_GIoU,MaxDice):

        labels = skimage.measure.label(MaskImg.cpu())  
        num = labels.max() 

        for k in range(num):  
            del_array = torch.Tensor([0] * (num + 1))  
            del_array[k+1] = 1
            del_mask = del_array[labels]
            out = (MaskImg.cuda()) * (del_mask.cuda())
            Box_p = torch.zeros([1, 4])
            points = torch.nonzero(out) 
            Box_p = torch.Tensor([min(points[:, 0]), min(points[:,1]), max(points[:,0]), max(points[:,1])]).cuda()
           
            dice = dice_coeff(out, real_GIoU)

            assert Box_p.shape == Box_gt.shape
            Box_p = Box_p.float()
            Box_gt = Box_gt.float()

            if (Box_p[2] - Box_p[0] == 0) | (Box_p[3]  - Box_p[1] == 0):
                box_p_area = (Box_p[2]  - Box_p[0] + 1)  * (Box_p[3]  - Box_p[1] + 1)
            else:
                box_p_area = (Box_p[2]  - Box_p[0])  * (Box_p[3]  - Box_p[1])

            if(Box_gt[2] - Box_gt[0] == 0) |(Box_gt[3] - Box_gt[1] == 0):
                box_gt_area = (Box_gt[2] - Box_gt[0] + 1) * \
                                (Box_gt[3] - Box_gt[1] + 1)
            else:
                box_gt_area = (Box_gt[2] - Box_gt[0]) * (Box_gt[3] - Box_gt[1])

            xI_1 = torch.max(Box_p[0], Box_gt[0])
            xI_2 = torch.min(Box_p[2], Box_gt[2])
            yI_1 = torch.max(Box_p[1], Box_gt[1])
            yI_2 = torch.min(Box_p[3], Box_gt[3])

            # intersection =(yI_2 - yI_1) * (xI_2 - xI_1)
            intersection = torch.max((yI_2 - yI_1), torch.tensor(0.0).cuda()) * \
                                        torch.max((xI_2 - xI_1), torch.tensor(0.0).cuda())

            xC_1 = torch.min(Box_p[0], Box_gt[0])
            xC_2 = torch.max(Box_p[2], Box_gt[2])
            yC_1 = torch.min(Box_p[1], Box_gt[1])
            yC_2 = torch.max(Box_p[3], Box_gt[3])

            c_area = (xC_2 - xC_1) * (yC_2 - yC_1)
            union = box_p_area + box_gt_area - intersection
            iou = intersection / union
            # GIoU
            giou = iou - (c_area - union) / c_area

            if(iou > MaxGIoU):  
                    MaxGIoU = iou
                    Threshold_batch[0, i].data.fill_(Threshold)
                    giou_batch[0, i].data.fill_(giou)
                    C_domain_batch[0, i].data.fill_(box_gt_area)
        
        return MaxGIoU, MaxDice

def dice_coeff(pred, target):
    smooth = 1.
    P = pred.sum()
    T = target.sum()
    intersection = (pred * target).sum()
    Dice = (2. * intersection + smooth) / (P + T + smooth)
    return Dice

def VGG_Loss(self, fake_img,real_img):
    feature_fake_vgg = self.vgg(fake_img)
    feature_real_vgg = Variable(self.vgg(real_img), requires_grad=False).cuda()
    return self.criterion_loss(feature_fake_vgg, feature_real_vgg)
