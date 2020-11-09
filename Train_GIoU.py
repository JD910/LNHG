import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import cv2
import skimage

class Train_GIoU(nn.Module):
    def __init__(self):
        super(Train_GIoU, self).__init__()


    def forward(self, fake_img, real_img):
        """
        Input:
            Box_p : (n,4)(x1,y,x2,y2),
            Box_gt: (n,4)(x1,y1,x2,y2)
        Output:
            loss_giou
        """

        fake = fake_img
        real = real_img
        
        giou_batch = torch.zeros(1, fake.shape[0])
        Threshold_batch = torch.zeros(1, fake.shape[0])
        C_domain_batch = torch.zeros(1, fake.shape[0])
        Dice_batch = torch.zeros(1, fake.shape[0])

        for i in range(fake.shape[0]): 

            Box_gt = torch.zeros([1, 4])

            fake_GIoU = fake[i].squeeze()
            fake_GIoU[:, :] = (fake_GIoU[:, :] - torch.min(fake_GIoU[:, :])) / \
                                (torch.max(fake_GIoU[:, :]) -
                                torch.min(fake_GIoU[:, :]))
            real_GIoU = real[i].squeeze()

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
                        C_domain_batch,real_GIoU,Dice_batch,MaxDice)
                MaxGIoU = max_giou
                MaxDice =max_dice


        loss_giou = 1 - giou_batch
        
        return loss_giou.cuda(), giou_batch.cuda(), Threshold_batch.cuda(),Dice_batch.cuda()

    def fillHole(self, im_in):

        im_floodfill = im_in
        im_inout = im_in
        im_inout = im_inout.data.cpu().numpy().astype(np.uint8)
        
        h, w = im_in.shape[:2]
        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        im_floodfill = im_floodfill.data.cpu().numpy().astype(np.uint8)
        
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = im_inout | im_floodfill_inv
        im_out = torch.from_numpy(im_out).to(torch.float32)
       
        return im_out

    def Max_GIoU_calculate(self,i,MaskImg,Box_gt,Threshold,MaxGIoU,Threshold_batch,giou_batch,C_domain_batch,real_GIoU,Dice_batch,MaxDice):

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
            
            dice = self.dice_coeff(out, real_GIoU)
            if (dice > MaxDice):
                MaxDice = dice
                Dice_batch[0,i] = dice

            
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
                    Threshold_batch[0, i] = Threshold 
                    giou_batch[0, i] = giou 
                    C_domain_batch[0, i] = box_gt_area
        
        return MaxGIoU, MaxDice

    def dice_coeff(self, pred, target):
        smooth = 1.
        P = pred.sum()
        T = torch.tensor(target.sum())
        intersection = (pred * target).sum()
        
        Dice = (2. * intersection + smooth) / (P + T + smooth)
        
        return Dice
        