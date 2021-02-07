import numpy as np
import argparse
from tqdm import tqdm
import os
import cv2
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import models 
import torch.optim as optim
import logger
from datetime import datetime
from HDF5_Read import *
from Discriminator import *
from Generator import GeneratorNet, Train_Loss
from VGG19 import *
from Weight_Init import *
from Train_Disc import *

class JSLWGAN_GIoU:

    def __init__(self, args):

        # parameters
        self.lr = args.learning_rate
        self.epoch = args.epoch
        self.d_iter = args.d_iter
        self.batch_size = args.batch_size
        self.dicom_Height = args.dicom_Height
        self.dicom_Width = args.dicom_Width
        self.dataset = []
        self.dataloader = []

        self.lambda_gp = args.lambda_gp
        self.lambda_d_real = args.lambda_d_real
        self.lambda_d_fake = args.lambda_d_fake
        self.lambda_g_fake = args.lambda_g_fake
        self.lambda_mse = args.lambda_mse
        self.lambda_vgg = args.lambda_vgg
        self.lambda_giou = args.lambda_giou

        self.loss_dir = "./loss/"
        
        self.v = "256_256_" 
        self.save_dir = "./model/" + self.v + "/"
        self.save_model_dir = "./model/" + self.v + "/"
        self.time = '{}'.format(datetime.now().strftime('%b%d_%H%M')) 
        self.LoadModel = 'Load your Model/' 
        
        self.gpu = False


        self.discriminator = DiscriminatorNet().to("cuda:0")
        self.vgg19 = VGG19().to("cuda:0")
        # TODO: 
        self.generator = GeneratorNet().to("cuda:0")
        self.Train_Loss = Train_Loss().to("cuda:0")
        
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        
        self.G_loss = nn.MSELoss()

        if torch.cuda.is_available():
            if ',' in args.gpu_ids:
                gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
                print(gpu_ids)
            else:
                gpu_ids = int(args.gpu_ids)

            if type(gpu_ids) is not int:

                self.discriminator = nn.DataParallel(self.discriminator, device_ids = gpu_ids)
                self.vgg19 = nn.DataParallel(self.vgg19, device_ids = gpu_ids)
                self.generator = nn.DataParallel(self.generator, device_ids=gpu_ids)
                self.Train_Loss = nn.DataParallel(self.Train_Loss, device_ids=gpu_ids)
            self.gpu = True
    
        if not self.load_model():
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)


    def train(self, args):
        self.Traindataset = H5Dataset(args.input_dir_train)
        self.Traindataloader = torch.utils.data.DataLoader(
        self.Traindataset,
        batch_size=self.batch_size,
        num_workers=4, 
        shuffle=False,
        drop_last = True)

        self.Testdataset = H5Dataset(args.input_dir_test)
        self.Testdataloader = torch.utils.data.DataLoader(
            self.Testdataset,
            batch_size=self.batch_size,
            num_workers=4, 
            shuffle=False,
            drop_last = True)
        
        Test_Gloss = torch.tensor(1000) 
        Test_Dloss = torch.tensor(1000) 
        for batch_index in range(0, self.epoch):

            '''print(len(self.dataloader.dataset),len(self.dataloader))'''
            self.generator.train()
            self.discriminator.train()
            with tqdm(total=len(self.Traindataloader.dataset)) as progress_bar:
                for i, (Ori_img, Seg_img, Name_img) in enumerate(self.Traindataloader):
                    
                    Ori_img = Ori_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    Seg_img = Seg_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    
                    for _ in range(self.d_iter):
                        loss_d = Train_Disc(self, Seg_img, Ori_img)
                
                    z = Variable(Ori_img, requires_grad=True)
                    real_img = Variable(Seg_img, requires_grad=False) 

                    if self.gpu:
                        z = z.cuda()
                        real_img = real_img.cuda()

                    self.G_optimizer.zero_grad()
                    for p in self.discriminator.parameters():
                        p.requires_grad = False  # to avoid computation
                        
                    fake_img = self.generator(z).cuda()
                    fake_validity = self.discriminator(fake_img)
                    
                    if train:
                        for p in self.generator.parameters():
                            p.requires_grad = True  # to active computation

                        loss_g = self.Train_Loss(fake_img, real_img, fake_validity)
                        loss_g.backward()
                        self.G_optimizer.step()
                    progress_bar.update(self.batch_size)
                    
            self.discriminator.eval()
            self.generator.eval()
            with tqdm(total=len(self.Testdataloader.dataset)) as dev_progress_bar:

                for j, (Ori_img, Seg_img,Name_img) in enumerate(self.Testdataloader):
                    Ori_img = Ori_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    Seg_img = Seg_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)

                    loss_d = Train_Disc(self, Seg_img, Ori_img)
                    fake_img = self.generator(Seg_img).cuda()
                    real_img = Variable(Seg_img, requires_grad=False) 
                    fake_validity = self.discriminator(fake_img)
                    loss_g = self.Train_Loss(fake_img, real_img, fake_validity)

                    if Test_Gloss > abs(loss_g): 
                        Test_Gloss = abs(loss_g)
                        self.save_dir = self.save_model_dir + self.time + "/" + self.time
                        self.save_G_model(batch_index)
                    if Test_Dloss > abs(loss_d):
                        Test_Dloss = abs(loss_d)
                        self.save_dir = self.save_model_dir + self.time + "/" + self.time
                        self.save_D_model(batch_index)

                    dev_progress_bar.update(self.batch_size)


            if ((batch_index + 1) % 4 == 0 and self.lr > 1e-7):
                self.G_optimizer.defaults["lr"] *= 0.5
                self.D_optimizer.defaults["lr"] *= 0.5
                self.lr *= 0.5

    def load_model(self):
        if os.path.exists(self.save_dir + self.LoadModel + "G_256_256_" + ".pkl") and \
        os.path.exists(self.save_dir + self.LoadModel + "D_256_256_" + ".pkl"):

            self.generator.load_state_dict(
                torch.load(
                    self.save_dir + self.LoadModel + "G_256_256_" + ".pkl",
                    map_location='cuda:0'))
            self.discriminator.load_state_dict(
                torch.load(
                    self.save_dir + self.LoadModel + "D_256_256_" + ".pkl",
                    map_location='cuda:0'))
            print("load success")
            return True
        else:
            return False

    def save_G_model(self,batch_index):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.generator.state_dict(),
                   self.save_dir + str(batch_index) + "G_" + self.v + ".pkl")

    def save_D_model(self,batch_index):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.discriminator.state_dict(),
                   self.save_dir + str(batch_index) + "D_" + self.v + ".pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_train', type=str, default="Your Training Dataset")
    parser.add_argument('--input_dir_test', type=str,default="Your Test Dataset")
    parser.add_argument('--gpu_ids', type=str, default="Your GPUs")
    parser.add_argument('--log_train_name', type=str, default='Train')
    parser.add_argument('--log_test_name', type=str, default='Test')

    parser.add_argument('--learning_rate', type = float, default=1e-4)
    parser.add_argument('--batch_size', type = int, default = "Your Batch_size")
    parser.add_argument('--epoch', type = int, default = 200)
    parser.add_argument('--d_iter', type=int, default= "Default 5")
    parser.add_argument('--dicom_Height',type = int, default=256)
    parser.add_argument('--dicom_Width',type = int, default=256)

    args = parser.parse_args()

    mwgan = JSLWGAN_GIoU(args)
    # training
    mwgan.train(args)
