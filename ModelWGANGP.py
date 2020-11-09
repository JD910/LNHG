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
from Generator import *
from VGG19 import *
from Weight_Init import *
from Train_Disc import *
from WGANGPTrain_Gen import *
from Train_GIoU import *

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
        self.Train_GIoU = Train_GIoU().to("cuda:0")
        
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
                self.Train_GIoU = nn.DataParallel(self.Train_GIoU, device_ids=gpu_ids)
            self.gpu = True
    
        if not self.load_model():
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)


    def train(self, args):

        self.save_parameters()
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
                    #print(Name_img)
                    #print(Ori_img.shape,Ori_img.dtype,Seg_img.shape,Seg_img.dtype)
                    
                    Ori_img = Ori_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    Seg_img = Seg_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    
                    for _ in range(self.d_iter):
                        loss_d = Train_Disc(self, Seg_img, Ori_img)
                        #print("\tD_loss - lr: %.10f, Epoch: %d, bath_index: %d, er: %d, D-Loss: " %
                        #       (self.lr, self.epoch, batch_index, iter_i), loss_d)
                
                    loss_g = WGANGPTrain_Gen(self, Seg_img, Ori_img, batch_index)

                    if Test_Gloss > abs(loss_g[0]): 
                        Test_Gloss = abs(loss_g[0])
                        self.save_dir = self.save_model_dir + self.time + "/" + self.time
                        self.save_G_model(batch_index)
                    if Test_Dloss > abs(loss_d[0]):
                        Test_Dloss = abs(loss_d[0])
                        self.save_dir = self.save_model_dir + self.time + "/" + self.time
                        self.save_D_model(batch_index)

                    self.save_loss(loss_d[0], loss_d[1], loss_d[2], loss_d[3], loss_g[0],loss_g[1],loss_g[2], 'Train')

                    progress_bar.update(self.batch_size)
                    
            self.discriminator.eval()
            self.generator.eval()
            with tqdm(total=len(self.Testdataloader.dataset)) as dev_progress_bar:

                for j, (Ori_img, Seg_img,Name_img) in enumerate(self.Testdataloader):
                    Ori_img = Ori_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    Seg_img = Seg_img.view(self.batch_size, 1, 1, self.dicom_Height, self.dicom_Width)
                    with torch.set_grad_enabled(True):
                        loss_d = Train_Disc(self, Seg_img, Ori_img)
                        loss_g = WGANGPTrain_Gen(self, Seg_img, Ori_img,batch_index)

                        if Test_Gloss > abs(loss_g[0]): 
                            Test_Gloss = abs(loss_g[0])
                            self.save_dir = self.save_model_dir + self.time + "/" + self.time
                            self.save_G_model(batch_index)
                        if Test_Dloss > abs(loss_d[0]):
                            Test_Dloss = abs(loss_d[0])
                            self.save_dir = self.save_model_dir + self.time + "/" + self.time
                            self.save_D_model(batch_index)

                        self.save_loss(loss_d[0], loss_d[1], loss_d[2], loss_d[3], loss_g[0],loss_g[1],loss_g[2], 'Test')

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
                   
    def save_model(self,batch_index):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.generator.state_dict(),
                   self.save_dir + "G_" + self.v + ".pkl")
        torch.save(self.discriminator.state_dict(),
                   self.save_dir + "D_" + self.v + ".pkl")
        print(batch_index)

    def save_loss(self, loss_d1, loss_d2, loss_d3, loss_d4, loss_g1,loss_g2,loss_g3, flag):
        value = ""
        value = value+str(loss_d1)+" "+str(loss_d2)+" "+str(loss_d3)+" "+str(loss_d4)+" "+str(loss_g1)+" "+str(loss_g2)+" "+str(loss_g3)
        value += "\n"
        if flag == 'Train':
            with open("./loss/" + self.v + self.time + "_Train" + ".csv", "a+") as f:
                f.write(value)
        else:
            with open("./loss/" + self.v + self.time + "_Test" + ".csv", "a+") as f:
                f.write(value)

    def save_parameters(self):
        value = "Time: " + '{}'.format(datetime.now().strftime('%b%d_%H%M')) + ","\
        +"lambda_vgg: " + str(self.lambda_vgg) + ","\
        + "lambda_d_real: " + str(self.lambda_d_real) + ","\
        + "lambda_d_fake: " + str(self.lambda_d_fake) + ","\
        + "lambda_g_fake: " + str(self.lambda_g_fake) + ","\
        + "lambda_giou: " + str(self.lambda_giou)+"\n"
        with open("./parameters/" + "parameters.file", "a+") as f:
            f.write(value)

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
    parser.add_argument('--lambda_gp', type = float, default = 10) #Default is 10.0
    parser.add_argument('--lambda_d_real', type = float, default=1.0) #Default is 1.0
    parser.add_argument('--lambda_d_fake', type = float, default = 1.0) #Default is 1.0
    parser.add_argument('--lambda_g_fake', type = float, default = 1e-1) #Default is 0.1
    parser.add_argument('--lambda_mse', type = float, default = 1.0) #Default is 1.0
    parser.add_argument('--lambda_vgg', type = float, default = 1e-1) #Default is 0.1
    parser.add_argument('--lambda_giou', type = float, default = 10.0) #Default is 10.0

    args = parser.parse_args()

    mwgan = JSLWGAN_GIoU(args)
    # training
    mwgan.train(args)
