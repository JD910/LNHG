import h5py
import numpy as np
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        print(h5_path)
        with h5py.File(self.h5_path, 'r') as record:
            keys = list(record.keys())   
            #print(keys)

    def __getitem__(self, index):

        with h5py.File(self.h5_path, 'r') as record:
            keys = list(record.keys())
            train_data = np.array(record[keys[index]]['train']) 

            train_data = train_data[0::2,0::2]    #down_sampled to 1/2 of original size
            train_data[:,:] = (train_data[:,:] - np.mean(train_data[:,:]))/np.std(train_data[:,:]) 
    
            target_data = np.array(record[keys[index]]['target'])   
            target_data = target_data[0::2,0::2]  #down_sampled to 1/2 of original size

        return train_data, target_data, keys[index]

    def __len__(self):
        with h5py.File(self.h5_path,'r') as record:
            return len(record)

