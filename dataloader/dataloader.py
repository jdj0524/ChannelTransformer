import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class DeepMIMODataset(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/', 
        target_file = 'o1_channels_grid_1.npy'
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.target_file = target_file
        self.data = None
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + self.target_file)
        self.data = self.data / np.sqrt((self.data * self.data.conjugate()).real).max(axis=(1,2,3), keepdims=True)
        print('normalized')
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.data[idx]
    
class DeepMIMOSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/samples/', 
        target_file = '*.npy'
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.target_file = target_file
        self.data = None
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        print('file walk complete')
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        cur_data =np.load(os.path.join(self.files_dir, self.data[idx]+'.npy'))
        
        cur_data = cur_data / np.abs(cur_data).max(keepdims = True)
        
        # cur_data = cur_data / np.abs(cur_data).mean(keepdims = True)
        
        cur_data = np.concatenate([cur_data.real, cur_data.imag], axis = 2)
        
        return cur_data
    