import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random
from einops import rearrange

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
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.data = None
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        print('file walk complete')
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        cur_data =np.load(os.path.join(self.files_dir, self.data[idx]+'.npy'))
        
        # cur_data = cur_data / (np.abs(cur_data).max(keepdims = True) + 1e-9)
        
        cur_data = cur_data / np.abs(cur_data).mean(keepdims = True)
        
        
        # cur_data_abs = np.abs(cur_data)
        # cur_data = (cur_data - cur_data.mean(keepdims = True)) / cur_data_abs.std(keepdims = True)
        
        cur_data = np.stack([cur_data.real, cur_data.imag], axis = -1)
        cur_data = torch.from_numpy(cur_data)#.bfloat16()
        
        return cur_data
    
class DeepMIMOMultiuserDataset(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/samples/', 
        max_users = 16
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.data = None
        self.chunked_data = None
        self.max_users = max_users
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        self.reshuffle_users()
        print('file walk complete')
        
    def reshuffle_users(self):
        batch_no = (len(self.data)//self.max_users) * self.max_users
        users = list(range(len(self.data)))
        random.shuffle(users)
        users = np.asarray(users[:batch_no])
        users = np.split(users, len(users) // self.max_users)
        self.chunked_data = users
    
    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self,idx):
        tensors = []
        for subidx in self.chunked_data[idx]:
            cur_data =np.load(os.path.join(self.files_dir, self.data[subidx]+'.npy'))
            cur_data = (cur_data / np.sqrt((cur_data*cur_data.conj()).sum(axis=(-1,-2), keepdims=True))).squeeze()
            cur_data = np.stack([cur_data.real, cur_data.imag], axis = -1)
            tensors.append(cur_data)
        tensors = rearrange(tensors, 'users nrx ntx complex -> users nrx ntx complex')
        return tensors
    
class DeepMIMOMultiuserDataset_Single(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/samples/', 
        max_users = 16
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.data = None
        self.max_users = max_users
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        print('file walk complete')
        
    def __len__(self):
        return len(self.data)    

    def __getitem__(self,idx):
        cur_data =np.load(os.path.join(self.files_dir, self.data[idx]+'.npy'))
        cur_data = (cur_data / np.sqrt((cur_data*cur_data.conj()).sum(axis=(-1,-2), keepdims=True))).squeeze()
        cur_data = np.stack([cur_data.real, cur_data.imag], axis = -1)
        return cur_data
    
class DeepMIMOMultiuserNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/samples/', 
        max_users = 16, train_snr_range = [-10, 40],
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.data = None
        self.max_users = max_users
        self.train_snr_range = train_snr_range
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        print('file walk complete')
        
    def snr_to_sigma(self, snr):
        return 1/10**(snr / 10)
    
    def __len__(self):
        return len(self.data)    

    def __getitem__(self,idx):
        cur_data =np.load(os.path.join(self.files_dir, self.data[idx]+'.npy'))
        cur_data = (cur_data / np.sqrt((cur_data*cur_data.conj()).sum(axis=(-1,-2), keepdims=True))).squeeze()
        cur_data = np.stack([cur_data.real, cur_data.imag], axis = -1).astype(np.float32)
        snr = np.random.uniform(low = self.train_snr_range[0], high = self.train_snr_range[1])
        sigma = np.float32(self.snr_to_sigma(snr))
        
        
        return cur_data, sigma