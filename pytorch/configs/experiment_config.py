import torch
from ..lr_scheduler.sgdr import CosineAnnealingWarmUpRestarts
from ..launcher.base_launcher import BaseLauncher
from ..models.transformer import ChannelTransformerSimple, ChannelTransformerSimpleV2
from ..models.autoencoder_transformer import TransformerMIMOEncoder, CNNMIMOEncoder, TransformerMIMOEncoderNoise
from ..trainer.svd_trainer import SVDTrainer
from ..trainer.svd_shuffle_trainer import SVDShuffleTrainer
from ..trainer.svd_shuffle_trainer_noise import SVDShuffleTrainerNoise
from ..trainer.eval_trainer import DummyTrainer
from ..trainer.eval_trainer_snr import DummySNRTrainer
from ..models.csinet import ChannelAttention, CSINet
from ..models.mimo_methods import Collaboration_Transciever, MMSE_Transciever
from ..trainer.base_trainer import BaseTrainer
from ..trainer.sgdr_trainer import SGDR_Trainer
from ..dataloader.dataloader import DeepMIMOSampleDataset, DeepMIMOMultiuserDataset, DeepMIMOMultiuserDataset_Single, DeepMIMOMultiuserNoiseDataset
from ..loss.nmse import MSE_loss, NMSE_loss, Cosine_distance
from ..loss.mimo_rate import SumRate, Interference, SumRate_TX, Interference_TX, SumRate_SU, Interference_SU, ChannelCapacity, SumRate_Noise
from torch.nn import MSELoss
from copy import deepcopy
def channeltransformer_full():
    proto_config = channeltransformer()
    configs = []
    # feedback_lengths = [8,16,32,64,128,256]
    # feedback_lengths = [256, 128, 64, 32, 16, 8]
    feedback_lengths = [8, 16]
    for l in feedback_lengths:
        cur_config = deepcopy(proto_config)
        cur_config[0][4]['model_options']['dim_feedback'] = l
        configs += cur_config
    
    return configs

def channelattention_full():
    proto_config = channelattention()
    configs = []
    feedback_lengths = [8,16,32,64,128]
    for l in feedback_lengths:
        cur_config = deepcopy(proto_config)
        cur_config[0][4]['model_options']['dim_feedback'] = l
        configs += cur_config
    
    return configs
    
def channeltransformer():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 64,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_massive/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':256, 'nhead':4, 'dim_feedforward':1024, 
            'n_tx':256, 'n_rx':4, 'n_carrier':128, 'dim_feedback':16
        },
        'trainer_options': {
            'epochs' : 200, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-3,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 500,
                'T_mult' : 2,
                'T_up': 2,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    # trainer = BaseTrainer
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_massive/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':7, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':256, 'n_rx':4, 'n_carrier':128, 'dim_feedback':128
        },
        'trainer_options': {
            'epochs' : 500, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 500,
                'T_mult' : 2,
                'T_up': 10,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channelattention():
    launcher = BaseLauncher
    model = ChannelAttention
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 64,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_massive/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'no_blocks':3,
            'n_tx':256, 'n_rx':4, 'n_carrier':128, 'dim_feedback':128
        },
        'trainer_options': {
            'epochs' : 200, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 5e-4
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 30,
                'T_mult' : 2,
                'T_up': 2,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.5
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def csinet():
    launcher = BaseLauncher
    model = CSINet
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'no_blocks':5,
            'n_tx':16, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'trainer_options': {
            'epochs' : 500, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-3
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 30,
                'T_mult' : 2,
                'T_up': 2,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.5
            },
        
    }
    return [(launcher, model, trainer, data, options)]
