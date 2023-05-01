import torch
from ..lr_scheduler.sgdr import CosineAnnealingWarmUpRestarts
from ..launcher.base_launcher import BaseLauncher
from ..models.transformer import ChannelTransformerSimple
from ..models.csinet import ChannelAttention, CSINet
from ..trainer.base_trainer import BaseTrainer
from ..trainer.sgdr_trainer import SGDR_Trainer
from ..dataloader.dataloader import DeepMIMOSampleDataset
from ..loss.nmse import MSE_loss, NMSE_loss, Cosine_distance
from torch.nn import MSELoss
from copy import deepcopy
def channeltransformer_full():
    proto_config = channeltransformer()
    configs = []
    feedback_lengths = [8,16,32,64,128,256]
    for l in feedback_lengths:
        cur_config = deepcopy(proto_config)
        cur_config[0][4]['model_options']['dim_feedback'] = l
        configs += cur_config
    
    return configs

def channelattention_full():
    proto_config = channelattention()
    configs = []
    feedback_lengths = [8,16,32,64,128,256]
    for l in feedback_lengths:
        cur_config = deepcopy(proto_config)
        cur_config[0][4]['model_options']['dim_feedback'] = l
        configs += cur_config
    
    return configs
    
def channeltransformer():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/jdj0524/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/jdj0524/DeepMIMO_Datasets/O1_140/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':256, 'nhead':4, 'dim_feedforward':1024, 
            'n_tx':16, 'n_rx':16, 'n_carrier':128, 'dim_feedback':32
        },
        'trainer_options': {
            'epochs' : 300, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-9
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 15,
                'T_mult' : 5,
                'T_up': 2,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.5
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channelattention():
    launcher = BaseLauncher
    model = ChannelAttention
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channelattention',
        'save_dir': '/home/jdj0524/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/jdj0524/DeepMIMO_Datasets/O1_140/samples/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'no_blocks':3,
            'n_tx':16, 'n_rx':16, 'n_carrier':128, 'dim_feedback':32
        },
        'trainer_options': {
            'epochs' : 300, 
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
                'T_0' : 15,
                'T_mult' : 5,
                'T_up': 2,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.5
            },
        
    }
    return [(launcher, model, trainer, data, options)]