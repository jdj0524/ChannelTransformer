import torch
from ..launcher.base_launcher import BaseLauncher
from ..models.transformer import ChannelTransformerSimple
from ..trainer.base_trainer import BaseTrainer
from ..trainer.sgdr_trainer import SGDR_Trainer
from ..dataloader.dataloader import DeepMIMOSampleDataset
from ..loss.nmse import MSE_loss, NMSE_loss, Cosine_distance
from torch.nn import MSELoss
def channeltransformer():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
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
                 'cosine' : Cosine_distance,
                 'NMSE' : NMSE_loss,
             },
        },
        'optimizer_options': {
            'lr' : 1e-3
        },
        'train_schedulers': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 10, 
                'T_mult' : 1,
                'eta_min':1e-8,
                'last_epoch':-1,
            },
        
    }
    return launcher, model, trainer, data, options