import torch
from ..lr_scheduler.sgdr import CosineAnnealingWarmUpRestarts
from ..launcher.base_launcher import BaseLauncher
from ..models.transformer import ChannelTransformerSimple, ChannelTransformerSimpleV2
from ..models.quantization_transformer import ChannelTransformerQuantization, ChannelTransformerQuantizationLarge, ChannelTransformerPQB
from ..models.bit_transformer import ChannelTransformerBitLevel, ChannelTransformerBitLevelLarge, ChannelTransformerBitLevelOneLinear, ChannelTransformerBitLevelLargeVariable, ChannelTransformerBitLevelLargeVariableMask
from ..models.autoencoder_transformer import TransformerMIMOEncoder, CNNMIMOEncoder, TransformerMIMOEncoderNoise
from ..trainer.svd_trainer import SVDTrainer
from ..trainer.base_eval_trainer import BaseEvalTrainer
from ..trainer.svd_shuffle_trainer import SVDShuffleTrainer
from ..trainer.sgdr_bit_trainer import SGDR_Bit_Trainer
from ..trainer.svd_shuffle_trainer_noise import SVDShuffleTrainerNoise
from ..trainer.eval_trainer import DummyTrainer
from ..trainer.eval_trainer_snr import DummySNRTrainer
from ..models.csinet import ChannelAttention, CSINet
from ..models.csi_methods import Lasso_Compressive
from ..models.mimo_methods import Collaboration_Transciever, MMSE_Transciever
from ..trainer.base_trainer import BaseTrainer
from ..trainer.sgdr_trainer import SGDR_Trainer
from ..trainer.sgdr_bit_variable_trainer import SGDR_Bit_Variable_Trainer
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

def channeltransformer_v2():
    launcher = BaseLauncher
    model = ChannelTransformerSimpleV2
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':256, 'nhead':4, 'dim_feedforward':1024, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
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

    
def channeltransformer():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':256, 'nhead':4, 'dim_feedforward':1024, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
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
            'lr' : 5e-4,
            
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

def channeltransformer_bit():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevel
    trainer = BaseTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':256, 'nhead':4, 'dim_feedforward':1024, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':512
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
            'lr' : 5e-4,
            
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
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevel
    trainer = SGDR_Bit_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_large():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLarge
    trainer = SGDR_Bit_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':32
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.1
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_large_blockage():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLarge
    trainer = SGDR_Bit_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_variable():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariable
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.7, 
            'val': 0.1, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 128,
            'eval_bits' : 128,
            'epochs' : 2000, 
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
                'T_0' : 2000,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_variable_mask():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariableMask
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.7, 
            'val': 0.1, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':512
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 512,
            'eval_bits' : 128,
            'epochs' : 2000, 
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
                'T_0' : 2000,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_variable_mask_blockage():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariableMask
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.7, 
            'val': 0.1, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 128,
            'eval_bits' : 128,
            'epochs' : 2000, 
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
                'T_0' : 2000,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]



def channeltransformer_sgdr_bit_onelinear():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelOneLinear
    trainer = SGDR_Bit_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_quantization():
    launcher = BaseLauncher
    model = ChannelTransformerQuantization
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128, 'bit_level': 1
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 200,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_quantization_blockage():
    launcher = BaseLauncher
    model = ChannelTransformerQuantization
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':64, 'bit_level': 1
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 200,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_pqb():
    launcher = BaseLauncher
    model = ChannelTransformerPQB
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':16, 'bit_level': 8
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_pqb_blockage():
    launcher = BaseLauncher
    model = ChannelTransformerPQB
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':16, 'bit_level': 8
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]


def channeltransformer_sgdr_quantization_large():
    launcher = BaseLauncher
    model = ChannelTransformerQuantizationLarge
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':16, 'bit_level': 1
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
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
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 200,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def lasso():
    launcher = BaseLauncher
    model = Lasso_Compressive
    trainer = BaseEvalTrainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 64,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':128
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
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_cost2100():
    launcher = BaseLauncher
    model = ChannelTransformerSimple
    # trainer = BaseTrainer
    trainer = SGDR_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'channeltransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints_cost2100/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/I2_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':32, 'n_rx':1, 'n_carrier':256, 'dim_feedback':32
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
                'T_up': 5,
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
        'batch_size': 100,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'no_blocks':3,
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':8
        },
        'trainer_options': {
            'epochs' : 100, 
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
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'no_blocks':5,
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':8
        },
        'trainer_options': {
            'epochs' : 100, 
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
