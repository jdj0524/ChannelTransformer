import torch
from ..lr_scheduler.sgdr import CosineAnnealingWarmUpRestarts
from ..launcher.base_launcher import BaseLauncher
from ..models.m2_bit_transformer import ChannelM2MixerBitLevel, ChannelTXRXM2MixerBitLevel
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

def m2_channel_bit_transformer():
    launcher = BaseLauncher
    model = ChannelM2MixerBitLevel
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.7, 
            'val': 0.1, 
            'test': 0.2
        },
        'model_options': {
            'config_tx':{
            },
            'config_rx':{
                "num_attention_heads": 12,
                "num_hidden_layers": 3, 
                'hidden_size': 512,
                'intermediate_size': 2048,
                "attention_probs_dropout_prob": 0.0, 
                "max_position_embeddings": 128,
                "monarch_mixer_sequence_mixing": True,
                "long_conv_l_max": 128,
                "long_conv_kernel_learning_rate": 1e-3,
                "hyena_lr_pos_emb": 1e-5,
                "hyena_w": 10,
                "hyena_w_mod": 1,
                "hyena_wd": 0.1,
                "hyena_emb_dim": 5,
                "hyena_filter_dropout": 0.2,
                "hyena_filter_order": 64,
                "hyena_training_additions": True,
                "bidirectional": True,
                "residual_long_conv": True,
                "use_glu_mlp": True,
                "use_monarch_mlp": True,
                "monarch_mlp_nblocks": 4,
                "use_positional_encodings": True,
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-5
            },
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':256
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 256,
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

def m2_channel_bit_transformer_txrx():
    launcher = BaseLauncher
    model = ChannelTXRXM2MixerBitLevel
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 128,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.7, 
            'val': 0.1, 
            'test': 0.2
        },
        'model_options': {
            'config_tx':{
                "num_attention_heads": 12,
                "num_hidden_layers": 1, 
                'hidden_size': 256,
                'intermediate_size': 1024,
                "attention_probs_dropout_prob": 0.0, 
                "max_position_embeddings": 128,
                "monarch_mixer_sequence_mixing": True,
                "long_conv_l_max": 128,
                "long_conv_kernel_learning_rate": 1e-3,
                "hyena_lr_pos_emb": 1e-5,
                "hyena_w": 10,
                "hyena_w_mod": 1,
                "hyena_wd": 0.1,
                "hyena_emb_dim": 5,
                "hyena_filter_dropout": 0.2,
                "hyena_filter_order": 64,
                "hyena_training_additions": True,
                "bidirectional": True,
                "residual_long_conv": True,
                "use_glu_mlp": True,
                "use_monarch_mlp": True,
                "monarch_mlp_nblocks": 4,
                "use_positional_encodings": True,
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-5
            },
            'config_rx':{
                "num_attention_heads": 12,
                "num_hidden_layers": 3, 
                'hidden_size': 512,
                'intermediate_size': 2048,
                "attention_probs_dropout_prob": 0.0, 
                "max_position_embeddings": 128,
                "monarch_mixer_sequence_mixing": True,
                "long_conv_l_max": 128,
                "long_conv_kernel_learning_rate": 1e-3,
                "hyena_lr_pos_emb": 1e-5,
                "hyena_w": 10,
                "hyena_w_mod": 1,
                "hyena_wd": 0.1,
                "hyena_emb_dim": 5,
                "hyena_filter_dropout": 0.2,
                "hyena_filter_order": 64,
                "hyena_training_additions": True,
                "bidirectional": True,
                "residual_long_conv": True,
                "use_glu_mlp": True,
                "use_monarch_mlp": True,
                "monarch_mlp_nblocks": 4,
                "use_positional_encodings": True,
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-5
            },
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':256
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 256,
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
