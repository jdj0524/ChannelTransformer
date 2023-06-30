def SVDTransformer():
    launcher = BaseLauncher
    model = TransformerMIMOEncoder
    trainer = SVDTrainer
    data = DeepMIMOMultiuserDataset
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 10,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 4
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 4,
            'd' : 3,
            'nblocks' : 3,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 1024,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'channel_capacity' : (ChannelCapacity, 'max'),
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

def SVDShuffleTransformer():
    launcher = BaseLauncher
    model = TransformerMIMOEncoder
    trainer = SVDShuffleTrainer
    data = DeepMIMOMultiuserDataset_Single
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 0,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 3,
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 3,
            'd' : 3,
            'nblocks' : 2,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 1024,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'channel_capacity' : (ChannelCapacity, 'max'),
                 'sumrate_tx' : (SumRate_TX, 'max'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-4
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

def SVDNoiseShuffleTransformer():
    launcher = BaseLauncher
    model = TransformerMIMOEncoderNoise
    trainer = SVDShuffleTrainerNoise
    data = DeepMIMOMultiuserNoiseDataset
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 1024,
        'train_snr': 10,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 3,
            'train_snr_range': [-10, 40]
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 3,
            'd' : 3,
            'nblocks' : 3,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 1024,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'channel_capacity' : (ChannelCapacity, 'max'),
                 'sumrate_tx' : (SumRate_TX, 'max'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-4
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


def SVDCNN():
    launcher = BaseLauncher
    model = CNNMIMOEncoder
    trainer = SVDTrainer
    data = DeepMIMOMultiuserDataset
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 10,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 4
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 4,
            'd' : 3,
            'nblocks' : 3,
        },
        'trainer_options': {
            'epochs' : 500, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'sumrate_tx' : (SumRate_TX, 'max'),
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


def SVDShuffleCNN():
    launcher = BaseLauncher
    model = CNNMIMOEncoder
    trainer = SVDShuffleTrainer
    data = DeepMIMOMultiuserDataset_Single
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 0,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 3
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 3,
            'd' : 3,
            'nblocks' : 3,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'sumrate_tx' : (SumRate_TX, 'max'),
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

def Full_Collaboration():
    launcher = BaseLauncher
    model = Collaboration_Transciever
    trainer = DummyTrainer
    data = DeepMIMOMultiuserDataset
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 0,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 4
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 4,
            'd' : 3,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate_SU,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference_SU, 'min'),
                 'channel_capacity' : (ChannelCapacity, 'max'),
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

def MSMSE():
    launcher = BaseLauncher
    model = MMSE_Transciever
    trainer = DummyTrainer
    data = DeepMIMOMultiuserDataset
    options = {
        'wandb_project_name': 'mimotransformer',
        'save_dir': '/home/automatic/projects/ChannelTransformer/checkpoints/',
        'batch_size': 256,
        'train_snr': 0,
        'test_snr': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40],
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_autoencoder/samples/',
            'max_users': 4
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'ntx' : 16,
            'nrx' : 4,
            'k' : 4,
            'd' : 3,
        },
        'trainer_options': {
            'epochs' : 2000, 
            'loss' : SumRate,
             'optimizer_cls' : torch.optim.AdamW,
             'metrics' : {
                 'interference' : (Interference, 'min'),
                 'sumrate_noise' : (SumRate_Noise, 'min'),
                 'channel_capacity' : (ChannelCapacity, 'max'),
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