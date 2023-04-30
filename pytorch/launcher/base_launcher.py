import torch
from torch.utils.data import DataLoader
from .launcher import Launcher
from ..trainer.base_trainer import BaseTrainer
class BaseLauncher(Launcher):
    def __init__(self, options, data_cls, model_cls, trainer_cls) -> None:
            super().__init__(options)
            self.data_cls = data_cls
            self.model_cls = model_cls
            self.trainer_cls = trainer_cls
            self.trainer = None
    
    def build_dataset(self):
        data = self.data_cls(**self.options['data_options'])
        split_data = torch.utils.data.random_split(data, lengths = list(self.options['data_split'].values()))
        for i, key in enumerate(self.options['data_split'].keys()):
            self.loaders[key] = DataLoader(
                 split_data[i],
                 batch_size=self.options['batch_size'],
                 shuffle = key == 'train',
                 num_workers=4
                 )
    
    def build_model(self):
        model = self.model_cls(**self.options['model_options'])
        self.trainer = self.trainer_cls(model = model, options = self.options, **self.options['trainer_options'])
        self.trainer.set_loaders(self.loaders)
        self.trainer.build_optimizer()

    def run(self):
        self.build_dataset()
        self.build_model()
        self.trainer.train()

