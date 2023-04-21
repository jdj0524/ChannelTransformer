from .trainer import Trainer
import torch

class BaseTrainer(Trainer):
    def __init__(self, epochs, model, loaders, loss, optimizer) -> None:
        super().__init__(epochs, model, loaders, loss, optimizer)
        