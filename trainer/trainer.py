import torch

class Trainer():
    def __init__(self, epochs, model, loaders, loss, optimizer) -> None:
        self.epochs = epochs
        self.model = model
        self.loaders = loaders
        self.loss = loss
        self.optimizer = optimizer
    
    def train(self):
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError