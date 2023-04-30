import torch
class Trainer:
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        self.options = options
        self.epochs = epochs
        self.model = model
        self.loss = loss
        self.optimizer_cls = optimizer_cls
        self.gpu = gpu
        self.metrics = metrics
        self.scheduler = None
        self.eval_metric_history = {'train_loss' : [], 'val_loss' : []}
        for key in self.metrics.keys():
            self.eval_metric_history[key] = []

    def build_optimizer(self):
            self.optimizer = self.optimizer_cls(self.model.parameters(), **self.options['optimizer_options'])

    def set_loaders(self, loaders):
        self.loaders = loaders
    
    def train_step(self, epoch):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def print_eval_metrics(self):
        print_string = ''
        for key in self.eval_metric_history.keys():
            print_string += "{} : {}, ".format(key, self.eval_metric_history[key][-1])
        print(print_string)