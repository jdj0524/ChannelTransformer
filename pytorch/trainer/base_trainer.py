from .trainer import Trainer
import torch
import numpy as np

class BaseTrainer(Trainer):
    def __init__(self, epochs, model, loaders, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(epochs, model, loaders, loss, optimizer_cls, gpu, metrics, options)

    def train_step(self, epoch):
        step_losses = []
        for data in self.loaders['train']:
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data)
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train(self):
        for i in range(self.epochs):
            self.train_step(i)
            self.eval()
            self.print_eval_metrics()
    
    def eval(self):
        eval_metrics = {}
        for key in self.metrics.keys():
            eval_metrics[key] = []
        eval_metrics['val_loss'] = []

        for data in self.loaders['val']:
            self.model.eval()
            data = data.to(self.gpu)
            output = self.model(data).detach()
            for key in self.metrics.keys():
                eval_metrics[key].append(self.metrics[key](output, data).mean().cpu())
            eval_metrics['val_loss'] = self.loss(output, data)

        for key in eval_metrics.keys():
            self.eval_metric_history[key].append(np.mean(eval_metrics[key]))