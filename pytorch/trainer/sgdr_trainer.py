from .base_trainer import BaseTrainer
import torch

class SGDR_Trainer(BaseTrainer):
    def __init__(self, epochs, model, loaders, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(epochs, model, loaders, loss, optimizer_cls, gpu, metrics, options)
        self.iters = len(self.loaders['train'])

    def train_step(self, epoch):
        step_losses = []
        for i, data in enumerate(self.loaders['train']):
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data)
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + i / self.iters)
    
    def build_optimizer(self):
        super().build_optimizer()
        self.scheduler = self.optimizer['train_schedulers'](
            optimizer = self.optimizer, **self.options['train_scheduler_options']
            )
        
            