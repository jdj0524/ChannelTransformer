from .trainer import Trainer
import torch
import numpy as np
import wandb
from copy import deepcopy
import time
from .base_trainer import BaseTrainer
from .svd_shuffle_trainer import SVDShuffleTrainer
from einops import rearrange

class SVDShuffleTrainerNoise(SVDShuffleTrainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        self.k = self.options['model_options']['k']
    def train_step(self, epoch):
        step_losses = []
        batch_data = []
        batch_sigma = []
        for data, sigma in self.loaders['train']:
            batch_data.append(data)
            batch_sigma.append(sigma)
            if len(batch_data) == self.k:
                data = rearrange(batch_data, 'users batch nrx ntx complex -> batch users nrx ntx complex')
                sigma = rearrange(batch_sigma, 'users batch -> batch users 1')
                batch_data = []
                batch_sigma = []
                self.model.train()
                data = data.to(self.gpu)
                sigma = sigma.to(self.gpu)
                u, v = self.model(data, sigma)
                step_loss = self.loss(u, v, data, sigma).mean()
                step_losses.append(step_loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
    
    def eval(self):
        eval_metrics = {}
        for key in self.metrics.keys():
            eval_metrics[key] = []
        eval_metrics['val_loss'] = []
        batch_data = []
        batch_sigma = []
        for data, sigma in self.loaders['val']:
            batch_data.append(data)
            batch_sigma.append(sigma)
            if len(batch_data) == self.k:
                data = rearrange(batch_data, 'users batch nrx ntx complex -> batch users nrx ntx complex')
                sigma = rearrange(batch_sigma, 'users batch -> batch users 1')
                batch_data = []
                batch_sigma = []
                sigma = torch.ones(sigma.shape, dtype = torch.float32).to(self.gpu) * self.train_sigma
                self.model.eval()
                data = data.to(self.gpu)
                u, v = self.model(data, sigma)
                u = u.detach()
                v = v.detach()
                for key in self.metrics.keys():
                    eval_metrics[key].append(self.metrics[key](u, v, data, sigma).mean().cpu().numpy())
                eval_metrics['val_loss'] = self.loss(u, v, data, sigma).mean().cpu().numpy()

        for key in eval_metrics.keys():
            eval_metrics[key] = np.mean(eval_metrics[key])
            self.eval_metric_history[key].append(eval_metrics[key])
        
        
    
    def test(self):
        test_metrics = {}
        for key in self.metrics.keys():
            test_metrics[key] = {}
            for sigma in self.test_sigma:
                test_metrics[key][sigma] = []
        test_metrics['loss'] = {}
        for sigma in self.test_sigma:
                test_metrics['loss'][sigma] = []
        test_times = []
        batch_data = []
        batch_sigma = []
        for data, sigma in self.loaders['test']:
            batch_data.append(data)
            batch_sigma.append(sigma)
            if len(batch_data) == self.k:
                for sigm in self.test_sigma:
                    data = rearrange(batch_data, 'users batch nrx ntx complex -> batch users nrx ntx complex')
                    sigma = rearrange(batch_sigma, 'users batch -> batch users 1')
                    sigma = torch.ones(sigma.shape, dtype = torch.float32).to(self.gpu) * sigm
                    batch_data = []
                    batch_sigma = []
                    self.best_model.eval()
                    data = data.to(self.gpu)
                    start = time.time()
                    u, v = self.best_model(data, sigma)
                    end = time.time()
                    u = u.detach()
                    v = v.detach()
                    test_times.append(end-start)
                
                    for key in self.metrics.keys():
                        test_metrics[key][sigma].append(self.metrics[key](u, v, data, sigma).mean().cpu().numpy())    
                    test_metrics['loss'][sigma].append(self.loss(u, v, data, sigma).mean().cpu().numpy())
            
        wandb.run.summary['batch_inference_time'] = np.mean(test_times)
        for key in test_metrics.keys():
            metric_list = []
            for sigma in self.test_sigma:
                metric_list.append(np.mean(test_metrics[key][sigma]))
            wandb.run.summary['test_'+key] = metric_list
        wandb.run.summary['test_SNR'] = self.test_snr