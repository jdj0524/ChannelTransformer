from .trainer import Trainer
import torch
import numpy as np
import wandb
from copy import deepcopy
import time
from .base_trainer import BaseTrainer

class SVDTrainer(BaseTrainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        self.train_snr = options['train_snr']
        self.train_sigma = self.convert_snr(self.train_snr)
        self.test_snr = options['test_snr']
        self.test_sigma = []
        for snr in self.test_snr:
            self.test_sigma.append(self.convert_snr(snr))
    def convert_snr(self, snr_db):
        return 1/10**(snr_db / 10)
    def train_step(self, epoch):
        step_losses = []
        #self.loaders['train'].dataset.reshuffle_users()
        for data in self.loaders['train']:
            self.model.train()
            data = data.to(self.gpu)
            u, v = self.model(data)
            step_loss = self.loss(u, v, data, self.train_sigma).mean()
            step_losses.append(step_loss.detach().cpu().numpy())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
        
    def save_best_model(self, cur_loss):
        if self.best_loss is None or self.best_loss > cur_loss:
            self.best_loss = cur_loss
            self.best_model = deepcopy(self.model)
            torch.save(self.best_model.state_dict(), self.options['save_dir'] + self.best_model.get_save_name())

    
    def train(self):
        self.model = self.model.to(self.gpu)
        for i in range(self.epochs):
            self.train_step(i)
            self.eval()
            self.save_best_model(self.eval_metric_history['val_loss'][-1])
            self.print_eval_metrics(i)
        self.test()
    
    def eval(self):
        eval_metrics = {}
        for key in self.metrics.keys():
            eval_metrics[key] = []
        eval_metrics['val_loss'] = []

        for data in self.loaders['val']:
            self.model.eval()
            data = data.to(self.gpu)
            u, v = self.model(data)
            u = u.detach()
            v = v.detach()
            for key in self.metrics.keys():
                eval_metrics[key].append(self.metrics[key](u, v, data, self.train_sigma).mean().cpu().numpy())
            eval_metrics['val_loss'] = self.loss(u, v, data, self.train_sigma).mean().cpu().numpy()

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
        for data in self.loaders['test']:
            self.best_model.eval()
            data = data.to(self.gpu)
            start = time.time()
            u, v = self.best_model(data)
            end = time.time()
            u = u.detach()
            v = v.detach()
            test_times.append(end-start)
            for sigma in self.test_sigma:
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