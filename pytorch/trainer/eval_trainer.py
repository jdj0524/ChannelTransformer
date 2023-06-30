import torch
import numpy as np
from .trainer import Trainer
import time
import wandb

class DummyTrainer(Trainer):
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
        pass
    
    def build_optimizer(self):
        pass
        
    def save_best_model(self, cur_loss):
        pass

    def train(self):
        self.test()
    
    def eval(self):
        pass
        
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
            data = data.to(self.gpu)
            start = time.time()
            u, v = self.model(data)
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
        print(wandb.run.summary['test_loss'])
        print(wandb.run.summary['test_channel_capacity'])