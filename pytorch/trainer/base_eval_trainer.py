from .base_trainer import BaseTrainer
import numpy as np
import time
import wandb

class BaseEvalTrainer(BaseTrainer):
    def train(self):
        self.best_model = self.model
        self.test()
        
    def build_optimizer(self):
        pass
        
    def save_best_model(self, cur_loss):
        pass
    
    def test(self):
        test_metrics = {}
        for key in self.metrics.keys():
            test_metrics[key] = []
        test_metrics['loss'] = []
        test_times = []
        for data in self.loaders['test']:
            self.best_model.eval()
            start = time.time()
            output = self.best_model(data)
            end = time.time()
            output = output.detach().cpu()
            test_times.append(end-start)
            for key in self.metrics.keys():
                test_metrics[key].append(self.metrics[key](output, data).mean().cpu().float().numpy())
            test_metrics['loss'] = self.loss(output, data).mean().cpu().float().numpy()
            break
        
        wandb.run.summary['batch_inference_time'] = np.mean(test_times)
        for key in test_metrics.keys():
            wandb.run.summary['test_'+key] = np.mean(test_metrics[key])