import numpy as np
import cvxpy as cvx
import torch
from sklearn.linear_model import Lasso
from einops import rearrange

class Lasso_Compressive(torch.nn.Module):
    def __init__(self, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        self.n_feedback = dim_feedback
        
    def forward(self, x):
        num_batch = x.shape[0]
        batch_outs = []
        for i in range(num_batch):
            print("starting run")
            temp = x[i].cpu().numpy()
            temp = rearrange(temp, 'nrx ntx channel complex -> (nrx ntx channel complex)')
            coding_mat = np.random.randn(self.n_feedback, self.n_tx * self.n_rx * self.n_carrier * 2)
            y = np.matmul(coding_mat, temp.T)
            lasso = Lasso(alpha = 0.5)
            lasso.fit(coding_mat, y)
            recovered_tensor = torch.from_numpy(np.asarray(lasso.coef_))
            recovered_tensor = rearrange(recovered_tensor, '(nrx ntx channel complex) -> nrx ntx channel complex', nrx = self.n_rx, ntx = self.n_tx, channel = self.n_carrier, complex = 2)
            batch_outs.append(recovered_tensor)
        out = rearrange(batch_outs, 'b nrx ntx channel complex -> b nrx ntx channel complex')
        return out
    
class OMP_Compressive(torch.nn.Module):
    def __init__(self, ntx, nrx, d, k, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d = d
        self.k = k