import torch
from einops import rearrange, repeat
from ..loss.mimo_rate import hermitian
from functools import partial
class MMSE_Transciever(torch.nn.Module):
    def __init__(self, ntx, nrx, d, k, iterations = 10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.k = k
        self.d = d
        self.nrx = nrx
        self.ntx = ntx
        self.eigen = None
    def forward(self, x, sigma = 1e-2):
        x = x[:,:,:,:,0] + 1j * x[:,:,:,:,1]
        v = torch.rand(x.shape[0], self.k, self.ntx, self.d, dtype=torch.cfloat).to(x.get_device())
        u = torch.rand(x.shape[0], self.k, self.nrx, self.d, dtype=torch.cfloat).to(x.get_device())
        lam = torch.tensor(1.0).to(x.get_device())
        for i in range(self.iterations):
            v_inside = (hermitian(x, -1, -2) @ u @ hermitian(u, -1, -2) @ x).sum(dim = 1) + lam * torch.eye(self.ntx).to(x.get_device())
            v_inside = rearrange(v_inside, 'batch nrx ntx -> batch 1 nrx ntx')
            v = torch.linalg.solve(v_inside, hermitian(x, -1, -2) @ u)
            
            u_channel = repeat(x, 'batch user nrx d -> batch user broad nrx d', broad = self.k)
            u_channel_h = hermitian(u_channel, -1, -2)
            u_inner = v @ hermitian(v, -1, -2)
            u_inner = repeat(u_inner, 'batch user nrx d -> batch broad user nrx d', broad = self.k)
            u_sum = (u_channel @ u_inner @ u_channel_h).sum(dim=2) + sigma * torch.eye(self.nrx).to(x.get_device())
            u = torch.linalg.solve(u_sum, x @ v)
            
            _, self.eigen, _ = torch.linalg.svd((hermitian(x, -1, -2) @ u @ hermitian(u, -1, -2) @ x).sum(dim = 1))
            
            lam = self.newton(self.lambda_func, lam, threshold = 1e-7)
        
        u = torch.stack([u.real, u.imag], dim = -1)
        v = torch.stack([v.real, v.imag], dim = -1)
        
        return u, v
                
    def lambda_func(self, lam_guess):
        temp_eigen = torch.diagonal(self.eigen, dim1 = -1, dim2 = -2)
        temp_eigen = (temp_eigen / (temp_eigen + lam_guess)).sum(dim=-1)
        return temp_eigen - 1
        
    def newton(self, func, guess, threshold = 1e-7):
        guess = torch.autograd.Variable(guess, requires_grad=True)
        value = func(guess)
        iteration = 0
        while abs(value.mean()) > threshold:
            iteration = iteration + 1
            if iteration > 50:
                break
            value = func(guess)
            value.backward()
            guess.data -= (value / guess.grad).data
            guess.grad.data.zero_()
        return guess.data
    

class Collaboration_Transciever(torch.nn.Module):
    def __init__(self, ntx, nrx, d, k, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d = d
        self.k = k
        
    def forward(self, x):
        x = rearrange(x, 'b user nrx ntx complex -> b (user nrx) ntx complex')
        x = x[:,:,:,0] + 1j * x[:,:,:,1]
        u, s, vh = torch.linalg.svd(x)
        v = torch.transpose(torch.conj(vh), dim0=-1, dim1=-2)
        u = u[:,:,:self.k * self.d]
        v = v[:,:,:self.k * self.d]
        u = torch.stack([u.real, u.imag], dim = -1)
        v = torch.stack([v.real, v.imag], dim = -1)
        
        u = rearrange(u, 'b dim1 (user d) complex -> b 1 dim1 (user d) complex', user = self.k, d = self.d)
        v = rearrange(v, 'b dim1 (user d) complex -> b 1 dim1 (user d) complex', user = self.k, d = self.d)
        
        return u, v
