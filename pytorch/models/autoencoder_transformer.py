
import torch
from torch.nn import TransformerEncoderLayer, Linear
from .transformer import PositionalEmbedding

from einops import rearrange, repeat

class TransformerMIMOEncoder(torch.nn.Module):
    def __init__(self, ntx, nrx, k, d, nblocks, d_model, nhead, dim_feedforward, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ntx = ntx
        self.nrx = nrx
        self.k = k
        self.d = d
        self.encoders = torch.nn.ModuleList()
        for i in range(nblocks):
            self.encoders.append(
                TransformerEncoderLayer(
                    d_model = d_model,
                    nhead = nhead,
                    dim_feedforward = dim_feedforward,
                    activation = "gelu",
                    batch_first = True
                )
            )
        
        self.input_adapters = torch.nn.ModuleList()
        for i in range(k):
            self.input_adapters.append(Linear(in_features=ntx * nrx * 2, out_features=d_model))
        
        self.output_adapters = torch.nn.ModuleList()
        for i in range(k):
            self.output_adapters.append(Linear(in_features=d_model, out_features=2 * (ntx * d + nrx * d)))
        self.output_activation = torch.nn.Tanh()
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=k)
        self.name = self.__class__.__name__ + '_' + str(self.ntx)+ '_' + str(self.nrx)+ '_' + str(self.k)+ '_' + str(self.d)
        
    def forward(self, x):
        x = rearrange(x, 'b user nrx ntx complex -> b user (nrx ntx complex)')
        temp = []
        for i, adapter in enumerate(self.input_adapters):
            temp.append(adapter(x[:,i,:]))
        x = rearrange(temp, 'users b model_dim -> b users model_dim')
        x = self.positional_emb(x, batch_size = x.shape[0])
        for enc in self.encoders:
            x = enc(x)
            
        temp = []
        for i, adapter in enumerate(self.output_adapters):
            temp.append(adapter(x[:,i,:]))
        x = rearrange(temp, 'users b output_dim -> b users output_dim')
        
        v = rearrange(x[:,:,:2*self.ntx*self.d], 'b users (ntx d complex) -> b users ntx d complex', ntx = self.ntx, d = self.d, complex = 2)
        u = rearrange(x[:,:,2*self.ntx*self.d:], 'b users (nrx d complex) -> b users nrx d complex', nrx = self.nrx, d = self.d, complex = 2)
        
        v_complex = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
        u_complex = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
        
        v_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(v_complex @ torch.transpose(torch.conj(v_complex), -1, -2))).sum(dim=-1))
        # u_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(u_complex @ torch.transpose(torch.conj(u_complex), -1, -2))))
        
        v = v / v_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # u = u / u_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return u, v
    def get_save_name(self):
        return self.name
class TransformerMIMOEncoderNoise(torch.nn.Module):
    def __init__(self, ntx, nrx, k, d, nblocks, d_model, nhead, dim_feedforward, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ntx = ntx
        self.nrx = nrx
        self.k = k
        self.d = d
        self.encoders = torch.nn.ModuleList()
        for i in range(nblocks):
            self.encoders.append(
                TransformerEncoderLayer(
                    d_model = d_model,
                    nhead = nhead,
                    dim_feedforward = dim_feedforward,
                    activation = "gelu",
                    batch_first = True
                )
            )
        
        self.input_adapters = torch.nn.ModuleList()
        for i in range(k):
            self.input_adapters.append(Linear(in_features=ntx * nrx * 2 + 1, out_features=d_model))
        
        self.output_adapters = torch.nn.ModuleList()
        for i in range(k):
            self.output_adapters.append(Linear(in_features=d_model, out_features=2 * (ntx * d + nrx * d)))
        self.output_activation = torch.nn.Tanh()
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=k)
        self.name = self.__class__.__name__ + '_' + str(self.ntx)+ '_' + str(self.nrx)+ '_' + str(self.k)+ '_' + str(self.d)
        
    def forward(self, x, sigma):
        x = rearrange(x, 'b user nrx ntx complex -> b user (nrx ntx complex)')
        sigma = rearrange(sigma, 'b user 1 -> b user 1')
        x = torch.cat([x, sigma], dim = -1)
        temp = []
        for i, adapter in enumerate(self.input_adapters):
            temp.append(adapter(x[:,i,:]))
        x = rearrange(temp, 'users b model_dim -> b users model_dim')
        x = self.positional_emb(x, batch_size = x.shape[0])
        for enc in self.encoders:
            x = enc(x)
            
        temp = []
        for i, adapter in enumerate(self.output_adapters):
            temp.append(adapter(x[:,i,:]))
        x = rearrange(temp, 'users b output_dim -> b users output_dim')
        
        v = rearrange(x[:,:,:2*self.ntx*self.d], 'b users (ntx d complex) -> b users ntx d complex', ntx = self.ntx, d = self.d, complex = 2)
        u = rearrange(x[:,:,2*self.ntx*self.d:], 'b users (nrx d complex) -> b users nrx d complex', nrx = self.nrx, d = self.d, complex = 2)
        
        v_complex = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
        u_complex = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
        
        v_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(v_complex @ torch.transpose(torch.conj(v_complex), -1, -2))).sum(dim=-1))
        # u_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(u_complex @ torch.transpose(torch.conj(u_complex), -1, -2))))
        
        v = v / v_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # u = u / u_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return u, v
    def get_save_name(self):
        return self.name

class CNNMIMOEncoder(torch.nn.Module):
    def __init__(self, ntx, nrx, k, d, nblocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ntx = ntx
        self.nrx = nrx
        self.k = k
        self.d = d
        self.encoders = torch.nn.ModuleList()
        for i in range(nblocks):
            self.encoders.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same'),
                    torch.nn.BatchNorm2d(1),
                    torch.nn.ELU()
                )
            )
        
        self.output_adapter = torch.nn.Linear(in_features=ntx * nrx * k * 2, out_features=2 * (ntx * d + nrx * d) * k)
        self.name = self.__class__.__name__ + '_' + str(self.ntx)+ '_' + str(self.nrx)+ '_' + str(self.k)+ '_' + str(self.d)
        
    def forward(self, x):
        x = rearrange(x, 'b user nrx ntx complex -> b 1 complex (nrx ntx  user)')
        for enc in self.encoders:
            x = enc(x)
        x = rearrange(x, 'batch channel w h -> batch (channel w h)')
        x = self.output_adapter(x)
        v = rearrange(x[:,:2*self.ntx*self.d*self.k], 'b (users ntx d complex) -> b users ntx d complex', users = self.k, ntx = self.ntx, d = self.d, complex = 2)
        u = rearrange(x[:,2*self.ntx*self.d*self.k:], 'b (users nrx d complex) -> b users nrx d complex', users = self.k, nrx = self.nrx, d = self.d, complex = 2)
        
        v_complex = v[:,:,:,:,0] + 1j * v[:,:,:,:,1]
        u_complex = u[:,:,:,:,0] + 1j * u[:,:,:,:,1]
        
        v_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(v_complex @ torch.transpose(torch.conj(v_complex), -1, -2))).sum(dim=-1))
        u_power = torch.sqrt(torch.abs(torch.vmap(torch.vmap(torch.trace))(u_complex @ torch.transpose(torch.conj(u_complex), -1, -2))).sum(dim=-1))
        
        v = v / v_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # u = u / u_power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return u, v
    def get_save_name(self):
        return self.name