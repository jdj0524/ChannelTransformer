import torch
from einops import rearrange, repeat
from torch.nn import TransformerEncoderLayer, Linear

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, sequence_len, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_emb = torch.nn.Parameter(torch.randn(sequence_len, d_model))
        
    def forward(self, x, batch_size):
        positional_emb = repeat(self.positional_emb, '() l emb -> b l emb', b = batch_size)
        x += positional_emb
        return x
        

class ChannelTransformerTransmitter(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = torch.nn.ModuleList
        for i in range(n_blocks):
            self.encoders.append(
                TransformerEncoderLayer(
                    d_model = d_model,
                    nhead = nhead,
                    dim_feedforward = dim_feedforward,
                    activation = "gelu",
                    batch_first = True
                )
            )
            
        self.input_adapters = torch.nn.ModuleList
        for i in range(n_carrier):
            self.input_adapters.append(Linear(in_features=n_tx * n_rx, out_features=d_model))
        self.output_adapter = Linear(in_features=d_model, out_features=dim_feedback)
        self.output_activation = torch.nn.Tanh()
        
        self.cls_token = torch.nn.Parameter(torch.randn(1,1,d_model))
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=n_carrier + 1)
        
    def forward(self, input_tensor):
        batch_size, seq_len, _ = x.shape
        x = rearrange(input_tensor, 'b nrx ntx c -> b c (nrx ntx)')
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_emb(x)
        for i in range(seq_len):
            x[i] = self.output_adapter[i](x[i])
        
        for enc in self.encoders:
            x = enc(x)
        
        x = self.output_adapter(x[0])
        x = self.output_activation(x)
        
        return x
        
        
        
        
        
class ChannelTransformerReceiverSimple(torch.nn.Module):
    def __init__(self, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.linear_1 = Linear(in_features=dim_feedback, out_features=dim_feedback*2)
        self.activation_1 = torch.nn.ELU()
        self.linear_2 = Linear(in_features=dim_feedback*2, out_features=dim_feedback*4)
        self.activation_2 = torch.nn.ELU()
        self.linear_3 = Linear(in_features=dim_feedback*4, out_features=n_tx*n_rx*n_carrier)
        
    def forward(self,x):
        x = self.activation_1(self.linear_1(x))
        x = self.activation_2(self.linear_2(x))
        x = self.linear_3(x)
        
        return x
        