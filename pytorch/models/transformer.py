import torch
from einops import rearrange, repeat
from torch.nn import TransformerEncoderLayer, Linear

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, sequence_len, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_emb = torch.nn.Parameter(torch.randn(sequence_len, d_model))
        
    def forward(self, x, batch_size):
        positional_emb = repeat(self.positional_emb, 'seq emb -> b seq emb', b = batch_size)
        x += positional_emb
        return x

class ChannelTransformerMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features, (in_features + out_features) // 2),
            torch.nn.BatchNorm1d((in_features + out_features) // 2),
            torch.nn.GELU(),
            torch.nn.Linear((in_features + out_features) // 2, out_features)
            
            # torch.nn.Linear(in_features, out_features)
        )
    def forward(self, x):
        
        return self.sequential(x)

class ChannelTransformerTransmitter(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = torch.nn.ModuleList()
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
            
        self.input_adapters = torch.nn.ModuleList()
        for i in range(n_carrier * 2):
            self.input_adapters.append(ChannelTransformerMLP(in_features=n_tx * n_rx, out_features=d_model))
        self.output_adapter = Linear(in_features=d_model, out_features=dim_feedback)
        self.output_activation = torch.nn.Tanh()
        
        self.cls_token = torch.nn.Parameter(torch.randn(1,1,d_model))
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=n_carrier * 2 + 1)
        
    def forward(self, input_tensor):
        input_tensor = rearrange(input_tensor, 'b nrx ntx c -> b c (nrx ntx)')
        batch_size, seq_len, _ = input_tensor.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = batch_size)
        x = []
        for i in range(seq_len):
            x.append(self.input_adapters[i](input_tensor[:,i,:]))
        x = torch.stack(x, dim = 1)
        
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_emb(x, batch_size = batch_size)
        
        for enc in self.encoders:
            x = enc(x)
        
        x = self.output_adapter(x[:,0,:])
        x = self.output_activation(x)
        
        return x

class ChannelTransformerTransmitterSimple(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        # self.output_adapter = Linear(in_features=n_tx * n_rx * n_carrier * 2, out_features=dim_feedback)
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.output_activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=1)
        
        
    def forward(self, input_tensor):
        # input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b (c complex nrx ntx)')
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        
        # x = x.mean(dim = 1)
        x = self.output_activation(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = torch.squeeze(self.output_adapter_2(x))
        
        x = x / torch.max(torch.abs(x), dim = 1, keepdim = True).values
        
        
        
        # x = self.output_activation(x)
        
        return x
        

class ChannelTransformerReceiver(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = torch.nn.ModuleList()
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
            
        self.input_adapter = Linear(in_features=dim_feedback, out_features=d_model)
        self.output_adapters = torch.nn.ModuleList()
        for i in range(n_carrier):
            self.output_adapters.append(Linear(in_features=d_model, out_features=n_tx * n_rx * 2))
        self.output_activation = torch.nn.Tanh()
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=n_carrier)
        
    def forward(self, input_tensor):
        input_tensor = self.input_adapter(input_tensor)
        
        batch_size, _ = input_tensor.shape
        input_tensor = repeat(input_tensor, 'b e -> b n e', n = self.n_carrier).clone()
        x = self.positional_emb(input_tensor, batch_size = batch_size)
        
        for enc in self.encoders:
            x = enc(x)
        
        out = []
        
        for i in range(self.n_carrier):
            out.append(self.output_adapters[i](x[:,i,:]))
        out = torch.stack(out, dim = 2) 
        
        out = rearrange(out, 'b (ntx nrx complex) ncarrier -> b nrx ntx ncarrier complex', ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        # out = self.output_activation(out)
        
        return out

class ChannelTransformerTransmitterSimpleV2(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        # self.output_adapter = Linear(in_features=n_tx * n_rx * n_carrier * 2, out_features=dim_feedback)
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=1)
        
        
    def forward(self, input_tensor):
        # input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b (c complex nrx ntx)')
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        x = self.activation(x)
        # x = x.mean(dim = 1)
        
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = torch.squeeze(self.output_adapter_2(x))
        
        x = x / torch.max(torch.abs(x), dim = 1, keepdim = True).values
        
        
        
        # x = self.output_activation(x)
        
        return x
        

class ChannelTransformerReceiverV2(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = torch.nn.ModuleList()
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
            
        self.input_adapter = Linear(in_features=dim_feedback, out_features=d_model)
        self.output_adapter = Linear(in_features=d_model, out_features=n_tx * n_rx * 2)
        self.output_activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=n_carrier, out_features=n_carrier)
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=n_carrier)
        
    def forward(self, input_tensor):
        input_tensor = self.input_adapter(input_tensor)
        
        batch_size, _ = input_tensor.shape
        input_tensor = repeat(input_tensor, 'b e -> b n e', n = self.n_carrier).clone()
        x = self.positional_emb(input_tensor, batch_size = batch_size)
        
        for enc in self.encoders:
            x = enc(x)
        
        # x = rearrange(x, 'b model_dim ncarrier -> b ncarrier model_dim')
        x = self.output_adapter(x)
        x = self.output_activation(x)
        x = rearrange(x, 'b ncarrier dout -> b dout ncarrier')
        x = self.output_adapter_2(x)
        
        out = rearrange(x, 'b (ntx nrx complex) ncarrier -> b nrx ntx ncarrier complex', ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        # out = self.output_activation(out)
        
        return out
                
        
        
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
        self.linear_3 = Linear(in_features=dim_feedback*4, out_features=n_tx*n_rx*n_carrier*2)
        self.output_activation = torch.nn.Tanh()
        
    def forward(self,x):
        x = self.activation_1(self.linear_1(x))
        x = self.activation_2(self.linear_2(x))
        x = self.linear_3(x)
        x = rearrange(x, "b (ntx nrx ncarrier complex) -> b nrx ntx ncarrier complex", ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        x = self.output_activation(x)
        
        return x
    
class ChannelTransformerBase(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitter(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelTransformerReceiverSimple(n_tx=n_tx, n_rx=n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        
    def forward(self, x: torch.Tensor):
        channel_dim = x.shape[-1]
        x = torch.cat([x.real, x.imag], dim = 3)
        x = self.tx_model(x)
        x = self.rx_model(x)
        x = torch.complex(x[:,:,:,:channel_dim], x[:,:,:,channel_dim:])
        return x
        
class ChannelTransformer(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitter(n_blocks=3, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b ntx nrx c complex -> b ntx nrx (c complex)')
        x = self.tx_model(x)
        x = self.rx_model(x)
        return x

class ChannelTransformerSimple(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitterSimple(n_blocks=3, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = self.tx_model(x)
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

class ChannelTransformerSimpleV2(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitterSimpleV2(n_blocks=3, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelTransformerReceiverV2(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = self.tx_model(x)
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name
        