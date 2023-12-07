import torch
from einops import rearrange, repeat
from torch.nn import Linear, TransformerEncoderLayer
from .transformer import PositionalEmbedding
import numpy as np

class ChannelTransformerTransmitterSimple(torch.nn.Module):
    def __init__(self, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
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
        
        return x

class ChannelTransformerTransmitterLarge(torch.nn.Module):
    def __init__(self, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        # self.output_adapter = Linear(in_features=n_tx * n_rx * n_carrier * 2, out_features=dim_feedback)
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.output_activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=32)
        
        self.output_adapter_3 = Linear(in_features=dim_feedback, out_features=dim_feedback)
        self.output_activation_2 = torch.nn.GELU()
        self.output_adapter_4 = Linear(in_features=32, out_features=1)
        
    def forward(self, input_tensor):
        # input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b (c complex nrx ntx)')
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        
        x = self.output_activation(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = self.output_adapter_2(x)
        
        x = rearrange(x, 'b feedback_dim c -> b c feedback_dim')
        x = self.output_adapter_3(x)
        x = self.output_activation_2(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        
        x = torch.squeeze(self.output_adapter_4(x))
        
        x = x / torch.max(torch.abs(x), dim = 1, keepdim = True).values
        
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
            
        self.input_adapter = torch.nn.Sequential(
            Linear(in_features=dim_feedback, out_features=int((dim_feedback + d_model)//2)),
            torch.nn.BatchNorm1d(int((dim_feedback + d_model)//2)),
            torch.nn.GELU(),
            Linear(in_features=int((dim_feedback + d_model)//2), out_features=d_model),
        )
        
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
        
        out = rearrange(out, 'b (nrx ntx complex) ncarrier -> b nrx ntx ncarrier complex', ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        # out = self.output_activation(out)
        
        return out
class QuantizationOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins):
        input = repeat(input, 'b feedback -> b feedback repeat', repeat = bins.shape[0])
        bin_distance = (input - bins).abs()
        indices = torch.argmin(bin_distance, dim = 2)
        return bins[indices]
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
# class PQBOperation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, bins, B):
#         ctx.save_for_backward(input)
#         input = repeat(input, 'b feedback -> b feedback repeat', repeat = bins.shape[0])
#         bin_distance = (input - bins).abs()
#         indices = torch.argmin(bin_distance, dim = 2)
#         ctx.B = B
#         return bins[indices]
#     @staticmethod
#     def backward(ctx, grad_output):
#         B = ctx.B
#         input, = ctx.saved_tensors
#         lbd = 1
#         C = 0.4439938161680786
#         step = 2 ** B
#         a_val = 2

#         # Gradient calculation
#         input_expanded = step * input
#         y = input_expanded - 0.5 - torch.round(input_expanded - 0.5)
#         y = y.clamp(min=0, max=1 / a_val - 1e-25)
#         grad = grad_output * lbd * (1 / C) * a_val * (torch.exp(-1 / (1 - (a_val * y)**2)) - np.exp(-1 / (a_val * 1e-25)**2))

#         # Return the gradient with respect to the input and None for the bins, since we don't compute gradients with respect to the bins
#         return grad, None, None
    
class PQBOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bins, B):
        ctx.save_for_backward(input)
        ctx.B = B

        # Applying the logic from the TensorFlow snippet
        step = 2 ** B
        input_clipped = input.clamp(min=0.5 / step, max=1 - (0.5 / step))
        result = torch.round(input_clipped * step - 0.5)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # The backward method remains the same as in the original PQBOperation
        input, = ctx.saved_tensors
        B = ctx.B
        lbd = 1
        C = 0.4439938161680786
        step = 2 ** B
        a_val = 2

        # Gradient calculation
        input_expanded = step * input
        y = input_expanded - 0.5 - torch.round(input_expanded - 0.5)
        y = y.clamp(min=0, max=1 / a_val - 1e-25)
        grad = grad_output * lbd * (1 / C) * a_val * (torch.exp(-1 / (1 - (a_val * y)**2)) - np.exp(-1 / (a_val * 1e-25)**2))

        return grad, None, None
    
class QuantizationLayer(torch.nn.Module):
    def __init__(self, bit_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bit_level = bit_level
        self.bin_width = 2 / (2**self.bit_level - 1)
        
        self.bins = torch.FloatTensor([-1 + i * self.bin_width for i in range(2**self.bit_level)])
        self.bins = torch.nn.Parameter(self.bins)
        self.bins.requires_grad = False
        
    
    def forward(self, x):
        x = QuantizationOperation.apply(x, self.bins)
        return x
    
class PQBLayer(torch.nn.Module):
    def __init__(self, bit_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bit_level = bit_level
        self.bin_width = 2 / (2**self.bit_level - 1)
        
        self.bins = torch.FloatTensor([-1 + i * self.bin_width for i in range(2**self.bit_level)])
        self.bins = torch.nn.Parameter(self.bins)
        self.bins.requires_grad = False
        
    
    def forward(self, x):
        x = PQBOperation.apply(x, self.bins, self.bit_level)
        return x
        

class ChannelTransformerQuantization(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, bit_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitterSimple(n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.quantization = QuantizationLayer(bit_level=bit_level)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = self.tx_model(x)
        x = self.quantization(x)
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

class ChannelTransformerPQB(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, bit_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitterSimple(n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.quantization = PQBLayer(bit_level=bit_level)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = self.tx_model(x)
        x = self.quantization(x)
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

class ChannelTransformerQuantizationLarge(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, bit_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tx_model = ChannelTransformerTransmitterLarge(n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.quantization = QuantizationLayer(bit_level=bit_level)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor):
        x = self.tx_model(x)
        x = self.quantization(x)
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name
