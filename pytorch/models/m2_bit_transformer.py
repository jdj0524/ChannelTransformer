import torch
from einops import rearrange, repeat
from torch.nn import Linear, TransformerEncoderLayer
from torch.nn.functional import gumbel_softmax
from .transformer import PositionalEmbedding
from .mm_mixer import MonarchMixerLayer
from .hyena_mm_mixer_layer import M2BertEncoder

class GumbelSoftmaxBitRelaxation(torch.nn.Module):
    def __init__(self, temperature = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def forward(self, x):
        x = gumbel_softmax(x, tau=self.temperature, hard=not(self.training), dim=-1)
        return x

class ChannelM2MixerBitLevel(torch.nn.Module):
    def __init__(self, config_tx, config_rx, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_feedback = dim_feedback
        self.tx_model = ChannelMLPMixerTransmitterBitLevel(config = config_tx, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelM2MixerReceiverBitLevel(config = config_rx, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor, no_bits: int):
        mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device)
        neg_mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device) * -1
        
        for i in range(len(no_bits)):
            mask[i, no_bits[i]:] = 0 
            neg_mask[i, :no_bits[i]] = 0
        
        x = self.tx_model(x, mask)
        x = x * mask
        x = x + neg_mask
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

    def set_temperature(self, temperatrue):
        self.tx_model.set_temperature(temperatrue)
        
class ChannelTXRXM2MixerBitLevel(torch.nn.Module):
    def __init__(self, config_tx, config_rx, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_feedback = dim_feedback
        self.tx_model = ChannelM2MixerTransmitterBitLevel(config = config_tx, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelM2MixerReceiverBitLevel(config = config_rx, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        
    def forward(self, x: torch.Tensor, no_bits: int):
        mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device)
        neg_mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device) * -1
        
        for i in range(len(no_bits)):
            mask[i, no_bits[i]:] = 0 
            neg_mask[i, :no_bits[i]] = 0
        
        x = self.tx_model(x, mask)
        x = x * mask
        x = x + neg_mask
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

    def set_temperature(self, temperatrue):
        self.tx_model.set_temperature(temperatrue)
        
class ChannelMLPMixerTransmitterBitLevel(torch.nn.Module):
    def __init__(self, config, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.output_activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=32)
        
        self.output_adapter_3 = Linear(in_features=dim_feedback, out_features=dim_feedback)
        self.output_activation_2 = torch.nn.GELU()
        self.output_adapter_4 = Linear(in_features=32, out_features=2)
        
        self.bit_quantizer = GumbelSoftmaxBitRelaxation(temperature=5)

        self.mask_processor = torch.nn.Sequential(
            Linear(in_features=dim_feedback, out_features=64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.GELU(),
            Linear(in_features=64, out_features=32),
        )
        
    def set_temperature(self, temperature):
        self.bit_quantizer.set_temperature(temperature)
    
    def forward(self, input_tensor, mask):
        
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        
        # x = x.mean(dim = 1)
        x = self.output_activation(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = self.output_adapter_2(x)
        
        processed_mask = self.mask_processor(mask)
        processed_mask = rearrange(processed_mask, 'batch mask_dim -> batch 1 mask_dim')
        x = x + processed_mask
        
        x = rearrange(x, 'b feedback_dim c -> b c feedback_dim')
        x = self.output_adapter_3(x)
        x = self.output_activation_2(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = self.output_adapter_4(x)
        x = self.bit_quantizer(x)
        x = x[:,:,0]
        return x        

class ChannelM2MixerTransmitterBitLevel(torch.nn.Module):
    def __init__(self, config, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.output_activation = torch.nn.GELU()
        
        self.output_encoder = M2BertEncoder(config)
        
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=2)
        
        self.mask_processor = torch.nn.Sequential(
            Linear(in_features=dim_feedback, out_features=dim_feedback),
            torch.nn.BatchNorm1d(num_features=dim_feedback),
            torch.nn.GELU(),
            Linear(in_features=dim_feedback, out_features=dim_feedback),
        )
        
        self.bit_quantizer = GumbelSoftmaxBitRelaxation(temperature=5)
        
    def set_temperature(self, temperature):
        self.bit_quantizer.set_temperature(temperature)
    
    def forward(self, input_tensor, mask):
        
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        mask = self.mask_processor(mask)
        x = x + mask
        x = self.output_encoder(x, None)
        
        x = rearrange(x, "enc b channel feedback -> enc b feedback channel")
        x = torch.mean(x, dim = 0)
        x = self.output_adapter_2(x)
        x = self.bit_quantizer(x)
        x = x[:, :, 0]
        
        return x        
    
class ChannelM2MixerReceiverBitLevel(torch.nn.Module):
    def __init__(self, config, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = M2BertEncoder(config)
               
        self.input_adapter = torch.nn.Sequential(
            Linear(in_features=dim_feedback, out_features=int((dim_feedback + config["hidden_size"])//2)),
            torch.nn.BatchNorm1d(int((dim_feedback + config["hidden_size"])//2)),
            torch.nn.GELU(),
            Linear(in_features=int((dim_feedback + config["hidden_size"])//2), out_features=config["hidden_size"]),
        )
        self.output_adapters = torch.nn.ModuleList()
        for i in range(n_carrier):
            self.output_adapters.append(Linear(in_features=config["hidden_size"], out_features=n_tx * n_rx * 2))
        
    def forward(self, input_tensor):
        input_tensor = self.input_adapter(input_tensor)
        
        input_tensor = repeat(input_tensor, 'b e -> b n e', n = self.n_carrier).clone()
        
        x = self.encoders(input_tensor, None)
        
        x = rearrange(x, "enc b channel txrx -> enc b channel txrx")
        x = torch.mean(x, dim = 0)
        
        # x = x[-1]
        
        out = []
        
        for i in range(self.n_carrier):
            out.append(self.output_adapters[i](x[:,i,:]))
        out = torch.stack(out, dim = 2) 
        
        out = rearrange(out, 'b (nrx ntx complex) ncarrier -> b nrx ntx ncarrier complex', ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        
        return out
