import torch
from einops import rearrange

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs ) -> None:
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.LeakyReLU()
        )
        
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(in_channels * 4),
            torch.nn.LeakyReLU()
        )
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.out_activation = torch.nn.LeakyReLU()
    def forward(self, x):
        skip = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) + skip
        x = self.out_activation(x)
        return x

class SelfAttention2D(torch.nn.Module):
    def __init__(self, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key_conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1))
        self.value_conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1))
        self.query_conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1))
        self.out_conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1))
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, x):
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)
        
        attention_map = self.softmax(torch.matmul(torch.transpose(key, -1, -2), query))
        output = torch.matmul(value, attention_map)
        output = self.out_conv(output)
        output = output + x
        return output
        
class CSINet(torch.nn.Module):
    def __init__(self, dim_feedback, n_tx, n_rx, in_channels, no_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.c = in_channels * 2
        self.input_conv = torch.nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=(3,3), padding = 'same')
        self.input_dense = torch.nn.Linear(in_features=n_tx*n_rx*in_channels*2, out_features=dim_feedback)
        self.output_dense = torch.nn.Linear(in_features=dim_feedback, out_features=n_tx*n_rx*in_channels*2)
        self.resblocks = torch.nn.ModuleList()
        self.out_activation = torch.nn.Tanh()
        for i in range(no_blocks):
            self.resblocks.append(ResBlock(in_channels=in_channels*2, out_channels=in_channels*2))
    
    def forward(self,x):
        x = rearrange(x, 'b ntx nrx c -> b c ntx nrx ')
        x = self.input_conv(x)
        x = rearrange(x, 'b ntx nrx c -> b (ntx nrx c)')
        x = self.input_dense(x)
        x = self.output_dense(x)
        x = rearrange(x, 'b (ntx nrx c) -> b c ntx nrx', ntx = self.n_tx, nrx = self.n_rx, c = self.c)
        for block in self.resblocks:
            x = block(x)
        x = rearrange(x, 'b c ntx nrx -> b ntx nrx c')
        x = self.out_activation(x)
        return x
        
class ChannelAttention(torch.nn.Module):
    def __init__(self, dim_feedback, n_tx, n_rx, in_channels, no_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.c = in_channels * 2
        self.input_conv = torch.nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=(3,3), padding = 'same')
        self.input_dense = torch.nn.Linear(in_features=n_tx*n_rx*in_channels*2, out_features=dim_feedback)
        self.output_dense = torch.nn.Linear(in_features=dim_feedback, out_features=n_tx*n_rx*in_channels*2)
        self.resblocks = torch.nn.ModuleList()
        for i in range(no_blocks):
            self.resblocks.append(ResBlock(in_channels=in_channels*2, out_channels=in_channels*2))
        
        self.attention = torch.nn.ModuleList()
        for i in range(2):
            self.attention.append(SelfAttention2D(channels = in_channels * 2))
        self.out_activation = torch.nn.Tanh()
    def forward(self,x):
        x = rearrange(x, 'b ntx nrx c -> b c ntx nrx ')
        x = self.input_conv(x)
        x = rearrange(x, 'b ntx nrx c -> b (ntx nrx c)')
        x = self.input_dense(x)
        x = self.output_dense(x)
        x = rearrange(x, 'b (ntx nrx c) -> b c ntx nrx', ntx = self.n_tx, nrx = self.n_rx, c = self.c)
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if i == 0 or i == 3:
                x = self.attention[i%2](x)
        
                
        x = rearrange(x, 'b c ntx nrx -> b ntx nrx c')
        x = self.out_activation(x)
        return x
        
        