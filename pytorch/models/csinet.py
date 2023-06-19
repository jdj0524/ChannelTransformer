import torch
from einops import rearrange

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs ) -> None:
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(in_channels * 2),
            torch.nn.GELU()
        )
        
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(in_channels * 4),
            torch.nn.GELU()
        )
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=(3,3), padding='same'),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.skip_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), padding='same')
        self.out_activation = torch.nn.GELU()
    def forward(self, x):
        skip = self.skip_conv(x)
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
        self.softmax = torch.nn.Softmax(dim=(1))
    
    def forward(self, x):
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)
        attention_map = torch.matmul(torch.transpose(key, -1, -2), query)
        _, channel, dim1, dim2 = attention_map.shape
        attention_map = rearrange(attention_map, 'b channel dim1 dim2 -> b (channel dim1 dim2)')
        attention_map = self.softmax(attention_map)
        attention_map = rearrange(attention_map, 'b (channel dim1 dim2) -> b channel dim1 dim2', channel = channel, dim1 = dim1, dim2 = dim2)
        output = torch.matmul(value, attention_map)
        output = self.out_conv(output)
        output = output + x
        return output
        
class CSINet(torch.nn.Module):
    def __init__(self, dim_feedback, n_tx, n_rx, n_carrier, no_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.c = n_carrier
        self.input_conv = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3,3), padding = 'same')
        self.input_dense = torch.nn.Linear(in_features=n_tx*n_rx*n_carrier*2, out_features=dim_feedback)
        self.output_dense = torch.nn.Linear(in_features=dim_feedback, out_features=n_tx*n_rx*n_carrier*2)
        self.resblocks = torch.nn.ModuleList()
        self.out_conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), padding='same')
        self.out_conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), padding='same')
        self.out_activation = torch.nn.Tanh()
        # for i in range(no_blocks):
        self.resblocks.append(ResBlock(in_channels=2, out_channels=2))
        self.resblocks.append(ResBlock(in_channels=2, out_channels=4))
        self.resblocks.append(ResBlock(in_channels=4, out_channels=4))
        # self.resblocks.append(ResBlock(in_channels=2, out_channels=2))
        # self.resblocks.append(ResBlock(in_channels=2, out_channels=2))
    def get_save_name(self):
        return self.name
    def forward(self,x):
        x = rearrange(x, 'b ntx nrx c complex -> b complex c (ntx nrx) ')
        x = self.input_conv(x)
        x = rearrange(x, 'b complex c nant -> b (complex c nant)')
        x = self.input_dense(x)
        x = self.output_dense(x)
        x = rearrange(x, 'b (complex c nant) -> b complex c nant', nant = self.n_tx * self.n_rx, c = self.c, complex = 2)
        for block in self.resblocks:
            x = block(x)

        
        real = self.out_conv_1(x)
        imag = self.out_conv_2(x)
        # x = rearrange(x, 'b complex c (ntx nrx) -> b ntx nrx c complex', ntx = self.n_tx, nrx = self.n_rx, c = self.c, complex = 2)
        x = rearrange([real, imag], 'complex b 1 c (ntx nrx) -> b ntx nrx c complex', ntx = self.n_tx, nrx = self.n_rx, c = self.c, complex = 2)
        
        # x = self.out_activation(x)
        return x
        
class ChannelAttention(torch.nn.Module):
    def __init__(self, dim_feedback, n_tx, n_rx, n_carrier, no_blocks, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.c = n_carrier
        self.input_conv = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3,3), padding = 'same')
        self.input_dense = torch.nn.Linear(in_features=n_tx*n_rx*n_carrier*2, out_features=dim_feedback)
        self.output_dense = torch.nn.Linear(in_features=dim_feedback, out_features=n_tx*n_rx*n_carrier*2)
        self.input_block_1 = ResBlock(in_channels=2, out_channels=4)
        self.input_block_2 = ResBlock(in_channels=4, out_channels=4)
        self.input_block_3 = ResBlock(in_channels=4, out_channels=4)
        # self.input_block_4 = ResBlock(in_channels=2, out_channels=2)
        # self.input_block_5 = ResBlock(in_channels=2, out_channels=2)
        # self.input_block_6 = ResBlock(in_channels=2, out_channels=2)
        # self.input_block_7 = ResBlock(in_channels=2, out_channels=2)
        # self.input_block_4 = ResBlock(in_channels=32, out_channels=2)
        self.out_conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), padding='same')
        self.out_conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), padding='same')
        
        self.attention_1 = SelfAttention2D(channels = 4)
        self.attention_2 = SelfAttention2D(channels = 4)
        
        self.in_activation = torch.nn.Tanh()
    def get_save_name(self):
        return self.name
    def forward(self,x):
        x = rearrange(x, 'b ntx nrx c complex -> b complex c (ntx nrx) ')
        # x = self.input_conv(x)
        x = rearrange(x, 'b complex c nant -> b (complex c nant)')
        x = self.input_dense(x)
        x = x / torch.max(torch.abs(x), dim = 1, keepdim = True).values
        x = self.output_dense(x)
        x = rearrange(x, 'b (complex c nant) -> b complex c nant', nant = self.n_tx * self.n_rx, c = self.c, complex = 2)
        x = self.input_block_1(x)
        x = self.attention_1(x)
        x = self.input_block_2(x)
        x = self.attention_2(x)
        x = self.input_block_3(x)
        
        real = self.out_conv_1(x)
        imag = self.out_conv_2(x)
        
        x = torch.cat([real,imag], dim=1)
        # x = self.input_block_4(x)
        x = rearrange(x, 'b complex c (ntx nrx) -> b ntx nrx c complex', ntx = self.n_tx, nrx = self.n_rx, c = self.c, complex = 2)
        # x = self.out_activation(x)
        return x
        
        