import torch
from pytorch.models.transformer import ChannelTransformerBase, ChannelTransformer, ChannelTransformerSimple
from pytorch.models.csinet import CSINet, ChannelAttention
from pytorch.loss.nmse import NMSE_loss, Cosine_distance, MSE_loss
from pytorch.dataloader.dataloader import DeepMIMODataset, DeepMIMOSampleDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_optimizer import Ranger21

import numpy as np
from sklearn import preprocessing

from fvcore.nn import flop_count, FlopCountAnalysis
# from distributed.distributed import setup_ddp, cleanup

# torch.utils.data.random_split(data, lengths = [0.7, 0.1, 0.2])
gpu = 0

print("O1_140")

# data = DeepMIMOSampleDataset(files_dir = '/home/jdj0524/DeepMIMO_Datasets/I2_28B/samples/')
data = DeepMIMOSampleDataset(files_dir = '/home/jdj0524/DeepMIMO_Datasets/O1_140/samples/')
# data = DeepMIMOSampleDataset(files_dir = '/home/jdj0524/DeepMIMO_Datasets/samples/', target_file = 'o1_channels_grid_1.npy')

data, _ = torch.utils.data.random_split(data, lengths = [1, 0])

print("data length : {}".format(len(data)))

train, test, validation = torch.utils.data.random_split(data, lengths = [0.6, 0.2, 0.2])
batch_size = 128
epochs = 100

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=4)

print("train set length : {}".format(len(train_loader)))
print("test set length : {}".format(len(test_loader)))
print("val set length : {}".format(len(val_loader)))

# model = ChannelAttention(
#     no_blocks=3, n_tx=16, n_rx=16, in_channels=128,dim_feedback=32
# ) # Model flops : 173 MFLOPs
# model = CSINet(
#     no_blocks=4, n_tx=16, n_rx=16, in_channels=128,dim_feedback=128
# )

# model = ChannelTransformer(
#     n_blocks=5, d_model=256, nhead=8, dim_feedforward=1024, n_tx=16, n_rx=16, n_carrier=64,dim_feedback=128
# ) # Model flops : 330 MFLOPs

model = ChannelTransformerSimple(
    n_blocks=3, d_model=256, nhead=4, dim_feedforward=1024, n_tx=16, n_rx=16, n_carrier=128,dim_feedback=32
) # Model flops : 190 MFLOPs

# flops, params = get_model_complexity_info(model.to(gpu), (16,16,64,2))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))

print(flop_count(model, torch.rand(64, 16,16,128,2)))

# print(flops.total())

# flops, params = count_ops(model, torch.rand(1,16,16,64,2))

# print("FLOPS : {}".format(flops))
# print("PARAMS : {}".format(params))

loss = MSELoss()


optimizer = AdamW(params = model.parameters(), lr = 0.0005)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True, factor = 0.5, patience = 10)
# optimizer = Ranger21(params = model.parameters(), lr = 0.001, num_iterations=len(train_loader) * epochs)


model.to(gpu)
for i in range(epochs):
    train_mse = []
    for data in train_loader:
        model.train()
        data = data.to(gpu)
        output = model(data)
        mse = MSE_loss(output, data).mean()
        train_mse.append(mse.detach().cpu())
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
    
    cosine = []
    mse = []
    nmse = []
    for data in val_loader:
        model.eval()
        data = data.to(gpu)
        output = model(data).detach()
        mse.append(MSE_loss(output, data).mean().cpu())
        nmse.append(NMSE_loss(output, data).mean().cpu())
        cosine.append(Cosine_distance(output, data).mean().cpu())
    print("Epoch : {}, cosine : {}, MSE : {}, train MSE : {}, NMSE : {}, current LR : {}".format(i, np.mean(cosine), np.mean(mse), np.mean(train_mse), np.mean(nmse), optimizer.param_groups[-1]['lr']))
    scheduler.step(np.mean(nmse))



torch.save(model.state_dict(), '/home/jdj0524/projects/ChannelTransformer/checkpoints/channeltransformer_feedback32.pth')
# torch.save(model.state_dict(), '/home/jdj0524/projects/ChannelTransformer/checkpoints/channelattention_feedback32.pth')

            
            