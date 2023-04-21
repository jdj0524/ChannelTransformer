import torch
from models.transformer import ChannelTransformerBase, ChannelTransformer
from loss.nmse import NMSE_loss, Cosine_distance, MSE_loss
from dataloader.dataloader import DeepMIMODataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from sklearn import preprocessing

data = DeepMIMODataset(target_file = 'o1_channels_orig_grid_1.npy')


train, test, validation = torch.utils.data.random_split(data, lengths = [0.7, 0.1, 0.2])

batch_size = 32
epochs = 1000

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

model = ChannelTransformer(
    n_blocks=3, d_model=32, nhead=8, dim_feedforward=16, n_tx=32, n_rx=1, n_carrier=256,dim_feedback=128
)
loss = MSELoss()

optimizer = Adam(params = model.parameters(), lr = 0.001)
model.to('cuda:0')
for i in range(epochs):
    for data in train_loader:
        model.train()
        data = data.to('cuda:0')
        output = model(data)
        mse = MSE_loss(output, data)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
    
    cosine = []
    mse = []
    nmse = []
    for data in val_loader:
        model.eval()
        data = data.to("cuda:0")
        output = model(data).detach()
        mse.append(MSE_loss(output, data).cpu())
        nmse.append(NMSE_loss(output, data).cpu())
        cosine.append(Cosine_distance(output, data).cpu())
    print("Epoch : {}, cosine : {}, MSE : {}, NMSE : {}".format(i, np.mean(cosine), np.mean(mse), np.mean(nmse)))

        
        