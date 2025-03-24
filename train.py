import glob
from data_loader import SRDataset
from torch.utils.data import DataLoader
import torch
from model import SR
from torch.optim import Adam
import torch.nn as nn

tra_img_name_list = glob.glob('dataset/*.jpg')
train_dataset = SRDataset(tra_img_name_list)

batch_size = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain_dir = None
# pretrain_dir = 'model.pth'

model = SR()

if pretrain_dir is not None:
    model.load_state_dict(torch.load(pretrain_dir))
    
model = model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x_lr, x_hr):
    return nn.functional.binary_cross_entropy(x_lr, x_hr)

def train(model, optimizer, epochs, device):
    model.train()
    
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, data in enumerate(train_loader):
            x_hr = data['hr'].to(device)
            x_lr = data['lr'].to(device)
                        
            optimizer.zero_grad()
            x_hr_pred = model(x_lr)
            
            loss = loss_function(x_hr_pred, x_hr)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        if (epoch+1)%50 == 0:
            torch.save(model.state_dict(), f'saved_models/model_ep{epoch}_l_{overall_loss/(batch_idx*batch_size)}.pth')
        
train(model, optimizer, epochs=500, device=device)
torch.save(model.state_dict(), 'saved_models/model.pth')