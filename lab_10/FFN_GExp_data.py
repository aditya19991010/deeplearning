# FFN
import numpy as np
# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import data
df = pd.read_table('GD462.GeneQuantRPKM.50FN.samplename.resk10.txt', sep='\t')
lm_id = pd.read_csv('map_lm.txt', sep='\t', header=None)
tg_id = pd.read_csv('map_tg.txt', sep='\t', header=None)

tg_id.columns = lm_id.columns = ['gene_name', 'Gene_Symbol', 'loc']


## Data preprocessing
print(
    df.describe(),'\n'
)

print(
    df.head(),'\n',
    df.columns,'\n',
    df.shape
)

## Check for NA rows

print('---'*9)
print("\nChecking NA rows...")
na_rows = df.isna().any(axis=1)
num_na_rows = df[na_rows]


if num_na_rows.shape[0] == 0:
    print("\nNo, NA rows found")
else:
    print(f'Found {num_na_rows.shape[0]} NA vals')

# Removing unecessary columns
col_drop = ['TargetID', 'Chr', 'Coord'] # drop columns
df = df.drop(columns=col_drop)

# ## Subset data for Landmark and target genes
df['Gene_Symbol'] = df['Gene_Symbol'].str.split('.').str[0]
print(df.head(5))

lm_df = df[df['Gene_Symbol'].isin(lm_id['Gene_Symbol'])].T
tg_df = df[df['Gene_Symbol'].isin(tg_id['Gene_Symbol'])].T

X = lm_df.iloc[1:].apply(pd.to_numeric)
y = tg_df.iloc[1:].apply(pd.to_numeric)


print(X.shape,y.shape) # (462, 943) (462, 9520)

# train test split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

# Train - val split
X_train_val, X_val_test_val, y_train_val, y_test_val = train_test_split(X_train_temp,y_train_temp, test_size=0.30, random_state=42 )

print(X_train_val.shape,y_train_val.shape) #(216, 943) (93, 9520)

# Data loading and converting to tensor
X_train_temp_ten = torch.tensor(X_train_temp.values,dtype=torch.float32)
X_test_ten = torch.tensor(X_test.values,dtype=torch.float32)
X_train_val_ten = torch.tensor(X_train_val.values,dtype=torch.float32)

y_test_ten = torch.tensor(y_test.values,dtype=torch.float32)
y_train_temp_ten = torch.tensor(y_train_temp.values,dtype=torch.float32)
y_train_val_ten = torch.tensor(y_train_val.values,dtype=torch.float32)


val_data = TensorDataset(X_train_val_ten, y_train_val_ten)
test_data = TensorDataset(X_test_ten, y_test_ten)
train_data = TensorDataset(X_train_temp_ten, y_train_temp_ten)

batch_size = 64
train_load = DataLoader(train_data, batch_size=64,shuffle=True)
test_load = DataLoader(test_data, batch_size=64,shuffle=True)
val_load = DataLoader(val_data, batch_size=64,shuffle=True)

### setting up device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Set class

class NN(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(NN,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self,x):
        return self.model(x)


# set params
inp_dim = X.shape[1]
outdim = y.shape[1]
model = NN(inp_dim,outdim).to(device)

## Hyperparameter
lr = 0.01
loss_func = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr)
epochs = 1000


# define train loop
def train_loop(dataloader, model, lossfun, optimizer):
    model.train()
    tot_loss =0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = lossfun(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss +=loss
    ave_loss = tot_loss/len(dataloader)
    return ave_loss

def eval_loop(dataloader, model, lossfun, optimizer):
    model.eval()
    tot_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = lossfun(pred,y_batch)
        tot_loss += loss
    ave_loss = tot_loss/len(dataloader)
    return  ave_loss

train_loss_arr = []
test_loss_arr = []
for i in range(epochs):
    train_loss_epoch = train_loop(train_load, lossfun=loss_func, optimizer=optim, model=model)
    test_loss_epoch = eval_loop(test_load, lossfun=loss_func, optimizer=optim, model=model)
    train_loss_arr.append(train_loss_epoch.item() if isinstance(train_loss_epoch, torch.Tensor) else train_loss_epoch)
    test_loss_arr.append(test_loss_epoch.item() if isinstance(test_loss_epoch, torch.Tensor) else test_loss_epoch)
    if i%100==0:
        print(f"\nEpoch {i+1}")
        print(f'Train loss : {train_loss_epoch}')
        print(f'Test loss : {test_loss_epoch}')


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(train_loss_arr, label='Train Loss', linewidth=2)
plt.plot(test_loss_arr, label='test Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('Train_test_loss.png')