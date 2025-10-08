# FFN
# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

### setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class NN_Bnorm_drop(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(NN_Bnorm_drop,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.model(x)


# define train loop
def train_loop(dataloader, model, lossfun, optimizer):
    model.train()
    tot_loss =0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = lossfun(pred, y_batch)

        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # update model weights

        tot_loss +=loss
    ave_loss = tot_loss/len(dataloader)
    return ave_loss

def eval_loop(dataloader, model, lossfun):
    model.eval()
    tot_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = lossfun(pred,y_batch)
        tot_loss += loss
    ave_loss = tot_loss/len(dataloader)
    return  ave_loss

def main():
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

    ## Standardization
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    scalar = StandardScaler()
    y = scalar.fit_transform(y)


    print(X.shape,y.shape) # (462, 943) (462, 9520)

    # train test split
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

    # Train - val split
    X_train_val, X_val_test_val, y_train_val, y_test_val = train_test_split(X_train_temp,y_train_temp, test_size=0.30, random_state=42 )

    print(X_train_val.shape,y_train_val.shape) #(216, 943) (93, 9520)

    # Data loading and converting to tensor
    X_train_temp_ten = torch.tensor(X_train_temp,dtype=torch.float32)
    X_test_ten = torch.tensor(X_test,dtype=torch.float32)
    X_train_val_ten = torch.tensor(X_train_val,dtype=torch.float32)

    y_test_ten = torch.tensor(y_test,dtype=torch.float32)
    y_train_temp_ten = torch.tensor(y_train_temp,dtype=torch.float32)
    y_train_val_ten = torch.tensor(y_train_val,dtype=torch.float32)

    ## Loading Tensor into Tensor dataset
    val_data = TensorDataset(X_train_val_ten, y_train_val_ten)
    test_data = TensorDataset(X_test_ten, y_test_ten)
    train_data = TensorDataset(X_train_temp_ten, y_train_temp_ten)

    batch_size = 64
    train_load = DataLoader(train_data, batch_size=64,shuffle=True)
    test_load = DataLoader(test_data, batch_size=64,shuffle=True)
    val_load = DataLoader(val_data, batch_size=64,shuffle=True)

    # set params
    inp_dim = X.shape[1]
    outdim = y.shape[1]
    model1 = NN(inp_dim,outdim).to(device)
    model2 = NN_Bnorm_drop(inp_dim,outdim).to(device)

    ## Hyperparameter
    lr = 0.1
    loss_func = nn.MSELoss()
    optim_1 = torch.optim.Adam(model1.parameters(), lr)
    optim_2 = torch.optim.Adam(model2.parameters(), lr)

    epochs = 400



    train_loss_arr_m1 = []
    test_loss_arr_m1 = []

    train_loss_arr_m2 = []
    test_loss_arr_m2 = []

    print("\n\nTraining model...")
    for i in range(epochs):

        ## Model without batch norm
        train_loss_epoch = train_loop(train_load, lossfun=loss_func, optimizer=optim_1, model=model1)
        test_loss_epoch = eval_loop(test_load, lossfun=loss_func, optimizer=optim_1, model=model1)
        train_loss_arr_m1.append(train_loss_epoch.item() if isinstance(train_loss_epoch, torch.Tensor) else train_loss_epoch)
        test_loss_arr_m1.append(test_loss_epoch.item() if isinstance(test_loss_epoch, torch.Tensor) else test_loss_epoch)

        ## Model with batch norm
        if i%100==0:
            print(f"\nWithout norm and dropouts\nEpoch {i+1}")
            print(f'Train loss : {train_loss_epoch}')
            print(f'Test loss : {test_loss_epoch}')

        train_loss_epoch = train_loop(train_load, lossfun=loss_func, optimizer=optim_2, model=model2)
        test_loss_epoch = eval_loop(test_load, lossfun=loss_func, optimizer=optim_2, model=model2)
        train_loss_arr_m2.append(
            train_loss_epoch.item() if isinstance(train_loss_epoch, torch.Tensor) else train_loss_epoch)
        test_loss_arr_m2.append(test_loss_epoch.item() if isinstance(test_loss_epoch, torch.Tensor) else test_loss_epoch)

        if i%100==0:
            print(f"\nWith norm and dropouts\nEpoch {i+1}")
            print(f'Train loss : {train_loss_epoch}')
            print(f'Test loss : {test_loss_epoch}')


    ## Plotting
    import matplotlib.pyplot as plt
    print('\n\nGenerating plots...')
    fig, ax = plt.subplots(1,2,figsize=(15,7),sharex=True,sharey=True)
    ax[0].plot(train_loss_arr_m1, label='Train Loss; Model1', linewidth=2)
    ax[0].plot(test_loss_arr_m1, label='test Loss; Model1', linewidth=2)
    ax[0].legend()
    ax[0].set_ylabel('MSE Loss')

    ax[1].plot(train_loss_arr_m2, label='Train Loss; Model2', linewidth=2)
    ax[1].plot(test_loss_arr_m2, label='test Loss; Model2', linewidth=2)
    ax[1].legend()

    fig.text(0.5,0.04,'Epoch', fontsize=12)


    plt.suptitle('Training and Validation Loss')
    ax[0].grid(True)
    ax[1].grid(True)

    plt.savefig('Train_test_loss.png')
    print('Saved train-test error plot..')

if __name__=="__main__":
    main()