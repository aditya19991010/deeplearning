# FFN
import numpy as np
# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

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

# Model training
feat_train = torch.from_numpy(X_train_val.values)
targ_train = torch.from_numpy(y_train_val.values)

feat_train_shape = feat_train.shape
targ_train_shape = targ_train.shape

### setting up device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Hyper parameter tuning
model = NeuralNetwork(943,9520 ).to(device)
learning_rate = 1e-3
batch_size = 64
epochs = 5

## loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Done!")


# Generate stats
# Train full model
# Generate statistics
# Conclusion