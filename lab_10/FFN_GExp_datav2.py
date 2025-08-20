# FFN
import numpy as np
# Libraries
from torch.utils.data import DataLoader, TensorDataset
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

### Subset data for Landmark and target genes
df['Gene_Symbol'] = df['Gene_Symbol'].str.split('.').str[0]
print(df.head(5))


lm_df = df[df['Gene_Symbol'].isin(lm_id['Gene_Symbol'])].T
tg_df = df[df['Gene_Symbol'].isin(tg_id['Gene_Symbol'])].T


# Convert to numeric and remove headers/labels
X = lm_df.iloc[1:].apply(pd.to_numeric)
y = tg_df.iloc[1:].apply(pd.to_numeric)

# Shape check
print("X shape:", X.shape, "Y shape:", y.shape)

# Train/test split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.30, random_state=42)

print("Train shapes:", X_train.shape, y_train.shape)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# ========== Dataloaders ==========

batch_size = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ========== Device Setup ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ========== Model ==========

class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
input_dim = X.shape[1]
output_dim = y.shape[1]
model = RegressionModel(input_dim, output_dim).to(device)

# ========== Training Setup ==========

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

# ========== Training Loop ==========

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return f"{avg_loss:.4f}"

def eval_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return f"{avg_loss:.4f}"

# Training on validation dataset
for epoch in range(epochs):
    train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    val_loss = eval_loop(val_loader, model, loss_fn)
    if epoch %100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}; Train loss: {train_loss}")
        print(f"Epoch {epoch + 1}/{epochs}; Val   loss: {val_loss}\n")

import matplotlib.pyplot as plt
predicted = model(X_train_tensor)
plt.plot(X_train_tensor, y_train_tensor, 'ro', label='Original data')
plt.plot(X_train_tensor, predicted, label='Fitted line')
plt.legend()
plt.show()


print("Training complete.")
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("\nFinal evaluation on test set:")
test_loss = eval_loop(test_loader, model, loss_fn)
print("Test: ",test_loss)



# Generate stats
# Train full model
# Conclusion