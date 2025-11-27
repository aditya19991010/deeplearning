#GRU

import torch
from sklearn.model_selection import train_test_split
from torch import nn
device = 'cpu'
seqs = ['CAGTAGCTAGCT', 'CGTAGCTGAGCA','CGTACGTAGCTA', 'TGCATGCATGCA', 'GCTACGTAGGCA', 'GCTACGTAGCTA']
y = [1,1,0,1,0,1]
seqs_all = seqs*100
y = y*100

def seq2tok_generator(sequence):
    indices = []
    seq = [x for x in sequence]
    for i in seq:
        indices.append(seq.index(i))

    return indices


nucls = list(sorted(set(x for x in seqs[0])))
print(nucls)

nuc2vec = {nuc:idx for idx, nuc in enumerate(nucls)}
vec2vec = {idx:nuc for idx, nuc in enumerate(nucls)}

print(nuc2vec)

X_data = []
y_data= []
for seq, label in zip(seqs, y):
    X_data.append(seq2tok_generator(seq))
    y_data.append(label)

import numpy as np
## converting to array
X_data = np.array(X_data)
y_data = np.array(y_data)

## train test split
test_size=0.2
print(f"Training={100* (1-test_size)} data")
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

X_tr = torch.tensor(X_train, dtype=torch.long)
y_tr = torch.tensor(y_train, dtype=torch.long)

X_te = torch.tensor(X_test, dtype=torch.long)
y_te = torch.tensor(y_test, dtype=torch.long)

from torch.utils.data import Dataset, DataLoader, TensorDataset

# preparing dataloaders
train_data = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
test_data = DataLoader(TensorDataset(X_te, y_te), batch_size=16, shuffle=False)


vocab_size = len(nuc2vec)

class GRU(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size, output_size=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        embedded = self.embed(x)
        output,_ = self.gru(embedded)
        output = self.fc(output)
        return output

##train loop
def training_loop(train_data, loss_func, optimiser, model):
    model.train()
    tota_loss = 0
    correct = 0

    for X_load, y_load in train_data:
        X_load, y_load = X_load.to(device), y_load.to(device)
        y_load = y_load.unsqueeze(1)

        optimiser.zero_grad()
        pred = model(X_load)

        loss = loss_func(pred, y_load)

        loss.backward()
        optimiser.step()

        _,predicted = pred.max(1)
        correct += predicted.eq(y_load).sum().item()


        #calculate loss
        tota_loss+=loss.item()

    loss_net = tota_loss / len(train_data)
    accuracy = 100 * correct / len(train_data.dataset)

    return loss_net,accuracy

## test loop
def test_loop(test_data, loss_func, model):
    model.train()
    tota_loss = 0
    correct = 0

    with torch.no_grad():
        for X_load, y_load in test_data:
            X_load, y_load = X_load.to(device), y_load.to(device)
            y_load = y_load.unsqueeze(1)

            pred = model(X_load)
            loss = loss_func(pred, y_load)

            _, predicted = pred.max(1)
            correct += predicted.eq(y_load).sum().item()

            # calculate loss
            tota_loss += loss.item()


    loss_net = tota_loss / len(test_data)
    accuracy = 100 * correct / len(test_data.dataset)

    return loss_net,accuracy

## Set up model
model = GRU(vocab_size=vocab_size, embed_size=12, hidden_size=64, output_size=1)
model.to(device)
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

## train the model
epoch = 50

print("\nTraining the model")
for i in range(epoch):
    train_loss, train_acc = training_loop(train_data, loss_func, optimiser, model)
    if i % 10 == 0:
        print(f"Epoch {i}/{epoch} ; train loss={train_loss}; Accuracy = {train_acc}")

print("\n\nTesting the model...")
for i in range(epoch):
    test_loss, test_acc = test_loop(test_data, loss_func, model)

    if i % 10 == 0:
        print(f"Epoch {i}/{epoch} ; test loss={test_loss} ; Accuracy = {test_acc}")
