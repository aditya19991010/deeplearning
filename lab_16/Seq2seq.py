# Sequence to Sequence Learning using RNN
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = 'cpu'

# generate data
vocab_size=50
seq_length=10
n_samples=1000
X = np.random.randint(1,vocab_size,(n_samples,seq_length))
y = X.copy()


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33)

X_tr = torch.LongTensor(X_train)
y_tr = torch.LongTensor(y_train)

X_te = torch.LongTensor(X_test)
y_te = torch.LongTensor(y_test)


train_loader = DataLoader(TensorDataset(X_tr,y_tr), batch_size=32,shuffle=True)
test_loader = DataLoader(TensorDataset(X_te,y_te), batch_size=32,shuffle=True)

# Prepare encoder block

class encoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, hid_dim, rnn_type="LSTM"):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size,embed_dim)
        if rnn_type=="LSTM":
            self.rnn = nn.LSTM(embed_dim, hid_dim,batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hid_dim,batch_first=True)

    def forward(self,x):
        embedded = self.embeddings(x)
        out, hid = self.rnn(embedded)

        return hid


class decoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, hid_dim, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        if rnn_type=="LSTM":
            self.rnn = nn.LSTM(embed_dim,hid_dim,batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim,hid_dim,batch_first=True)

        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self,x, hid):
        embedded = self.embeddings(x)
        out, hid = self.rnn(embedded,hid)
        preds = self.fc(out)

        return preds, hid

class seq2sq_model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_seq, target_seq):
        hidden = self.encoder(source_seq)
        out, hidden = self.decoder(target_seq,hidden)

        return out


#



def training_loop(dataloader, model, loss_func, optimizer):
    model.train()
    total_loss = 0
    for X_data,Y_data in train_loader:
        X_data,Y_data = X_data.to(device), Y_data.to(device)

        optimizer.zero_grad()
        outputs = model(X_data,Y_data)

        loss = loss_func(outputs.view(-1, vocab_size), Y_data.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return total_loss / len(dataloader)


def eval_loop(dataloader, model, loss_func):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_data, Y_data in dataloader:
            X_data, Y_data = X_data.to(device), Y_data.to(device)

            outputs = model(X_data, Y_data)

            loss=loss_func(outputs.view(-1, vocab_size), Y_data.view(-1))
            total_loss += loss.item()
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return total_loss / len(dataloader)


## Training with LSTM

epoch=10


rnn_type='LSTM'
enc_model = encoder(vocab_size,embed_dim=32, hid_dim=64, rnn_type=rnn_type)
deco_model = decoder(vocab_size,embed_dim=32, hid_dim=64)

model = seq2sq_model(enc_model,deco_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

train_loss_hist = []
test_loss_hist = []

print(f"\nTraining for {epoch} epochs...")
for ep in range(epoch):
    train_loss = training_loop(train_loader, model, loss_func, optimizer)
    test_loss = eval_loop(test_loader, model, loss_func)

    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)

    if (ep + 1) % 1 == 0:
        print(f"Epoch {ep + 1}/{epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")




## Training with GRU
rnn_type='GRU'

enc_model = encoder(vocab_size,embed_dim=32, hid_dim=64, rnn_type=rnn_type)
deco_model = decoder(vocab_size,embed_dim=32, hid_dim=64,rnn_type=rnn_type)

model = seq2sq_model(enc_model,deco_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

train_loss_hist_gru = []
test_loss_hist_gru = []

print(f"\nTraining for {epoch} epochs...")
for ep in range(epoch):
    train_loss = training_loop(train_loader, model, loss_func, optimizer)
    test_loss = eval_loop(test_loader, model, loss_func)

    train_loss_hist_gru.append(train_loss)
    test_loss_hist_gru.append(test_loss)

    if (ep + 1) % 1 == 0:
        print(f"Epoch {ep + 1}/{epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")



## Plotting
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,2,figsize=(18,8))
ax[0].plot(train_loss_hist, label='LSTM')
ax[0].plot(train_loss_hist_gru, label='GRU')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].set_title(f"Train loss: {rnn_type}")


ax[1].plot(test_loss_hist, label='LSTM')
ax[1].plot(train_loss_hist_gru, label='GRU')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].set_title(f"Test loss: {rnn_type}")
fig.suptitle(f"Train-test Loss in RNN")

plt.savefig("rnn_lstm_gru_performance.png")
plt.show()
quit()
# Using LSTM




# Prepare decoder block









import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 1. Generate synthetic sequence data ======
def generate_data(n_samples=1000, seq_len=10, vocab_size=20):
    X = np.random.randint(1, vocab_size, (n_samples, seq_len))
    y = X.copy()  # Copy task
    return X, y, vocab_size

X, y, vocab_size = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# ====== 2. Dataset Class ======
class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(Seq2SeqDataset(X_train, y_train), batch_size=64, shuffle=True)

# ====== 3. Encoder ======
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden

# ====== 4. Decoder ======
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        predictions = self.fc(output)
        return predictions, hidden

# ====== 5. Seq2Seq Model ======
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden = self.encoder(src)
        outputs, _ = self.decoder(trg, hidden)
        return outputs

# ====== 6. Training Function ======
def train_model(model, optimizer, criterion, n_epochs=10):
    model.train()
    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, y_batch)
            loss = criterion(outputs.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {losses[-1]:.4f}")
    return losses

# ====== 7. Initialize & Train Models ======
embed_dim = 32
hidden_dim = 64

# LSTM Model
encoder_lstm = Encoder(vocab_size, embed_dim, hidden_dim, rnn_type="LSTM")
decoder_lstm = Decoder(vocab_size, embed_dim, hidden_dim, rnn_type="LSTM")
model_lstm = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training LSTM Seq2Seq...")
losses_lstm = train_model(model_lstm, optimizer_lstm, criterion)

# GRU Model
encoder_gru = Encoder(vocab_size, embed_dim, hidden_dim, rnn_type="GRU")
decoder_gru = Decoder(vocab_size, embed_dim, hidden_dim, rnn_type="GRU")
model_gru = Seq2Seq(encoder_gru, decoder_gru).to(device)
optimizer_gru = optim.Adam(model_gru.parameters(), lr=0.001)

print("\nTraining GRU Seq2Seq...")
losses_gru = train_model(model_gru, optimizer_gru, criterion)

# ====== 8. Compare Performance ======
plt.plot(losses_lstm, label="LSTM")
plt.plot(losses_gru, label="GRU")
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
