# Sequence to Sequence Learning using RNN --  machine translation -- Hindi English

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = 'cpu'

# Process data
import pandas as pd

df = pd.read_csv("Hindi_eng_dataset/Hindi_English_Truncated_Corpus.csv")
print(df.shape)

df = df.iloc[:3*10**4,1:] # subset 25% samples only



def prepare_dict(df, col=0):

    words = []
    for inx in range(len(df)):
        sent =  df.iloc[inx,col]
        sent = " ".join(str(sent).split())  # normalize whitespace
        sent = str(sent).split(" ")
        words.extend(sent)

    words.extend(["<START>","<END>","<PAD>","<UNK>"])

    uniq_words = pd.DataFrame()
    uniq_words["words"] = sorted(set(words))
    uniq_words["Index"] = range(len(uniq_words)) # Words and index


    print("\nNo. of unique words: ",len(uniq_words))

    print("Initial 10 words",uniq_words[:10])

    word2vec = dict(zip(uniq_words["words"],uniq_words["Index"]))
    vec2word = dict(zip(uniq_words["Index"],uniq_words["words"]))
    return  word2vec, vec2word


print("\nPreparing dictionary...")
Eng_word2vec, Eng_vec2word = prepare_dict(df, col=0)
# print(Eng_Inx2vec.items())

Hin_word2vec, Hin_vec2word = prepare_dict(df, col=1)
# print("\nPrinting inital 5 dict items: ",dict(islice(Hin_Inx2vec.items(),5)))



## Truncate and Padding the sentences
### Token preparation

max_len = 20

def sent2index(sentence,max_len, word2vec):
    tokens = ["<START>"] +  str(sentence).split(" ") + ["<END>"]
    indices = [ word2vec.get(token,word2vec["<UNK>"]) for token in tokens]
    end_inx = indices[-1]

    if len(indices) >= max_len:
        indices = indices[:max_len-1]
        indices.extend([end_inx])
    else:
        pad_indx = word2vec.get("<PAD>")
        indices.extend([pad_indx for x in range((max_len - len(indices)))]) ## Filling the blanks
    return indices


# Prepare dataset for train and test

X = []
y = []
for i in range(len(df)):

    #English
    query=df.iloc[i, 0]
    X.append(sent2index(sentence=query, max_len=20, word2vec=Eng_word2vec))

    #Hindi
    target = df.iloc[i, 1]
    y.append(sent2index(sentence=target, max_len=20, word2vec=Hin_word2vec))


print("\nPreparing train-test dataset")
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33)

X_tr = torch.LongTensor(X_train)
y_tr = torch.LongTensor(y_train)

X_te = torch.LongTensor(X_test)
y_te = torch.LongTensor(y_test)


train_loader = DataLoader(TensorDataset(X_tr,y_tr), batch_size=32,shuffle=True)
test_loader = DataLoader(TensorDataset(X_te,y_te), batch_size=32,shuffle=True)


# Prepare encoder block
'''Model was overfitting so added regularization method such as Dropouts'''
class encoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, hid_dim, rnn_type="LSTM"):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size,embed_dim)
        self.dropout= nn.Dropout(0.5)


        if rnn_type=="LSTM":
            self.rnn = nn.LSTM(embed_dim, hid_dim,batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hid_dim,batch_first=True)

    def forward(self,x):
        embedded = self.embeddings(x)
        embedded = self.dropout(embedded)

        out, hid = self.rnn(embedded)

        return hid


class decoder(nn.Module):
    def __init__(self,vocab_size, embed_dim, hid_dim, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout= nn.Dropout(0.5)


        if rnn_type=="LSTM":
            self.rnn = nn.LSTM(embed_dim,hid_dim,batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim,hid_dim,batch_first=True)

        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self,x, hid):
        embedded = self.embeddings(x)
        embedded = self.dropout(embedded)

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
        out, hidden = self.decoder(target_seq[:,:-1],hidden) ## Removing last token

        return out


#training loop

def training_loop(dataloader, model, loss_func, optimizer):
    model.train()
    total_loss = 0
    for X_data,Y_data in train_loader:
        X_data,Y_data = X_data.to(device), Y_data.to(device)

        optimizer.zero_grad()
        outputs = model(X_data,Y_data)

        loss = loss_func(outputs.view(-1, vocab_size_hin), Y_data[:, 1:].reshape(-1)) # removing <START> token
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

            loss = loss_func(outputs.view(-1, vocab_size_hin), Y_data[:, 1:].reshape(-1))  # removing <START> token
            total_loss += loss.item()
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return total_loss / len(dataloader)


## Training with LSTM

epoch=10
vocab_size_eng = len(Eng_word2vec)
vocab_size_hin = len(Hin_word2vec)
print("English vocab:", vocab_size_eng, "Hindi vocab:", vocab_size_hin)


rnn_type='LSTM'
enc_model = encoder(vocab_size_eng,embed_dim=64, hid_dim=128, rnn_type=rnn_type)
deco_model = decoder(vocab_size_hin,embed_dim=64, hid_dim=128,rnn_type=rnn_type)

model = seq2sq_model(enc_model,deco_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss(ignore_index=Hin_word2vec["<PAD>"])

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

enc_model = encoder(vocab_size_eng,embed_dim=64, hid_dim=128, rnn_type=rnn_type)
deco_model = decoder(vocab_size_hin ,embed_dim=64, hid_dim=128,rnn_type=rnn_type)

model = seq2sq_model(enc_model,deco_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss(ignore_index=Hin_word2vec["<PAD>"])

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


## Translate

def translate_sentence(model, sentence, Eng_word2vec, Hin_vec2word, max_len=20):
    model.eval()
    with torch.no_grad():
        # 1. Convert English input sentence to indices
        src_indices = sent2index(sentence, max_len, Eng_word2vec)
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # add batch dimension

        # 2. Encode
        hidden = model.encoder(src_tensor)

        # 3. Initialize decoder input with <START> token
        start_token = Eng_word2vec.get("<START>")
        dec_input = torch.LongTensor([[start_token]]).to(device)

        translated_tokens = []

        # 4. Iteratively predict next tokens
        for _ in range(max_len):
            output, hidden = model.decoder(dec_input, hidden)
            pred_token = output.argmax(2)[:, -1].item()  # take the last token prediction
            word = Hin_vec2word.get(pred_token, "<UNK>")

            if word == "<END>":
                break
            translated_tokens.append(word)
            dec_input = torch.LongTensor([[pred_token]]).to(device)

        return " ".join(translated_tokens)



# Test Translation
print("Translation using (LSTM)")
print("=" * 50)

test_sentences = df.iloc[:5,0]

print(test_sentences)

for s in test_sentences:
    translation = translate_sentence(model, s, Eng_word2vec, Hin_vec2word=Hin_vec2word)
    print(f"\nEnglish: {s}")
    print(f"Hindi: {translation}")
