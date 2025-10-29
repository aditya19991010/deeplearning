import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
from torch.utils.data import TensorDataset

device='cpu'
## Data processing

df_gpcr = pd.read_table("/content/uniprotkb_protein_name_GPCR_OR_protein_2025_10_23.tsv", sep='\t')
df_p450 = pd.read_table("/content/uniprotkb_protein_name_P450_AND_reviewe_2025_10_23.tsv", sep='\t')
df_ferritin = pd.read_table("/content/uniprotkb_protein_name_Ferritin_OR_prot_2025_10_23.tsv", sep='\t')
df_protease = pd.read_table("/content/uniprotkb_protein_name_Protease_AND_rev_2025_10_23.tsv", sep='\t')

df_gpcr.iloc[:,-2] = "GPCR"
df_p450.iloc[:,-2] = "P450"
df_ferritin.iloc[:,-2] = "Ferritin"
df_protease.iloc[:,-2] = "Protease"

df = pd.concat([df_gpcr,df_p450,df_ferritin,df_protease])
print(df.columns)

df_main = df[['Sequence','Protein families' ]].copy()
print(df_main['Protein families'].unique())


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label2enc = { val:indx for indx, val in enumerate(df_main['Protein families'].unique())}
enc2label = { indx:val for indx, val in enumerate(df_main['Protein families'].unique())}

lab_df = pd.DataFrame.from_dict(label2enc,orient='index').reset_index(drop=False)
df_main['label'] = df_main['Protein families'].map(label2enc)

print(df_main.head())

## Trunc or Pad sequences
import numpy as np
seq_len = np.array(df_main.iloc[:,0].apply(len))
ave_seq = np.ceil(np.mean(seq_len))
print('\nAverage seq length:',ave_seq)


print('\nThis length will be useful for truncating and padding the sequencing for constant embeddings')


## Prepare vocab
## Creating vocab of aa
aa_vocab = [list(seq) for seq in df_main['Sequence'][:100].astype(str)]
aa = sorted(list(set([aa for seq in aa_vocab for aa in seq])))
print("Total amino acids:", len(aa))

aa_df = pd.DataFrame(aa)

enc = LabelEncoder()
aa_df = pd.DataFrame(aa, columns=["AA"])

## padding and truncating code
aa_df.loc[len(aa_df)] = ['<PAD>']
aa_df.loc[len(aa_df)] = ['<START>']
aa_df.loc[len(aa_df)] = ['<END>']
aa_df.loc[len(aa_df)] = ['<UNK>']

aa_enc = enc.fit_transform(aa_df)



aa2vec_dict = {aa_df.iloc[i,0]:aa_enc[i] for i in range(len(aa_df))}
# aa_df_enc = pd.DataFrame({
#     'AA': list(aa_vec_dict.keys()),
#     'encoding': list(aa_vec_dict.values())
# })



# Prepare tokens

'''
Read each sentence
Read each amino acid
fetch encoding and add into the list in the corresponding column
use each cells as a sample encoding
'''

## set len of the seq
seq_len_min = 400
print(f"Set Sequence length={seq_len_min}")

# enc_dict_seq = df_main.iloc[:n_seqs,[0,2]].copy()
def tokenization(sequence, aa2idx, seq_len_min=10):
    token = ["<START>"] + list(sequence) + ["<END>"]
    if len(token) > seq_len_min:
        token = token[:seq_len_min-1]
        token.append("<END>")
    else:
        token += ["<PAD>"] * (seq_len_min - len(token))

    encoded = [aa2idx.get(a, aa2idx["<UNK>"]) for a in token]
    return encoded


## Prepare encoded dataset


# dataset = pd.DataFrame()
n_seqs = np.random.randint(0,len(df_main), size=1000) ##Randomly selecting samples
sequences = df_main.iloc[n_seqs,[0,2]]

X = []
y = []
for i in range(len(sequences)):
    seq_aa = [list(sequences.iloc[i,0]) for x in range(len(sequences))][0]
    seq_aa1 = tokenization(seq_aa, aa2vec_dict)
    X.append(seq_aa1)

    label = sequences.iloc[i,1]
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"\nX shape: {X.shape}")  # Should be (n_samples, seq_len, vocab_size)
print(f"y shape: {y.shape}")  # Should be (n_samples,)
## preparing dataset

# Split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33)

# Convert to tensors

print('\nPreparing Tensor dataset...')
X_tr_ten = torch.tensor(X_train, dtype=torch.long)
y_tr_ten = torch.tensor(y_train, dtype=torch.long)

X_te_ten = torch.tensor(X_test, dtype=torch.long)
y_te_ten = torch.tensor(y_test, dtype=torch.long)


train_ten = TensorDataset(X_tr_ten,y_tr_ten)
test_ten = TensorDataset(X_te_ten,y_te_ten)

from torch.utils.data import DataLoader

train_data = DataLoader(train_ten, batch_size=64, shuffle=True)
test_data = DataLoader(test_ten, batch_size=64, shuffle=False)



## Model
vocab_size=len(aa2vec_dict)

class LSTM_model(nn.Module):
    def __init__(self,vocab_size,n_class,emb_dim, hid_dim,n_layer):
        super(LSTM_model,self).__init__()

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layer,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        self.batch_norm = nn.BatchNorm1d(hid_dim * 2)
        self.dropouts = nn.Dropout(0.5 )

        self.fc = nn.Linear(hid_dim*2,n_class)

    def forward(self,x):
        embedded= self.embeddings(x)

        out, (h_n, c_n) = self.lstm(embedded)

        h_for = h_n[-2,:,:]
        h_back = h_n[-1,:,:]

        h_cat = torch.cat((h_for, h_back), dim=1)
        h_cat = self.batch_norm(h_cat)
        h_cat = self.dropouts(h_cat)
        logits = self.fc(h_cat)
        return logits



def training_loop(dataloader, model, loss_func, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for X_data,Y_data in dataloader:
        X_data,Y_data = X_data.to(device), Y_data.to(device)

        optimizer.zero_grad()
        outputs = model(X_data)

        loss = loss_func(outputs, Y_data)
        loss.backward()
        optimizer.step()
        _,predicted = outputs.max(1)
        correct += predicted.eq(Y_data).sum().item()
        total_loss += loss.item()
    # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    accuracy = 100* correct/len(dataloader.dataset)

    return total_loss / len(dataloader), accuracy


def eval_loop(dataloader, model, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_data, Y_data in dataloader:
            X_data, Y_data = X_data.to(device), Y_data.to(device)

            outputs = model(X_data)

            loss = loss_func(outputs, Y_data)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(Y_data).sum().item()
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    accuracy = 100 * correct/len(dataloader.dataset)
    return total_loss / len(dataloader) , accuracy

epoch=15

print("Model n_class:", len(set(y)))
model = LSTM_model(vocab_size=vocab_size,n_class=len(set(y)),emb_dim=64, hid_dim=128,n_layer=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)


train_loss_hist = []
test_loss_hist = []

print(f"\nTraining for {epoch} epochs...")
for ep in range(epoch):
    train_loss, train_acc = training_loop(train_data, model, loss_func, optimizer)
    train_loss_hist.append(train_loss)

    test_loss,test_acc = eval_loop(test_data, model, loss_func)
    test_loss_hist.append(test_loss)

    if (ep + 1) % 5 == 0:
        print(f"Epoch {ep + 1}/{epoch}: Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        print(f"Epoch {ep + 1}/{epoch}: Test Loss = {test_loss:.4f}, Test Loss = {test_acc:.4f}")


##Inference

def predict_protein_family(sequence, model, aa2vec_dict, enc2label, seq_len_min, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize and encode the sequence
        encoded_sequence = tokenization(sequence, aa2vec_dict, seq_len_min=seq_len_min)

        # Convert to tensor and add a batch dimension
        sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.long).unsqueeze(0).to(device)

        # Get model output
        outputs = model(sequence_tensor)

        # Get the predicted class index
        _, predicted_index = torch.max(outputs, dim=1)

        # Convert the predicted index back to the protein family label
        predicted_label = enc2label[predicted_index.item()]

        return predicted_label



seq_len_min=400
for x in range(10):
    sample_sequence = df_main.iloc[x]['Sequence']
    print(f"Sample sequence: {sample_sequence[:seq_len_min]}") # Print first 50 characters

    predicted_family = predict_protein_family(sample_sequence, model, aa2vec_dict, enc2label, seq_len_min, device)

    print(f"Predicted protein family: {predicted_family}")