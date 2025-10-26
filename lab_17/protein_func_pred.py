import torch
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
from torch.utils.data import TensorDataset

## Data processing

df_gpcr = pd.read_table("family_wise/uniprotkb_protein_name_GPCR_OR_protein_2025_10_23.tsv", sep='\t')
df_p450 = pd.read_table("family_wise/uniprotkb_protein_name_P450_AND_reviewe_2025_10_23.tsv", sep='\t')
df_ferritin = pd.read_table("family_wise/uniprotkb_protein_name_Ferritin_OR_prot_2025_10_23.tsv", sep='\t')
df_protease = pd.read_table("family_wise/uniprotkb_protein_name_Protease_AND_rev_2025_10_23.tsv", sep='\t')

df_gpcr.iloc[:,-2] = "GPCR"
df_p450.iloc[:,-2] = "P450"
df_ferritin.iloc[:,-2] = "Ferritin"
df_protease.iloc[:,-2] = "Protease"

df = pd.concat([df_gpcr,df_p450,df_ferritin,df_protease])
print(df.columns)

df_main = df[['Sequence','Protein families' ]]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df_main['label'] = le.fit_transform(df_main['Protein families'])

print(df_main.head())
## Trunc or Pad sequences
import numpy as np
seq_len = np.array(df_main.iloc[:,0].apply(len))
ave_seq = np.ceil(np.mean(seq_len))
print('\nAverage seq length:',ave_seq)


print('\nThis length will be useful for truncating and padding the sequencing for constant embeddings')



## Prepare encodings
## Creating vocab of aa
aa_vocab = [list(seq) for seq in df_main['Sequence'][:100].astype(str)]
aa = sorted(list(set([aa for seq in aa_vocab for aa in seq])))
print("Total amino acids:", len(aa))

aa_df = pd.DataFrame(aa)

enc = OneHotEncoder(sparse_output=False)
aa_df = pd.DataFrame(aa, columns=["AA"])

## padding and truncating code
aa_df.loc[len(aa_df)] = ['<PAD>']
aa_df.loc[len(aa_df)] = ['<Start>']
aa_df.loc[len(aa_df)] = ['<end>']



aa_enc = enc.fit_transform(aa_df)



aa_vec_dict = {aa_df.iloc[i,0]:aa_enc[i] for i in range(len(aa_df))}
aa_df_enc = pd.DataFrame({
    'AA': list(aa_vec_dict.keys()),
    'encoding': list(aa_vec_dict.values())
})

print(aa_df_enc)
# Prepare dataframe for training model

'''
Read each sentence
Read each amino acid
fetch encoding and add into the list in the corresponding column
use each cells as a sample encoding
'''

print('Preparing encodings...')
enc_dict_seq = {}


## set len of the seq
seq_len_min = 100
n_seqs = 100

enc_dict_seq = df_main[:n_seqs].copy()
encodings = []  # store per-sequence encodings

for inx in range(len(enc_dict_seq)):
    seq = enc_dict_seq.iloc[inx, 0]
    fam = enc_dict_seq.iloc[inx, 1]

    seq_aa = list(seq)

    # pad or truncate sequence
    if len(seq_aa) < seq_len_min:
        seq_aa.extend(['<PAD>'] * (seq_len_min - len(seq_aa)))
    else:
        seq_aa = seq_aa[:seq_len_min]

    # map AAs to encodings
    encoding = [aa_df_enc.loc[aa_df_enc['AA'] == x, 'encoding'].values[0] for x in seq_aa]

    encodings.append(np.array(encoding, dtype=np.float32))

enc_dict_seq['encoding'] = encodings

print(enc_dict_seq.head())

## preparing dataset
X = np.stack(enc_dict_seq['encoding'].values)
y = enc_dict_seq['label'].astype(np.int64).values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors

print('Preparing Tensor dataset...')
X_tr_ten = torch.tensor(X_train, dtype=torch.float32)
y_tr_ten = torch.tensor(y_train, dtype=torch.long)

X_te_ten = torch.tensor(X_test, dtype=torch.float32)
y_te_ten = torch.tensor(y_test, dtype=torch.long)


train_ten = TensorDataset(X_tr_ten,y_tr_ten)
test_ten = TensorDataset(X_te_ten,y_te_ten)

from torch.utils.data import DataLoader

train_data = DataLoader(train_ten, batch_size=64, shuffle=True)
test_data = DataLoader(test_ten, batch_size=64, shuffle=False)


## Model
aa_size=seq_len_min

class LSTM_model(nn.Module):
    def __init__(self,emb_dim, hid_dim):
        super(LSTM_model,self).__init__()

        self.embeddings = nn.Embedding(aa_size, emb_dim)
        self.rnn1 = nn.LSTM(emb_dim,hid_dim)
        self.dropout=  nn.Dropout(0.5)

        self.rnn2 = nn.LSTM(hid_dim, 128)
        self.dropout=nn.Dropout(0.5)

        self.rnn3 = nn.LSTM(128, 128)
        self.dropout=nn.Dropout(0.5)

        self.fc = nn.Linear(128,4)

    def forward(self,x, hidden):
        embedded= self.embeddings(x)
        out, hid = self.rnn1(embedded,hidden)
        out, hid = self.rnn2(out)
        out, hid = self.rnn3(out)

        pred = self.fc(out)

        return pred




