# Required Libraries
import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

from torchvision.models import ResNet18_Weights

device = "cpu"

# Params
train_img_data = '/home/workshop/learning/deeplearning/lab_15/Filcker8k_text/Flickr_8k.trainImages.txt'
test_img_data = '/home/workshop/learning/deeplearning/lab_15/Filcker8k_text/Flickr_8k.testImages.txt'
capt_data = '/home/workshop/learning/deeplearning/lab_15/Filcker8k_text/Flickr8k.token.txt'
image_Data = '/home/workshop/learning/deeplearning/lab_15/Flicker8k_Dataset'

# Load captions
caption = pd.read_table(capt_data, sep='\t', header=None)
caption['index'] = [c[1] for c in caption[0].str.split('#')]
caption['image'] = [c[0] for c in caption[0].str.split('#')]
caption['caption'] = caption.iloc[:,1]

cap_df = caption[['index', 'image', 'caption']]

cap_list = []
[cap_list.append(str(i).split(" ")) for i in cap_df['caption']]
l1 = cap_list[99]

vocab = {}

#preparing vocab with all lower case
counter = 0
for sent in cap_list:
    for word in sent:
        vocab[counter] = str(word).lower()
        counter += 1
vocab_df = pd.DataFrame().from_dict(vocab, orient='index', columns=['word'])

uniq_vocab = pd.DataFrame()
uniq_vocab['word'] = vocab_df['word'].unique()
uniq_vocab['index'] = range(0,len(uniq_vocab['word']))

## adding words to make up for padding for different length captions
uniq_vocab.loc[len(uniq_vocab)] = ['<START>', len(uniq_vocab)]
uniq_vocab.loc[len(uniq_vocab)] = ['<END>', len(uniq_vocab)]
uniq_vocab.loc[len(uniq_vocab)] = ['<PAD>', len(uniq_vocab)]
word_to_index = dict(zip(uniq_vocab['word'], uniq_vocab['index']))
index_to_word = dict(zip(uniq_vocab['index'], uniq_vocab['word']))



def caption_to_indices(sentence, uniq_vocab, max_length=20):
    # converting caption to indices
    # set max length of the caption, Padding will be done if less than this, else cut the length
    indices = [uniq_vocab['index'][uniq_vocab['word'] == word.lower()].iloc[0] for word in sentence.split()]
    if len(indices) < max_length:
        pad_len = max_length - len(indices)
        pad_index = uniq_vocab['index'][uniq_vocab['word'] == '<PAD>'].iloc[0]
        indices.extend([pad_index] * pad_len)
    else:
        indices = indices[:max_length]
    return indices



with open(train_img_data, 'r') as f:
    train_imgs = [line.strip() for line in f.readlines()]

with open(test_img_data, 'r') as f:
    test_imgs = [line.strip() for line in f.readlines()]


preprocess_img = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.486,0.456,0.406], std=[0.229,0.224,0.225])
])




## Read image and corresponding caption
### extract features from image

model= models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

for params in model.parameters():
    params.requires_grad = False

## feature extractor
feature_ext = nn.Sequential(*list(model.children())[:-1])


def feat_extractor(img_path):
    img = Image.open(os.path.join(image_Data, img_path)).convert('RGB')
    pros_img = preprocess_img(img).unsqueeze(0)
    with torch.no_grad():
        img_features = feature_ext(pros_img).squeeze().detach()

    return img_features




## convert caption to indices
class CreateImgCapDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_dir, vocab, max_length, cap_df):
        self.img_list= img_list
        self.img_dir= img_dir
        self.vocab=vocab
        self.max_length=max_length
        self.cap_df=cap_df

        # filter captions of the images
        self.data = cap_df[cap_df['image'].isin(img_list)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        row = self.data.iloc[indx]
        img_name = row['image']
        caption = row['caption']

        # Extract image features
        img_path = os.path.join(self.img_dir, img_name)
        img_features = feat_extractor(img_path)

        # Convert caption to indices
        caption_indices = caption_to_indices(caption, self.vocab, self.max_length)

        return img_features, torch.tensor(caption_indices[:-1]), torch.tensor(caption_indices[1:])



print('Preparing dataset..')
## Create a dataset
## prepare dataloader
train_dataset = CreateImgCapDataset(img_list=train_imgs[:500], img_dir=image_Data, vocab=uniq_vocab, max_length=20, cap_df=cap_df)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = CreateImgCapDataset(img_list=test_imgs[:500], img_dir=image_Data, vocab=uniq_vocab, max_length=20, cap_df=cap_df)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

## Prepare model
inp_dim=256 # h or w of the image
vocab_size = len(uniq_vocab)
class Image_RNN(nn.Module):
    def __init__(self, inp_dim, embed_dim, hid_dim, n_lay, feat_dim, vocab_size):
        super(Image_RNN, self).__init__()
        self.fc1 = nn.Linear(inp_dim, feat_dim)
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hid_dim, n_lay, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, img_feat, captions):
        img_embed = self.dropout(torch.relu(self.fc1(img_feat)))
        caption_embed = self.embedding(captions)
        img_embed = img_embed.unsqueeze(1)
        inputs = torch.cat((img_embed, caption_embed), dim=1)
        out_lstm, _ = self.lstm(inputs)
        outputs = self.fc_out(out_lstm)
        return outputs


model = Image_RNN(inp_dim=512, embed_dim=512, hid_dim=512, vocab_size=vocab_size, feat_dim=512, n_lay=2)
model.to(device)

## train-test-evaluate



def training_loop(dataloader, model, loss_func, optimizer):
    model.train()
    total_loss = 0
    for img_feat, cap_in, cap_out in train_loader:
        img_feat, cap_in, cap_out = img_feat.to(device), cap_in.to(device), cap_out.to(device)

        optimizer.zero_grad()
        outputs = model(img_feat, cap_in)

        # Shift outputs to match target length
        outputs = outputs[:, :-1, :].reshape(-1, vocab_size)
        cap_out = cap_out.reshape(-1)

        loss = loss_func(outputs, cap_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return total_loss / len(dataloader)


def eval_loop(dataloader, model, loss_func):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img_feat, cap_in, cap_out in dataloader:
            img_feat, cap_in, cap_out = img_feat.to(device), cap_in.to(device), cap_out.to(device)

            outputs = model(img_feat, cap_in)

            # Shift outputs to match target length
            outputs = outputs[:, :-1, :].reshape(-1, vocab_size)
            cap_out = cap_out.reshape(-1)

            loss = loss_func(outputs, cap_out)
            total_loss += loss.item()
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return total_loss / len(dataloader)


# Training
train_loss_hist = []
test_loss_hist = []
loss_func = nn.CrossEntropyLoss(ignore_index=uniq_vocab['index'][uniq_vocab['word'] == '<PAD>'].iloc[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch=5

print(f"\nTraining for {epoch} epochs...")
for ep in range(epoch):
    train_loss = training_loop(train_loader, model, loss_func, optimizer)
    test_loss = eval_loop(test_loader, model, loss_func)

    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)

    if (ep + 1) % 1 == 0:
        print(f"Epoch {ep + 1}/{epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

def generate_caption(model, image_path, max_length=20, temperature=1.0):
    "Generate caption"
    model.eval()
    with torch.no_grad():
        img_feat = feat_extractor(image_path).unsqueeze(0).to(device)  # [1,512]
        inputs = torch.tensor([[word_to_index['<START>']]], dtype=torch.long).to(device)
        generated_caption = []

        hidden = None
        img_embed = model.dropout(torch.relu(model.fc1(img_feat)))  # [1,512]
        img_embed = img_embed.unsqueeze(1)  # [1,1,512]

        for _ in range(max_length):
            emb = model.embedding(inputs)  # [1, seq_len, 512]
            inp_seq = torch.cat((img_embed, emb), dim=1)
            out, hidden = model.lstm(inp_seq, hidden)
            logits = model.fc_out(out[:, -1, :])  # [1, vocab_size]

            # apply temperature and sample
            probs = torch.softmax(logits / temperature, dim=-1)
            _, next_word = torch.max(probs, dim=-1)
            next_word_idx = next_word.item()
            word = index_to_word[next_word_idx]

            if word == '<END>' or word == '<PAD>':
                break
            generated_caption.append(word)

            inputs = torch.cat([inputs, torch.tensor([[next_word_idx]], device=device)], dim=1)

    caption = ' '.join(generated_caption)
    return caption

def show_image_with_caption(image_path, caption):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()


import random
test_image_path = os.path.join(image_Data, random.choice(test_imgs))
pred_caption = generate_caption(model, test_image_path)
print("\nGenerated Caption:")
print(pred_caption)
show_image_with_caption(test_image_path, pred_caption)