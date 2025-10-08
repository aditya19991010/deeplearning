# RNN for image caption problem
## LIBRARIES
from PIL import Image
import pandas as pd
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import tensorflow
from pandas.io.xml import preprocess_data
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch.optim as optim
from networkx.classes.filters import hide_edges



# Params
train_img_data = '/home/workshop/learning/deeplearning/lab_15/Filcker8k_text/Flickr_8k.trainImages.txt'
capt_data = '/home/workshop/learning/deeplearning/lab_15/Filcker8k_text/Flickr8k.token.txt'
image_Data = '/home/workshop/learning/deeplearning/lab_15/Flicker8k_Dataset'

caption = pd.read_table(capt_data,sep='\t',header=None)

caps = caption[0].str.split('#')
caps_0 = [caps[i][0] for i in range(len(caps))]
caps_1 = [caps[i][1] for i in range(len(caps))]

cap_df_1 = pd.DataFrame([caps_0, caps_1]).T

cap_df_1['caption'] = caption[1]
cap_df_1.columns = ['image', 'cap_num', 'caption']

# print(caps_0)

## Data Process
### Image data
img_files = os.listdir('Flicker8k_Dataset')

img_tensor = {img_files[i]: Image.open(os.path.join(image_Data,img_files[i])).convert('RGB') for i in range(len(img_files)) }


preprocess_img = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

for key, val in img_tensor.items():
    img_tensor[key] = preprocess_img(val)


## Images are transformed to tensor dataset saved in the dict

## Use transfer learning to extract the features from the dataset
## Using ResNet18
from torchvision import models

model1 = models.resnet18(weights=True)
print(model1)
quit()

# Fixed ConvLayers
for param in model1.parameters():
    param.requires_grad = False

model1.fc = nn.Linear(model1.fc.in_features, len(classes))
model1 = model1.to('cpu')
# 341ec7c8061bc3ddc0f48786beb856a2d479e8f6
# e42b19f3ce3d6c83f973fa026362517fe279a3b2
quit()
class VanillaRNN(nn.Module):
    def __init__(self, ip=10, hidden_Size=10,op=1):
        super(VanillaRNN, self).__init__()

        self.hid_size = hidden_Size

        self.rnn = nn.RNN(ip,hidden_Size, op, batch_first=True)

        self.fc = nn.Linear(hidden_Size, op )

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out =self.fc(out)
        return out, hidden


model = VanillaRNN().to(device).to(torch.float64)



## Set model hyper param



loss_func = nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(), lr)

model = model.to(torch.float32)



## Training loop
def training_loop(dataloader,loss_func, optim, model):
    model.train()
    tot_loss = 0

    # iterate through each batch of the dataloader
    # print(f'Training model...')
    for X_train, y_train in dataloader:

        # move the data into the device (gpu/cpu)
        X_train, y_train = X_train.to(device), y_train.to(device)

        ## predict --> calc loss --> set grad to 0 --> perf bwd prop --> perform optimization for updating model wghts

        pred = model(X_train)
        loss = loss_func(pred, y_train)

        optim.zero_grad()
        loss.backward()
        optim.step()

        tot_loss +=loss.item()
    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader)
    return ave_loss

def eval_loop(model, loss_fn, dataloader):
    model.eval()
    tot_loss = 0
    # print(f'\nEvaluating model...')
    for X_test, y_test in dataloader:
        X_test, y_test = X_test.to(device), y_test.to(device)

        pred = model(X_test)
        loss = loss_fn(pred, y_test)

        tot_loss +=loss.item()

    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader)
    return ave_loss

## Training with multiple epoch

train_loss_tot = []
test_loss_tot = []

print(f"Starting training with {epoch} epochs... ")
for i in range(epoch):
    tr_loss = training_loop(train_load, loss_func, optim, model)
    te_loss = eval_loop(model,loss_func,test_load)

    train_loss_tot.append(tr_loss)
    test_loss_tot.append(te_loss)
    print(f"\tEpoch {i}")
    print(f"Train loss: \t{tr_loss}")
    print(f"Test loss: \t{te_loss}")

    if i%100==0:
        print(f"\nEpoch {i}")
        print(f"Train loss at Epoch {i} = {np.mean(train_loss_tot)}")
        print(f"Test loss at Epoch {i} = {np.mean(test_loss_tot)}")



plt.plot(train_loss_tot, label='train')
plt.plot(test_loss_tot, label='test')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.show()


# Training loop
