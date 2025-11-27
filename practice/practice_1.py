## 1 IMPLEMENT FUNCTIONS

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

z = np.linspace(-10,10,100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh_func(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return np.maximum(0, x)


## LOAD CIFAR-10 data

import torchvision.datasets as DS
from torchvision import transforms

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                              transforms.Normalize([0.486, 0.456, 0.406], std = [0.229, 0.224, 0.225])                              ])

train_data=DS.CIFAR10(root='/home/workshop/learning/deeplearning/dataset/data/', train=True, transform=transform, download=False)

print(len(train_data))

n_classes = len(train_data.classes)
test_data=DS.CIFAR10(root='/home/workshop/learning/deeplearning/dataset/data/', train=False, transform=transform, download=False)

train_dataload=torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_dataload=torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

## Prepare model class
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes, ker_size=3, stride=1):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 32*224*224, n_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifer(x)
        return x


# print*
device = 'cpu'
model = CNN(n_classes=n_classes, ker_size=3, stride=1)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


## Prepare train loop

def train_loop(model=model, dataloader=train_dataload, loss_func=loss_function, optim=optimizer):
    model.train()
    total_loss=0
    accuracy = 0
    for X_data, y_data in dataloader:
        X_data, y_data = X_data.to(device), y_data.to(device)

        pred= model(X_data)
        loss = loss_func(pred, y_data)

        optim.zero_grad()

        loss.backward()

        optim.step()
        total_loss += loss.item()

        accuracy += pred.argmax(dim=1).eq(y_data).sum().item()

    mean_loss = total_loss/len(dataloader.dataset)
    mean_acc = accuracy/len(dataloader.dataset) *100

    return mean_loss, mean_acc


def test_loop(model=model, dataloader=test_dataload, loss_func=loss_function, optim=optimizer):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for X_data, y_data in dataloader:
            X_data, y_data = X_data.to(device), y_data.to(device)

            pred = model(X_data)
            loss = loss_func(pred, y_data)

            total_loss += loss.item()

            accuracy += pred.argmax(dim=1).eq(y_data).sum().item()

        mean_loss = total_loss / len(train_dataload)
        mean_acc = accuracy / len(train_dataload)

    return mean_loss, mean_acc


epoch =5
train_loss = []
train_acc = []

test_loss = []
test_acc = []

for i in range(epoch):
    print('Epoch: ',i)
    tr_loss, tr_acc = train_loop(model=model, dataloader=train_dataload, loss_func=loss_function, optim=optimizer)

    print(tr_loss,tr_acc)

    train_loss.append(tr_loss)
    train_acc.append(tr_acc)

    te_loss, te_acc = test_loop(model=model, dataloader=test_dataload, loss_func=loss_function, optim=optimizer)
    print(te_loss, te_acc)

    test_loss.append(te_loss)
    test_acc.append(te_acc)




