## Multiple attemps to make dataset and converting to Torch format -- Failed

import torch
import torchvision
from nltk import download
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import ToTensor
import os
import sys

from torchvision.transforms.v2.functional import pil_to_tensor

device = 'cpu'


## Create class

class CNN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()  # correct parent init

        self.features = nn.Sequential(
            # Conv  1
            nn.Conv2d(inp, 32, kernel_size=3, stride=1, padding=1),  # preserve HxW
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv  2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),   # must match 64 channels
            nn.ReLU(inplace=True),

            # Conv  3
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(32, out)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



## train loop


def train_loop(dataloader, lossfn, optim, model):
    model.train()
    tot_loss = 0

    tot_lab = 0
    correct =0

    for X_batch, y_batch in dataloader:
        X,y = X_batch.to(device), y_batch.to(device)

        pred = model(X)
        loss = lossfn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        tot_loss += loss.item()

        ## accuracy
        tot_lab += y_batch.size(0)


    average_loss = tot_loss/len(dataloader)
    accuracy = correct/ tot_lab

    return average_loss, accuracy


def test_loop(dataloader, lossfn, model):
    model.eval()
    tot_loss = 0
    tot_lab = 0
    correct =0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X, y = X_batch.to(device), y_batch.to(device)

            pred = model(X)
            loss = lossfn(pred, y)

            tot_loss += loss.item()
            ## accuracy
            tot_lab += y_batch.size(0)


    average_loss = tot_loss / len(dataloader)
    accuracy = correct / tot_lab

    return average_loss , accuracy


def train_model(train_dataloader, test_dataloader, lossfn, optim, model, epoch=3):
    train_loss= []
    train_acc = []

    test_loss = []
    test_acc = []



    for i in range(epoch):
        trainloss , trainacc = train_loop(dataloader=train_dataloader, lossfn=lossfn, optim=optim, model=model)

        testloss, testacc = test_loop(dataloader=test_dataloader, lossfn=lossfn, model=model)

        train_loss.append(trainloss)
        train_acc.append(trainacc)

        test_loss.append(testloss)
        test_acc.append(testacc)


    print(epoch)
    print(f'Train loss: {np.mean(train_loss)}')
    print(f'Train loss: {np.mean(test_loss)}')
    print(f'Train accuracy: {np.mean(train_acc)}')
    print(f'Train accuracy: {np.mean(test_acc)}')


## Main

def main():

    print('Failed \n Tried multiple attempts\n')

    # paths
    train_dir = 'data/imagenet_subset/train'
    test_dir = 'data/imagenet_subset/test'

    # sanity: check directory structure
    print("Train subfolders (classes):", os.listdir(train_dir))
    print("Test subfolders (classes):", os.listdir(test_dir))

    # transforms
    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor(),
    ])

    # use ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


    print("Classes:", train_dataset.classes)
    print("class_to_idx:", train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    ## Hyperparameters
    epoch =5
    lr = 0.01
    model = CNN(3,6).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()


    ## Training the model
    # Question 1

    print('Training model based on CNN')
    train_model(train_dataloader=train_loader, test_dataloader=test_loader, lossfn=lossfn, optim=optim, model=model, epoch=epoch)

    from torchvision.models import resnet18, ResNet18_Weights

    # Question 2
    model = resnet18(weights=ResNet18_Weights)

    for param in model.parameters():
        param.requires_grad = False

    ## Approach 1
    model.fc = nn.Linear(in_features=512, out_features=6)

    model_trans = model

    # Approach 2
    class Classfier(nn.Module):
        def __init__(self, output=6):
            super().__init__()
            self.classifier = nn.Linear(in_features=512, out_features=output)

        def forward(self, x):
            return self.classifier(x)

    optim = torch.optim.SGD(model_trans.parameters(), lr=0.01)
    lossfn = nn.CrossEntropyLoss()

    epoch = 5
    print('Training model based on transfer learning')

    train_model(train_dataloader=train_loader, test_dataloader=test_loader, lossfn=lossfn, optim=optim,
                model=model_trans, epoch=epoch)

if __name__=='__main__':
    main()