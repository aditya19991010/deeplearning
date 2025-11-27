## MNIST dataset
# input = 784 (28*28)
from torch.utils.data import DataLoader, TensorDataset

# 2 hidden later
## hid_1 = 128 + ReLU
## hid_2 = 64 + ReLU
# out = 10 + Softmax

epoch =5
device='cpu'

import torch
import torchvision.datasets as DS
import torch.nn as nn
import numpy
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = DS.MNIST(root='/home/workshop/learning/deeplearning/dataset/data/', train=True, transform=transform, download=False)
test_data = DS.MNIST(root='/home/workshop/learning/deeplearning/dataset/data/', train=False, transform=transform, download=False)

train_dataload = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataload = DataLoader(test_data, batch_size=64, shuffle=False)

class NN(nn.Module):
    def __init__(self,inp,hid_1,hid2,out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inp,hid_1),
            nn.ReLU(),
            nn.Linear(hid_1,hid2),
            nn.ReLU(),
            nn.Linear(hid2,out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0),-1) ## flatten the x
        return self.model(x)


loss_func = nn.MSELoss()

model = NN(inp=28*28,hid_1=128,hid2=64,out=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train_loop(model,dataloader,optim, loss_func):
    model.train()
    total_loss = 0
    accuracy=0

    for X_Data,y_data in dataloader:
        X_Data,y_data = X_Data.to(device) , y_data.to(device)

        optim.zero_grad()
        pred= model(X_Data)

        # Convert labels to one-hot
        y_onehot = torch.nn.functional.one_hot(y_data, num_classes=10).float()
        ## SOftmax + above code with MSELoss ; Cross entropy + SOftmax doesnot work together

        loss = loss_func(pred,y_onehot)
        loss.backward()
        optim.step()

        total_loss += loss.item()

        predic=pred.argmax(1)
        accuracy+=predic.eq(y_data).sum().item()

    mean_loss = total_loss/len(dataloader.dataset)
    mean_accuracy = accuracy/len(dataloader.dataset) *100

    return mean_loss,mean_accuracy

for i in range(epoch):
    mean_loss, mean_accuracy = train_loop(model, train_dataload, optimizer, loss_func)
    print(mean_loss, mean_accuracy)



