# learning source = [https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html,
# 'https://www.learnpytorch.io/06_pytorch_transfer_learning/']
## Import libraries
import os
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

os.environ["HTTP_PROXY"] = "http://249000049%40ibab.ac.in:Aditya0%40@proxy.ibab.ac.in:3128"
os.environ["HTTPS_PROXY"] = "http://249000049%40ibab.ac.in:Aditya0%40@proxy.ibab.ac.in:3128"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device ='cpu'

## Training loop
def training_loop(dataloader,loss_func, optim, model):
    model.train()
    tot_loss = 0
    total = 0
    correct = 0

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
        _,predicted = pred.max(1)
        total += y_train.size(0)
        correct += predicted.eq(y_train).sum().item()


    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader.dataset)
    accuracy = 100 * correct / total

    return ave_loss, accuracy

def eval_loop(model, loss_fn, dataloader):
    model.eval()
    tot_loss = 0

    total = 0
    correct = 0

    with torch.no_grad():
        # print(f'\nEvaluating model...')
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            pred = model(X_test)
            loss = loss_fn(pred, y_test)

            tot_loss +=loss.item()
            _,predicted = pred.max(1)
            total += y_test.size(0)
            correct += predicted.eq(y_test).sum().item()

    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader.dataset)
    accuracy = 100 * correct / total

    return ave_loss, accuracy




def train_model(model,epoch,train_load,test_load,loss_func,optim, save=True,pre_name="DL_model_checkpoint"):
    ## Training with multiple epoch
    train_loss_tot = []
    test_loss_tot = []

    acc_tr_tot = []
    acc_te_tot = []

    print(f"Starting training with {epoch} epochs... ")
    for i in range(epoch):
        tr_loss, acc_tr = training_loop(train_load, loss_func, optim, model)
        te_loss, acc_te = eval_loop(model,loss_func,test_load)

        train_loss_tot.append(tr_loss)
        test_loss_tot.append(te_loss)

        acc_tr_tot.append(acc_tr)
        acc_te_tot.append(acc_te)

        print(f"\n\tEpoch {i}")
        train_net_loss = np.mean(train_loss_tot)
        test_net_loss = np.mean(test_loss_tot)
        train_net_acc = np.mean(acc_tr_tot)
        test_net_acc = np.mean(acc_te_tot)
        print(f'Train loss: {train_net_loss:.3f}| acc: {train_net_acc:.3f}')
        print(f"Test loss: \t{test_net_loss:.3f} | acc: {test_net_acc:.3f}")

        if save:
            name = f'{pre_name}_{int(time.time())}'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim,
                'train_losses': train_net_loss,
                'test_losses': test_net_loss,
                'train_accuracies': train_net_acc,
                'test_accuracies': test_net_acc,
            },name )
            print(f"\nModel saved to '{name}'")

    loss = [train_loss_tot, test_loss_tot]
    acc = [acc_tr_tot, acc_te_tot]

    return loss, acc


def plot_loss_acc(losses1,losses2,accuracies1,accuracies2):
    fig, ax = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    # Model 1 - Loss
    ax[0, 0].plot(losses1[0], label='Train')
    ax[0, 0].plot(losses1[1], label='Test')
    ax[0, 0].set_title("Model 1 - ConvNet Fixed Loss")
    ax[0, 0].legend()

    # Model 1 - Accuracy
    ax[1, 0].plot(accuracies1[0], label='Train')
    ax[1, 0].plot(accuracies1[1], label='Test')
    ax[1, 0].set_title("Model 1 - ConvNet Fixed Accuracy")
    ax[1, 0].legend()

    # Model 2 - Loss
    ax[0, 1].plot(losses2[0], label='Train')
    ax[0, 1].plot(losses2[1], label='Test')
    ax[0, 1].set_title("Model 2 - Finetune ConvNet Loss")
    ax[0, 1].legend()

    # Model 2 - Accuracy
    ax[1, 1].plot(accuracies2[0], label='Train')
    ax[1, 1].plot(accuracies2[1], label='Test')
    ax[1, 1].set_title("Model 2 - Finetune ConvNet Accuracy")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig("transfer_learning_res18toCIFAR10.pdf")


def main():
    torch.manual_seed(42)

    print("Setting params...")
    lr = 0.1
    epoch = 5
    batch = 64

    trans_manual = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print('\nTransform - Normalization - loading...')
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_manual)
    train_load = DataLoader(trainset, batch_size=batch, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_manual)
    test_load = DataLoader(testset, batch_size=batch, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print("\n\nTRAINING MODEL")
    print('\n\tModel 1 - ConvNet as fixed feature extractor')

    ## Model 1 - ConvNet as fixed feature extractor
    model1 = models.resnet18(weights=True)

    # Fixed ConvLayers
    for param in model1.parameters():
        param.requires_grad = False

    model1.fc = nn.Linear(model1.fc.in_features, len(classes))
    model1 = model1.to('cpu')

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model1.parameters(), lr=lr, momentum=0.9)

    losses1, accuracies1 = train_model(model1,epoch,train_load,test_load,loss_func,optim)


    ## Model 2 -- Finetuning the ConvNet
    print('\n\tModel 2 - Finetuning the ConvNet')

    model2 = models.resnet18(weights=True)

    model2.fc = nn.Linear(model2.fc.in_features, len(classes))
    model2 = model2.to('cpu')

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9)
    losses2, accuracies2 = train_model(model2,epoch,train_load,test_load,loss_func,optim)

    plot_loss_acc(losses1, losses2, accuracies1, accuracies2)

    print("0-0-0-0-END-0-0-0-0-0")


if __name__=="__main__":
    main()