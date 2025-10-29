## Training MNIST handwriting data
import matplotlib.pyplot as plt
import torch
from tensorflow.keras.datasets import mnist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

device = torch.device("cuda")

## Model 
class CNN(nn.Module):
    def __init__(self, op=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, op)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


## Training loop
def training_loop(dataloader,loss_func, optim, model):
    model.train()
    tot_loss = 0
    correct=0
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
        _,predicted = pred.max(1)
        correct += predicted.eq(y_train).sum().item()

        tot_loss +=loss.item()
    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader)
    print(f'\nTrain Accuracy: {correct/len(dataloader.dataset)}')

    return ave_loss

def eval_loop(model, loss_fn, dataloader):
    model.eval()
    tot_loss = 0
    correct =0
    # print(f'\nEvaluating model...')
    for X_test, y_test in dataloader:
        X_test, y_test = X_test.to(device), y_test.to(device)

        pred = model(X_test)
        loss = loss_fn(pred, y_test)
        _,predicted = pred.max(1)
        correct += predicted.eq(y_test).sum().item()
        tot_loss +=loss.item()

    # print(f'\nCalculating loss...')
    ave_loss = tot_loss/len(dataloader)
    print(f'\nTest Accuracy: {correct/len(dataloader.dataset)}')
    return ave_loss


def main():
  # Import data
  df = mnist.load_data()

  (X_train, y_train), (X_test, y_test) =  df
  data = [X_train, y_train ,X_test, y_test]

  # Data preprocessing
  print('X_train:', X_train.shape)
  print('y_train:', y_train.shape)
  print('X_test:', X_test.shape)
  print('y_test:', y_test.shape)

  # Data standardatization

  torch.manual_seed(42)

  ## Convert np to tensor and reshaping the size
  X_train_ten = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0  # -> (60000, 1, 28, 28)
  X_test_ten  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0   # -> (10000, 1, 28, 28)
  y_train_ten = torch.tensor(y_train, dtype=torch.long)
  y_test_ten  = torch.tensor(y_test, dtype=torch.long)

  print('\nConvert data to tensor')

  ## transform data
  transform = transforms.Normalize(mean=[0.5], std=[0.5])
  X_train_ten = transform(X_train_ten)
  X_test_ten = transform(X_test_ten)


  ## Convert tensor to tensor dataset
  train_df = TensorDataset(X_train_ten, y_train_ten)
  test_df = TensorDataset(X_test_ten, y_test_ten)
  print('\nConvert tensor to tensor dataset')


  ## Data loading
  train_load = DataLoader(train_df,batch_size=64, shuffle=True)
  test_load = DataLoader(test_df,batch_size=64, shuffle=True)
  print('\nData loaded successfully')


  ## Set up models
  model = CNN(op=10).to(device).to(torch.float64)


  ## Set model hyper param
  l,n,m = X_train.shape
  p = y_train.shape[0]
  lr = 0.01
  batch = 64
  epoch=5

  loss_func = nn.CrossEntropyLoss()
  optim=torch.optim.Adam(model.parameters(), lr)

  model = model.to(torch.float32)
  ## Training with multiple epoch

  train_loss_tot = []
  test_loss_tot = []

  print(f"Starting training with {epoch} epochs... ")
  for i in range(epoch):
      tr_loss = training_loop(train_load, loss_func, optim, model)
      te_loss = eval_loop(model,loss_func,test_load)

      train_loss_tot.append(tr_loss)
      test_loss_tot.append(te_loss)
      print(f"\n\tEpoch {i}")
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

if __name__ == "__main__":
    main()
