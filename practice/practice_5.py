## GAN
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as DS
import torchvision.transforms as transforms

device = 'cpu'
torch.manual_seed(42)
# prepare data
## Real Sample data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

train_set = DS.MNIST(root='/home/workshop/learning/deeplearning/dataset/data', train=True, download=True, transform=transform)
test_set = DS.MNIST(root='/home/workshop/learning/deeplearning/dataset/data', train=False, download=True, transform=transform)

batch_size = 32
train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# Discriminator

class discriminator(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.model(x)


class generator(nn.Module):
    def __init__(self, latent_size, output_size=28*28):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 1,28,28)
        return x

X,y = train_set[0]

D = discriminator(X.shape[1]*X.shape[2], output_size=1)
latent_size=100
G = generator(latent_size=100, output_size=X.shape[1]*X.shape[2])

loss_func = nn.BCELoss()

import torch
optim_D =  torch.optim.Adam(D.parameters(), lr=0.0001)
optim_G =  torch.optim.Adam(G.parameters(), lr=0.0001)

epoch=5
for i in range(epoch):
    D.train()
    G.train()

    runn_loss_D=0
    runn_loss_G=0

    for X_data,y_Data in train_data:
        # prepare label for real and fake data
        b = X_data.size(0)
        real_label = torch.ones((b,1))
        fake_label = torch.zeros((b,1))

        ## train discriminator on real and fake data
        optim_D.zero_grad()
        pred_real = D(X_data)
        loss_real = loss_func(pred_real, real_label)

        ## Create latent space to generate fake data
        z = torch.randn((b,latent_size))
        fake_X = G(z)
        pred_fake = D(fake_X.detach())

        loss_fake = loss_func(pred_fake, fake_label)

        ## calculate average loss
        loss_D = (loss_fake + loss_real) /2
        loss_D.backward()
        optim_D.step()


        ## train generator on fake data
        optim_G.zero_grad()
        ## Create latent space to generate fake data
        z = torch.randn((b, latent_size))
        fake_X = G(z)
        pred_gen = D(fake_X)

        loss_G = loss_func(pred_gen, real_label)

        ## calculate average loss
        loss_G.backward()
        optim_G.step()

        runn_loss_D += loss_D.item()
        runn_loss_G += loss_G.item()

    print(runn_loss_D,runn_loss_G)

