import torch
import torch.nn as nn
import numpy

#forward pass - single layer
inp,out=3,1

class Single_layerNN(nn.Module):
    def __init__(self, input, out):
        super().__init__()
        self.fc1 = nn.Linear(input, out)

    def forward(self, x):
        x = self.fc1(x)
        return x

device='cpu'
model = Single_layerNN(input=inp,out=out)
model.to(device).float()

x = torch.tensor([1,2,3]).float()
y = model(x)



## Autoencoder

X = torch.rand(728).float()

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim=32, hid_dim=128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x

## Prepare  latent dim data

latent_sample = torch.randn(1,32)
latent_dim=latent_sample.shape[1]

class decoder(nn.Module):
    def __init__(self, latent_dim=32, hid_dim=64,out_dim=728):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder  = decoder

    def forward(self, latent_sample):
        latent = self.encoder(latent_sample)
        x = self.decoder(latent)
        return x


model = Autoencoder(Encoder(in_dim=728), decoder=decoder(latent_dim=latent_dim)).to(device)

print(model(X).shape)



