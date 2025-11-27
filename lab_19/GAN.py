# Import Libraries


# Load & Preprocess Dataset
# Define Hyperparameters

# Build Generator Model
#Build Discriminator Model

# Define Loss Functions
# Define Optimizers for G and D

# Combine Generator + Discriminator (Adversarial Model)
# Start Training Loop

# For each epoch:
#   1. Sample real data batch
#   2. Generate fake data batch using Generator
#   3. Train Discriminator on real and fake data
#   4. Train Generator via combined model (keeping D frozen)
#   5. Calculate & Log Loss/Accuracy
#   6. Save Generated Samples for Monitoring
#   7. Update Model Weights

# R --> S{Converged?}

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

train_set = DS.MNIST(root='.', train=True, download=True, transform=transform)
test_set = DS.MNIST(root='.', train=False, download=True, transform=transform)

batch_size = 32
train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

img_data, label = next(iter(train_data))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(img_data[i].reshape(28, 28), cmap='gray_r')
    ax.set_xlabel(label[i])
    plt.xticks([])
    plt.yticks([])

## Prepare model

### Discriminator
'''
output is 1-d
'''


class Discrimator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        output = self.model(x)
        return output


## preare a latent sample space


## Generator
'''
Generate 28x28 image from this
'''

class Generator(nn.Module):
    def __init__(self,latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), 1, 28, 28)
        return output


discrimator = Discrimator().to(device=device)
generator = Generator(latent_dim=100).to(device=device)

lr = 0.001
loss_function = nn.BCELoss()

optim_discrim = torch.optim.Adam(discrimator.parameters(), lr=lr)
optim_generate = torch.optim.Adam(generator.parameters(), lr=lr)


epoch=5
latent_dim=100


## TRAINING GENERATOR AND DISCRIMATOR
for i in range(epoch):
    discrimator.train()
    generator.train()

    run_loss_D = 0
    run_loss_G = 0

    for X_data, y_data in train_data:
        X_data, y_data = X_data.to(device=device), y_data.to(device=device)

        b = X_data.size(0) # get batch size

        # ------------------------
        ## generate real and fake labels for samples per batch
        real_label = torch.ones((b, 1)).to(device)
        fake_label = torch.zeros((b, 1)).to(device)

        # ------------------------
        # Train model on real data using discriminator
        optim_discrim.zero_grad()

        # prediction with discrimator using real data
        pred_discrim = discrimator(X_data)
        ## calculate loss on real data
        loss_real = loss_function(pred_discrim, real_label)

        # Train model on fake data using discriminator
        #generate fake images
        z = torch.randn((b, latent_dim)).to(device)
        fake_X = generator(z)

        pred_fake = discrimator(fake_X.detach()) # Disable update on Generator
        ## calculate loss on fake data
        loss_fake = loss_function(pred_fake, fake_label)

        # average loss by discriminator
        loss_discrimator = (loss_real + loss_fake) / 2

        #update discriminator
        loss_discrimator.backward()
        optim_discrim.step()

        # ------------------------
        ## train generator
        optim_generate.zero_grad()

        z = torch.randn((b, latent_dim)).to(device)
        fake_X = generator(z)

        pred_generate = discrimator(fake_X) # No detachment, SO that discriminator will update

        loss_generate = loss_function(pred_generate, real_label)
        loss_generate.backward()
        optim_generate.step()
        # ------------------------

        ## calculate loss

        run_loss_D += loss_discrimator.item()
        run_loss_G += loss_generate.item()

    print(f"Epoch:{i}", run_loss_D, run_loss_G)

generator.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, device=device)
    samples = generator(z).cpu() * 0.5 + 0.5   # denormalize from tanh to [0,1] range

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4,4, figsize=(6,6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
