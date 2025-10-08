# Vanishing and exploding gradient

## Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class DeepNN(nn.Module):
    def __init__(self, num_lay,input_size,output_size):
        super(DeepNN, self).__init__()
        layers = []

        for i in range(num_lay):
            layers.append(nn.Linear(input_size, input_size))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Params
num_layers = 50
model = DeepNN(num_layers, input_size=500, output_size=10)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Test data
input_data = torch.randn(1, 500)  # Batch size of 1, 500 feat
target = torch.randint(0, 10, (1,))  # target class

# Fwd pass
output = model(input_data)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

# Bwd pass
optimizer.zero_grad()
loss.backward()

# Track gradients
gradients = []
for i in model.parameters():
    gradients.append(i.grad.abs().mean().item())
    # [ for param in model.parameters() if param.grad is not None]

# Plot gradients
plt.plot(gradients)
plt.title("Gradient across layers")
plt.xlabel("Layer")
plt.ylabel("mean gradient")
plt.show()


# Experiment with different activation functions
activation_functions = [nn.Sigmoid, nn.ReLU]
for activation_fn in activation_functions:
    model = DeepNN(num_layers,input_data.shape[1],output_size=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
