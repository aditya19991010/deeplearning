

# Sigmoid

import numpy as np

z = np.linspace(-10,10, 100)


def sigmoid_fun(z):
    return 1/(1 + np.exp(-z))
# Tanh

def tanh_fun(z):
    return np.tanh(z)
# ReLU (Rectified Linear Unit)

def ReLU_fun(z):
    x = [i if i > 0 else 0 for i in z]
    return x
# Leaky ReLU
def leaky_ReLU_fun(z,a):
    x = [i if i > 0 else a*i for i in z]
    return x


# Softmax (no need for visualization)
def Softmax(z):
    n = np.exp(z)
    return n/np.sum(np.exp(z))


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2, figsize=(6,20))

ax[0].plot(z, sigmoid_fun(z), color='red', label='sigmoid')
ax[0].plot(z, tanh_fun(z),color='green', label='tanh')
ax[0].set_title("Sigmoid vs Tanh")

ax[1].plot(z, ReLU_fun(z), color='violet', label='ReLU')
ax[1].plot(z, leaky_ReLU_fun(z,0.01), color='yellow', linestyle='--', label='Leaky ReLU')
ax[1].set_title("ReLU and leaky ReLU")
fig.legend()

plt.suptitle("Comparsion of different activation function")
plt.show()
