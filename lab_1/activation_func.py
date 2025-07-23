

# Sigmoid

import numpy as np

z = np.linspace(-10,10, 100)

class activation_func:
    def __init__(self,z):
        self.z = z

    def sigmoid_fun(self):
        return 1/(1 + np.exp(-self.z))
    # Tanh

    def tanh_fun(self):
        return np.tanh(self.z)
    # ReLU (Rectified Linear Unit)

    def ReLU_fun(self):
        x = [i if i > 0 else 0 for i in self.z]
        return x
    # Leaky ReLU
    def leaky_ReLU_fun(self,a=0.01):
        x = [i if i > 0 else a*i for i in self.z]
        return x


    # Softmax (no need for visualization)
    def Softmax(self):
        n = np.exp(self.z)
        return n/np.sum(np.exp(self.z))


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2, figsize=(6,20))

act_func= activation_func(z)
act_func.sigmoid_fun()

ax[0].plot(z, act_func.sigmoid_fun(), color='red', label='sigmoid')
ax[0].plot(z, act_func.tanh_fun(),color='green', label='tanh')
ax[0].set_title("Sigmoid vs Tanh")

ax[1].plot(z, act_func.ReLU_fun(), color='violet', label='ReLU')
ax[1].plot(z, act_func.leaky_ReLU_fun(), color='yellow', linestyle='--', label='Leaky ReLU')
ax[1].set_title("ReLU and leaky ReLU")
fig.legend()

plt.suptitle("Comparsion of different activation function")
plt.show()
