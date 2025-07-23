'''
W is a matrix, x is a vector, z is a vector,
 and a is a vector. y^ is a scalar and a final prediction.
 Initialize x, w randomly,
 z is a dot product of x and w,
 a is ReLU(z).
  Initialize X and W randomly. Every neuron has a bias term.
'''


import numpy as np

from lab_1.activation_func import activation_func

np.random.seed(42)

#perceptron
x = np.linspace(-10,10, 4).reshape(4,1)
w = np.random.rand(4).reshape(4,1)
z = w.T@x

print(f'z: {z}')

import lab_1.activation_func

func = activation_func(z)
a=func.ReLU_fun()
print(f'a: {a}')


# NN
"- set input"
"- set layers"
"- set neurons per layer"

x = np.linspace(-10,10, 4).reshape(4,1)
w = np.random.rand(12).reshape(4,3)
z = x.T@w
print(z)

def FFN(inp):
    import lab_1.activation_func
    m,n = inp.shape

    # get inputs
    #input number of layers
    n_lay = int(input("How many hidden layers would you like proceed with?\n"))
    n_lay_cp= n_lay

    func_dict = {1:activation_func(z).sigmoid_fun(),2:activation_func(z).tanh_fun(),
                 3:activation_func(z).ReLU_fun(),
                 4:activation_func(z).leaky_ReLU_fun(), 5:activation_func(z).sigmoid_fun()}

    lay_neurons = {}
    func_per_lay = {}
    while n_lay != 0:
        n_lay -= 1
        lay_neurons[n_lay] = input(f'How many neurons you would like to set in {1} layer?')
        func_per_lay[n_lay] = input(f'Which activation function would you like to use in {1} layer?\n'
                                       f'type number only'
                                       f'1. Sigmoid\n2.Tanh\n3.ReLU\n4.Leaky ReLU\n5.Softmax)\n')


    for nlay,neu in lay_neurons.items():
        print(f'Training for layer {nlay} with {neu} neurons' )
        w = np.random.rand(m*n).reshape(m, neu)
        z = inp.T @ w
        a = func_dict[func_per_lay[nlay]]
        inp = a





FFN(x)