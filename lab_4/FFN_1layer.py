# Full neural netwwork from scratch

import pandas as pd

# input
## Creating dataframe
data = [[0, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 0]]

columns = ['x1', 'x2', 'x3', 'y']
df = pd.DataFrame(data, columns=columns)

# TRAIN-TEST SPLIT
train=df.iloc[:,:-1]
test = df.iloc[:,-1]

# loss function
def calc_RMSE(y_tru, y_pre):
    return np.mean(((y_tru - y_pre)**2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deri_sigmoid(z):
    sig = sigmoid(z)
    return sig*(1 - sig)



import numpy as np
mat = np.array(train)

print(mat.T)

'Create a layer --> add function to the neurons--> set number of neurons--> initialize weights--> perform FWD pass'

# Set params

n_neu = 5
output=4
epochs = 100
n, m = train.shape

## INITIALIZE WEIGHTS
wt_1 = np.random.rand(n_neu * n).reshape(n_neu, n)
wt_2 = np.random.rand(n_neu * n).reshape(n_neu, n)

## Bias
b_1 = np.zeros()

# FFN
# iterations
for i in range(epochs):
    #set linear function
    lin_mat = train.T@wt.T
    print('\nLinear func mat:\n', lin_mat)

    # Set activation func
    from lab_1.activation_func_lib import activation_func


    # No-linear activation func
    NON_LIN_MAT = activation_func(np.array(lin_mat)).sigmoid_fun()
    print('\nNon linear mat:\n',NON_LIN_MAT)

    #Set output
    output=4
    set_wt = np.random.rand(output*NON_LIN_MAT.shape[1]).reshape(output, NON_LIN_MAT.shape[1])
    print("\nSet weights for output\n",set_wt)

    out = NON_LIN_MAT@set_wt.T
    out_val = [1 if i > 0.5 else 0 for i in out.flatten()]


    print('\n\n',out)


quit()
#calc loss

# calc_RMSE(y_tru=test, y)

# Backpropagation




# calc loss func

##

import torch as nn

nn