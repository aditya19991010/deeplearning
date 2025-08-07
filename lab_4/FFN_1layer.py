# Full neural netwwork from scratch

#input
import pandas as pd

# Creating dataframe
data = [[0, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 0]]

columns = ['x1', 'x2', 'x3', 'y']
df = pd.DataFrame(data, columns=columns)

# TRAIN-TEST SPLIT
train=df.iloc[:,:1]
test = df.iloc[:,-1]

# loss function
def calc_RMSE(y_tru, y_pre):
    return 0.5*((y_tru - y_pre)**2)

epochs = 10

import numpy as np
mat = np.array(train)

print(mat.T)

'Create a layer --> add function to the neurons--> set number of neurons--> initialize weights--> perform FWD pass'


# iterations
# FFN

## set number of neurons
n_neu=5
n,m = train.shape
## INITIALIZE WEIGHTS

wt = np.random.rand(n_neu*n).reshape(n_neu,n)
print("\nWeight mat:\n", wt.T)

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
print('\n\n',out)
quit()
#calc loss

# calc_RMSE(y_tru=test, y)

# Backpropagation
def deri_sigmoid(x):
    sig_x = 1/(1 + np.exp(-x))
    return sig_x*(1 - sig_x)



# calc loss func

##

import torch as nn

nn