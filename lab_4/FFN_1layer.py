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


# FFN
'Create a layer --> add function to the neurons--> set number of neurons--> initialize weights--> perform FWD pass'

# Set params
n, m = train.shape
n_neu = m
n_neu_L1 = 4 #neurons in layer 1
n_neu_L2 = 3 #neurons in layer 2
output=np.random.rand(n_neu_L2,1)
epochs = 100



def train_loop(train, test):
    X = train
    y_pred = []

    for i in range(X.shape[0]):
        # INPUT
        X = train.iloc[i,:]
        X.values.reshape(-1,1)

        ## INITIALIZE WEIGHTS
        wt_1 = np.random.rand(n_neu * n_neu_L1).reshape(n_neu, n_neu_L1)
        wt_2 = np.random.rand(n_neu_L1 * n_neu_L2).reshape(n_neu_L1, n_neu_L2)

        ## Bias
        b_1 = np.zeros(n_neu_L1)
        b_2 = np.zeros(n_neu_L2)

        # Matrix multiplication
        mat_1 = X.T@wt_1 + b_1
        mat_2 = mat_1.T@wt_2 +b_2
        mat_2.reshape(-1,1)

        out_val = mat_2@output
        y_pred.append(sigmoid(out_val))

    #calculate loss
    loss = calc_RMSE(test, np.array(y_pred))
    return loss



for i in range(epochs):
    loss_info = train_loop(train, test)
    print(loss_info)

#     # INPUT

#     #set linear function
#     lin_mat = train.T@wt_1.T
#     print('\nLinear func mat:\n', lin_mat)

#     # Set activation func
#     from lab_1.activation_func_lib import activation_func


#     # No-linear activation func
#     NON_LIN_MAT = activation_func(np.array(lin_mat)).sigmoid_fun()
#     print('\nNon linear mat:\n',NON_LIN_MAT)

#     #Set output
#     output=4
#     set_wt = np.random.rand(output*NON_LIN_MAT.shape[1]).reshape(output, NON_LIN_MAT.shape[1])
#     print("\nSet weights for output\n",set_wt)

#     out = NON_LIN_MAT@set_wt.T
#     out_val = [1 if i > 0.5 else 0 for i in out.flatten()]


#     print('\n\n',out)


# quit()
# #calc loss

# # calc_RMSE(y_tru=test, y)

# # Backpropagation




# # calc loss func

# ##

# import torch as nn

# nn