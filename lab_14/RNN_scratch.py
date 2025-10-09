## RNN from scratch
import numpy as np

from lab_1.activation_func_lib import activation_func

## Initialization
# 1. w_hh = weight matrix of hidden states
# 2. w_xh = weight mat connecting input and hidden states
# 3. h_0 = weighted sum of 0th hidden state and input
# 4. x_t_1= input

# 5. h_t = activationfunct( w_hh* ht_1 + w_xh*x_t ) weighted sum of t-1th hidden state and input at t
# 6. y_t =  w_hy*h_t ( weighted sum of t-1th hidden state)


## Pseudocode
'''
1. Prepare encodings of the input
2. calculate ht
3. calculate y_t
4. pass ht to next w_hh
5. change the index to next 

'''

raw_inp = 'abcde'
inpt = list(raw_inp)
# print("\nInput for RNN: ",inpt)
t = len(inpt) # total states


inpt_arr = np.array(inpt).reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder

x_enc = OneHotEncoder(sparse_output=False).fit_transform(inpt_arr)
x_t = x_enc

np.random.seed(12)

#input of a single word --> encoded to a vector
print("Input embeddings",x_t,'\tShape:',x_t.shape)

#Initial hidden state
vocab_size = x_t.shape[1]
hidden_size = 4

# Initialize weights and hidden state
np.random.seed(12)
h_0 = np.random.rand(hidden_size, 1)

w_xh = np.random.rand(hidden_size, vocab_size)
w_hh = np.random.rand(hidden_size, hidden_size)
w_hy = np.random.rand(vocab_size, hidden_size)

y_true = x_t[1:]  # next character
x_t = x_t[:-1]    # input without last character
t = len(x_t)

# Run RNN forward pass
losses = []

for i in range(t):
    x = x_t[i].reshape(-1, 1)

    h_t = np.tanh(w_xh@x + w_hh@h_0) # Hidden state update

    y_pred = activation_func(w_hy @ h_t).Softmax()  # output

    y = y_true[i].reshape(-1, 1)

    # Compute  loss
    loss = -np.sum(y * np.log(y_pred ))
    losses.append(loss)

    # Update hidden state
    h_0 = h_t


print('mean loss=',np.mean(loss))

