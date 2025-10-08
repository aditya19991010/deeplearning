# XOR function

import numpy as np

# Truth table
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

def XOR_nn(x,b=1,wt=np.random.rand(x.shape[0]*x.shape[1]).reshape(x.shape[0],x.shape[1])):
  y_pred = []
  for i in x:
    z = wt@i + b
    y_pred.append(1 if z > 0 else 0)

  return np.array(y_pred)

# activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
def initialize_parameters(inp_size, hid_size, out_size):
    np.random.seed(42) # for reproducibility
    weights_input_hidden = np.random.randn(hid_size, inp_size) * 0.01

    bias_hidden = np.zeros((hid_size, 1))

    weights_hidden_output = np.random.randn(out_size, hidden_size) * 0.01

    bias_output = np.zeros((out_size, 1))

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Forward pass
def forward_pass(x, weights_in_hid, bias_hid, weights_hid_out, bias_out):
    # Input layer to hidden layer
    hidden_layer_input = np.dot(weights_in_hid, x.T) + bias_hid
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Hidden layer to output layer
    output_layer_input = np.dot(weights_hid_out, hidden_layer_output) + bias_out
    output_layer_output = sigmoid(output_layer_input)

    return output_layer_output, hidden_layer_output

# Backward pass
def backward_pass(x, y, out_layer_out, hid_layer_out, weights_hid_out, learning_rate):
    m = x.shape[0] # number of training examples

    # Output layer error
    output_layer_error = out_layer_out - y.T

    # Output layer gradients
    d_weights_hidden_output = (1/m) * np.dot(output_layer_error, hid_layer_out.T)
    d_bias_output = (1/m) * np.sum(output_layer_error, axis=1, keepdims=True)

    # Hidden layer error
    hidden_layer_error = np.dot(weights_hid_out.T, output_layer_error) * sigmoid_derivative(hid_layer_out)

    # Hidden layer gradients
    d_weights_input_hidden = (1/m) * np.dot(hidden_layer_error, x)
    d_bias_hidden = (1/m) * np.sum(hidden_layer_error, axis=1, keepdims=True)

    return d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output

# Training function
def train_xor_nn(x, y, hidden_size, learning_rate, epochs):
    input_size = x.shape[1]
    output_size = 1

    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        output_layer_output, hidden_layer_output = forward_pass(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

        # Backward pass
        d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output = backward_pass(x, y, output_layer_output, hidden_layer_output, weights_hidden_output, learning_rate)

        # Update parameters
        weights_input_hidden -= learning_rate * d_weights_input_hidden
        bias_hidden -= learning_rate * d_bias_hidden
        weights_hidden_output -= learning_rate * d_weights_hidden_output
        bias_output -= learning_rate * d_bias_output

        if epoch % 1000 == 0:
            loss = -np.mean(y * np.log(output_layer_output.T) + (1 - y) * np.log(1 - output_layer_output.T))
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def predict_xor_nn(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    output_layer_output, _ = forward_pass(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    predictions = (output_layer_output > 0.5).astype(int)
    return predictions.T

# Train
hidden_size = 4
learning_rate = 0.1
epochs = 10000

weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train_xor_nn(x, y, hidden_size, learning_rate, epochs)

predictions = predict_xor_nn(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print("\nPredictions:")
print(predictions)
print("\nTrue values:")
print(y)

# Evaluate accuracy
accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy * 100}%")

'Setting Bias term 1'
b = 1
wt = np.array([1 ,1])


'===Output==='
print(f'Params: bias:{b}, Weight Mat:{wt}')
print(f'Output XOR function\n', XOR_nn(x,b,wt))
print(f'y true:\n', y)

'Setting Bias term 2'
b = 0
wt = np.array([1 ,1])


'===Output==='
print(f'Params: bias:{b}, Weight Mat:{wt}')
print(f'Output XOR function\n', XOR_nn(x,b,wt))
print(f'y true:\n', y)

'Setting Bias term 3'
b = -1
wt = np.array([1 ,1])


'===Output==='
print(f'Params: bias:{b}, Weight Mat:{wt}')
print(f'Output XOR function\n', XOR_nn(x,b,wt))
print(f'y true:\n', y)
