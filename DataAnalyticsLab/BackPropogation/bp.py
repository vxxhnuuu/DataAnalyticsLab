import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward_propagation(inputs, outputs, hidden_layer_output, output_layer_output, weights_input_hidden, weights_hidden_output, learning_rate):
    output_error = outputs - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate

def train_neural_network(inputs, outputs, hidden_size, epochs, learning_rate):
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output)
        backward_propagation(inputs, outputs, hidden_layer_output, output_layer_output, weights_input_hidden, weights_hidden_output, learning_rate)

        if epoch % 1000 == 0:
            loss = np.mean(np.square(outputs - output_layer_output))
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, weights_hidden_output

# Load data from CSV
file_path = 'DataAnalyticsLab\BackPropogation\Bp.csv'
data = pd.read_csv(file_path)
inputs = data.iloc[:, :-1].values
outputs = data.iloc[:, -1].values.reshape(-1, 1)

# Normalize inputs (optional)
inputs = inputs / np.max(inputs, axis=0)

# Set hyperparameters
hidden_size = 4
epochs = 10000
learning_rate = 0.1

# Train the neural network
weights_input_hidden, weights_hidden_output = train_neural_network(inputs, outputs, hidden_size, epochs, learning_rate)
