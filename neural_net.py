import numpy as np
import scipy.io as io


import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        sizes = [input_size, hidden_size_1, hidden_size_2, output_size]

        # Using the notation w_ij: weight from neuron i in layer L to neuron j in L+1
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.layers = [np.zeros((size, 1)) for size in sizes]

    def relu(self, layer_vector):
        return np.maximum(0, layer_vector)

    def relu_derivative(self, layer_vector):
        return np.where(layer_vector > 0, 1, 0)

    def softmax(self, layer):
        exp_layer = np.exp(layer - np.max(layer))  # for numerical stability
        return exp_layer / exp_layer.sum(axis=0, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1. - 1e-10)  # Ensure numerical stability
        return -np.sum(y_true * np.log(y_pred))

    def forward_propagation(self, input_data):
        self.layers[0] = input_data
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i-1].T, self.weights[i-1]) + self.biases[i-1]
            if i != len(self.layers) - 1:
                self.layers[i] = self.relu(z)
            else:
                self.layers[i] = self.softmax(z)

    def backward_propagation(self, true_vector):
        
        #Initilize adjustment arrays
        
        d_layers = [np.zeros(layer.shape) for layer in self.layers]
        d_weights = [np.zeros(w.shape) for w in self.weights]
        d_biases = [np.zeros(b.shape) for b in self.biases]

        d_layers[-1] = self.layers[-1] - true_vector
        for l in range(len(self.layers)-1, 0, -1):
            # Compute gradients for weights and biases
            d_weights[l-1] = np.dot(self.layers[l-1], d_layers[l].T)
            d_biases[l-1] = np.sum(d_layers[l], axis=1, keepdims=True)
            
            if l != 1:  # No need to compute d_layers[0] as it's the input layer
                d_layers[l-1] = np.dot(self.weights[l-1], d_layers[l]) * self.relu_derivative(self.layers[l-1])

        return d_weights, d_biases

    



# # 1. Load the data
# data = io.loadmat("emnist-mnist.mat")

# # 2. Extract training and test datasets
# train_data = data["dataset"][0][0][0][0][0][0]
# train_labels = data["dataset"][0][0][0][0][0][1]
# test_data = data["dataset"][0][0][1][0][0][0]
# test_labels = data["dataset"][0][0][1][0][0][1]

# # 3. Convert to numpy arrays
# train_data_np = np.array(train_data)
# train_labels_np = np.array(train_labels)
# test_data_np = np.array(test_data)
# test_labels_np = np.array(test_labels)


# LEARNING_RATE = 0.01  

