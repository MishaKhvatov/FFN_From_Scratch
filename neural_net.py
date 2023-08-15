import numpy as np
import scipy.io as io

def relu(layer):
    return np.maximum(0, layer)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(layer):
    exp_layer = np.exp(layer - np.max(layer))  # for numerical stability
    return exp_layer / exp_layer.sum(axis=0, keepdims=True)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1. - 1e-10)  # Ensure numerical stability
    return -np.sum(y_true * np.log(y_pred))

def forward_propagation(input_layer, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3):
    z_layer1 = np.dot(weights_1, input_layer) + bias_1
    layer_1 = relu(z_layer1)
    
    z_layer2 = np.dot(weights_2, layer_1) + bias_2
    layer_2 = relu(z_layer2)
    
    z_output = np.dot(weights_3, layer_2) + bias_3
    output_probabilities = softmax(z_output)
    
    return layer_1, layer_2, output_probabilities

def backward_propagation(input_layer, true_labels, output_probabilities, layer_1, layer_2, weights_2, weights_3):
    delta_output = output_probabilities - true_labels

    dweights_3 = np.dot(delta_output, layer_2.T)
    dbias_3 = np.sum(delta_output, axis=1, keepdims=True)

    delta_layer2 = np.dot(weights_3.T, delta_output) * relu_derivative(layer_2)
    dweights_2 = np.dot(delta_layer2, layer_1.T)
    dbias_2 = np.sum(delta_layer2, axis=1, keepdims=True)

    delta_layer1 = np.dot(weights_2.T, delta_layer2) * relu_derivative(layer_1)
    dweights_1 = np.dot(delta_layer1, input_layer.T)
    dbias_1 = np.sum(delta_layer1, axis=1, keepdims=True)

    return dweights_1, dbias_1, dweights_2, dbias_2, dweights_3, dbias_3

def hot_encode(value):
    out_vector = np.zeros((10,1))
    out_vector[value, 1] =1;
    return out_vector

# Initialize layers and weights
input_size = 784
hidden_size_1 = 16
hidden_size_2 = 16
output_size = 10

#Initilize Hyper Parameters
BATCH_SIZE = 128


weights_1 = np.random.randn(hidden_size_1, input_size)
bias_1 = np.random.randn(hidden_size_1, 1)
weights_2 = np.random.randn(hidden_size_2, hidden_size_1)
bias_2 = np.random.randn(hidden_size_2, 1)
weights_3 = np.random.randn(output_size, hidden_size_2)
bias_3 = np.random.randn(output_size, 1)

# 1. Load the data
data = io.loadmat("emnist-mnist.mat")

# 2. Extract training and test datasets
train_data = data["dataset"][0][0][0][0][0][0]
train_labels = data["dataset"][0][0][0][0][0][1]
test_data = data["dataset"][0][0][1][0][0][0]
test_labels = data["dataset"][0][0][1][0][0][1]

# 3. Convert to numpy arrays
train_data_np = np.array(train_data)
train_labels_np = np.array(train_labels)
test_data_np = np.array(test_data)
test_labels_np = np.array(test_labels)

#Training Loop
while(True):
    random_indices = np.random.shuffle(np.arange(train_data.shape))
    train_data = train_data[random_indices]
    train_labels = train_labels[random_indices]
    loss_sum = 0;
    gradient_sum = np.zeros(hidden_size_1, input_size), np.zeros(hidden_size_1, 1), np.zeros(hidden_size_2, hidden_size_1), np.zeros(hidden_size_2, 1) , np.zeros(output_size, hidden_size_2), np.zeros((output_size, 1))
    
    #Mini Batch
    for i in range(BATCH_SIZE):
        input_vector = np.array(train_data[i])
        layers = forward_propagation(input_vector , weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)
        gradient_element= backward_propagation(input_vector, hot_encode(train_labels[i]), layers[2], layers[0], layers[1], weights_2, weights_3);
       
        gradient_sum = tuple(a+b for a,b in zip(gradient_sum, gradient_element) )
        loss_sum += cross_entropy(hot_encode(train_labels[i]), layers[2])
    loss = loss/BATCH_SIZE;
    gradient = tuple(x / BATCH_SIZE for x in gradient_sum )
    
    #




