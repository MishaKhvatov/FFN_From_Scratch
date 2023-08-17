import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        sizes = [input_size, hidden_size_1, hidden_size_2, output_size]

        # Using the notation w_ij: weight from neuron i in layer L to neuron j in L+1
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2. / sizes[i]) for i in range(len(sizes)-1)]
        self.biases = [np.zeros((size, 1)) for size in sizes[1:]]
        self.layers = [np.zeros((size, 1)) for size in sizes]

    def relu(self, layer_vector):
        return np.maximum(0, layer_vector)

    def relu_derivative(self, layer_vector):
        return np.where(layer_vector > 0, 1, 0)

    def softmax(self,x):
        # Compute the max value for each sample
        x_max = np.max(x, axis=0, keepdims=True)
        # Subtract the max (for numerical stability)
        x = x - x_max
        # Compute the stable exponential values
        e_x = np.exp(x)
        # Sum the exponentials
        sum_e_x = e_x.sum(axis=0, keepdims=True)
        return e_x / sum_e_x

    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1. - 1e-10)  # Ensure numerical stability
        return -np.sum(y_true * np.log(y_pred))

    def forward_propagation(self, input_data):
       # print(f"Input_data:\n {input_data} \n")
        self.layers[0] = input_data

        for i in range(1, len(self.layers)):
            z = np.dot(self.weights[i-1].T , self.layers[i-1]) + self.biases[i-1]
            
            if i != len(self.layers) - 1:
                self.layers[i] = self.relu(z)
            else:
                self.layers[i] = self.softmax(z) 
        
    def backward_propagation(self, true_vector):        #Initilize adjustment arrays
            
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

# 1. Load the data
data = io.loadmat("emnist-mnist.mat")

# 2. Extract training and test datasets
train_data = data["dataset"][0][0][0][0][0][0]
train_labels = data["dataset"][0][0][0][0][0][1]
test_data = data["dataset"][0][0][1][0][0][0]
test_labels = data["dataset"][0][0][1][0][0][1]

# 3. Convert to numpy arrays
train_data_np = np.array(train_data)     #Each row is a training example
train_labels_np = np.array(train_labels)
test_data_np = np.array(test_data)
test_labels_np = np.array(test_labels)

# 4. Normalize pixel values
train_data_np = train_data_np / 255.0
test_data_np = test_data_np / 255.0

# 5. Hyperparameters
learning_rate = 0.001
epochs = 10
#batch_size = 64

# 6. Convert labels to one-hot vectors
num_classes = 10
train_labels_one_hot = np.eye(num_classes)[train_labels_np.reshape(-1)]
test_labels_one_hot = np.eye(num_classes)[test_labels_np.reshape(-1)]

# 7. Initialize neural network
input_size = train_data_np.shape[1]  # 784 for MNIST
hidden_size_1 = 16
hidden_size_2 = 16
output_size = num_classes

nn = NeuralNetwork(input_size, hidden_size_1, hidden_size_2, output_size)

# TODO: Implement Mini-Batching, currently it's pure gradient descent

# 8. Training loop
losses = []
for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(train_data_np)):
        
        #Making data a column 
        data = train_data_np[i].reshape(784,1)
        label = train_labels_one_hot[i].reshape(10,1)
        
        # Forward pass
        nn.forward_propagation(data)
        
        # Compute loss
        loss = nn.cross_entropy_loss(label, nn.layers[-1])
        total_loss += loss

        # Backward pass and get the gradients
        d_weights, d_biases = nn.backward_propagation(label)
        
        # Update weights and biases
        for j in range(len(nn.weights)):    
            nn.weights[j] -= learning_rate * d_weights[j]
            nn.biases[j] -= learning_rate * d_biases[j]

    avg_loss = total_loss / len(train_data_np)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_model(nn, test_data, test_labels):
    correct_predictions = 0
    total_predictions = len(test_data)

    for i in range(total_predictions):
        data = test_data[i].reshape(784,1)
        
        # Forward pass to get the predictions
        nn.forward_propagation(data)
        predicted_output = nn.layers[-1]
        
        # Convert predicted probabilities to label by getting the index of max probability
        predicted_label = np.argmax(predicted_output)
        true_label = np.argmax(test_labels[i])
        
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Evaluate the model on test data
accuracy = evaluate_model(nn, test_data_np, test_labels_one_hot)
print(f"Accuracy on test data: {accuracy:.2f}%")


# 9. Plot loss over epochs
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()