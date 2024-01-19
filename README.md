# Neural Network Project - MNIST Handwritten Digit Classification

This project implements a simple neural network in Python for classifying handwritten digits from the MNIST dataset. It uses numpy for matrix operations, scipy to load dataset, and matplotlib for plotting loss curves.

## Features

- Implementation of a feedforward neural network with two hidden layers.
- Utilization of ReLU (Rectified Linear Unit) activation for hidden layers and Softmax for output layer.
- Cross-entropy loss function for training.
- Basic forward and backward propagation for learning.
- Evaluation function to test the model accuracy on test data.

## Requirements

- numpy
- scipy
- matplotlib

## Usage

1. **Load and preprocess the data:** The data is loaded using scipy's `io.loadmat`, extracted, and normalized.

2. **Define hyperparameters:** Set the learning rate, number of epochs, and other relevant parameters.

3. **Initialize the neural network:** Specify the size of input, hidden, and output layers.

4. **Train the network:** The network is trained using a basic loop with forward and backward propagation.

5. **Evaluate the model:** Test the trained model on test data to evaluate its accuracy.

6. **Plot loss over epochs:** Visualize the training process with a plot of loss vs. epochs.

## Dataset

The project uses the EMNIST-MNIST dataset, which is a set of handwritten digits (0-9).

## Note

- The current implementation does not include mini-batching; it's a basic version using pure gradient descent for learning.
- Users can modify hyperparameters, network architecture, or add new functionalities as needed.

## Example Output

The output includes the loss at each epoch during training and the accuracy of the model on the test dataset.

## Contributions

Contributions to this project are welcome. You can improve upon the network architecture, optimize the code, or add new features.

## License

This project is open-source and available under the MIT License.
