import numpy as np
from typing import List

class mlp:
    def __init__(self, seed = False, size: List, epochs: 50, learning_rate = 0.001, regularization_rate = 0.001, verbose = True) -> None:
        self.seed = seed
        self.size = size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.verbose = verbose

        self.weights = None
        self.biases = None
        self.layer_outputs = []
        self.loss_over_epoch = []

        if size:
            self.initialize()


    def initialize(self) -> None:
        """
        Initialize weights and biases for each layer.
        Weights are initialized using He initialization: N(0, sqrt(2 / size)).
        Biases are initialized to 0.
        """
        if self.seed:
            np.random.seed(self.seed)
            
        self.weights = [np.random.randn(self.size[i - 1], self.size[i]) * np.sqrt(2. / self.size[i])
                        for i in range(1, len(self.size))]
        self.biases = [np.zeros((1, self.size[i])) for i in range(1, len(self.size))]


    def sigmoid(z):
        return 1 /(1 + np.exp(-z))


    def loss(self, yPred: np.ndarray, yTrue: np.ndarray, size: int) -> float:
        cross_entropy_loss = np.mean(-np.log(yPred[np.arange(size), yTrue]))
        reg_loss = 0.5 * self.regularization_rate * sum(
            np.sum(weights**2) for weights in self.weights
        )
        return cross_entropy_loss, reg_loss


    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Args:
            X (np.ndarray): Input data of shape (batch_size, num_features).
        
        Returns:
            np.ndarray: The output of the network after forward propagation.
        """
        self.layer_outputs =[]
        current_output = X

        for i in range(len(self.weights) - 1):
            z = np.dot(current_output, self.weights[i] + self.biases[i])
            current_output = self.sigmoid(z)
            self.layer_outputs.append(current_output)

        z = np.dot(current_output, self.weights[-1] + self.biases[-1])
        output = self.sigmoid(z)
        self.layer_outputs.append(output)

        return output


    def backward_propagation(self, X: np.ndarray, y: np.ndarray, yPred: np.ndarray) -> None:
        """
        Perform backward propagation to update weights and biases.
        
        Args:
            X (np.ndarray): Input data of shape (batch_size, num_features).
            y (np.ndarray): True labels of shape (batch_size, 1).
            yPred (np.ndarray): Predicted output of shape (batch_size, 1).
        """
        m = X.shape[0]
        gradients = []
        error = yPred - y

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.layer_outputs[i - 1].T if i > 0 else X.T, error) / m
            dB = np.sum(error, axis=0, keepdims=True) / m

            dW += self.regularization_rate * self.weights[i] / m

            gradients.append((dW, dB))

            if i > 0:
                error = np.dot(error, self.weights[i].T) * (self.layer_outputs[i - 1] * (1 - self.layer_outputs[i - 1]))

         for i in range(len(self.weights)):
            dW, dB = gradients[-(i + 1)]
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB


    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = None, patience: int = 10) -> None:
        """
        Train the neural network using forward and backward propagation.
        
        Args:
            X (np.ndarray): Input data of shape (num_samples, num_features).
            y (np.ndarray): True labels of shape (num_samples, 1).
            epochs (int, optional): Number of training epochs. Defaults to self.epochs.
        """
        epochs = epochs or self.epochs
        best_loss = float('inf')
        no_improvement_count = 0


        for epoch in range(epochs):

            yPred = self.forward_propagation(X)

            data_loss, reg_loss = self.loss(yPred, y, len(y))
            total_loss = data_loss + reg_loss

            self.loss_over_epoch.append(total_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best loss: {best_loss:.2f}")
                break

            self.backward_propagation(X, y, yPred)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}: Loss = {total_loss:.2f}")
