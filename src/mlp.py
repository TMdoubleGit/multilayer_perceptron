import numpy as np
from typing import List

class mlp:
    def __init__(self, seed = False, layers_config: List[dict] = None, epochs = 50, learning_rate = 0.001, regularization_rate = 0.001, verbose = True) -> None:
        self.seed = seed
        self.layers_config = layers_config if layers_config else [
            {'units': 24, 'activation': 'relu'},
            {'units': 24, 'activation': 'relu'},
            {'units': 2, 'activation': 'softmax'}
        ]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.verbose = verbose

        self.weights = None
        self.biases = None
        self.layer_outputs = []
        self.loss_over_epoch = []
        self.validation_loss_over_epoch = []


    def initialize(self, input_size: int) -> None:
        """
        Initialize weights and biases for each layer.
        Weights are initialized using He initialization: N(0, sqrt(2 / size)).
        Biases are initialized to 0.
        """
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if self.seed:
            np.random.seed(self.seed)

        layer_sizes = [input_size] + [layer['units'] for layer in self.layers_config]
            
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [
            np.zeros((1, layer_sizes[i +1]))
            for i in range(len(layer_sizes) -1)
        ]


    def sigmoid(self, z):
        return 1 /(1 + np.exp(-z))


    def sigmoid_derivative(self, a):
        return a * (1 - a)
    

    def relu(self, z):
        return np.maximum(0, z)
    

    def relu_derivative(self, z):
        return (z > 0).astype(float)


    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    def loss(self, yPred: np.ndarray, yTrue: np.ndarray) -> float:
        """
        Compute the total loss (data loss + regularization loss).

        Args:
            yPred (np.ndarray): Predicted probabilities from the model.
            yTrue (np.ndarray): True class labels.

        Returns:
            float: Total loss.
        """
        m = yTrue.shape[0]

        y_one_hot = np.zeros_like(yPred)
        y_one_hot[np.arange(m), yTrue] = 1

        data_loss = -np.sum(y_one_hot * np.log(yPred + 1e-8)) / m

        reg_loss = 0.5 * self.regularization_rate * sum(
            np.sum(weights**2) for weights in self.weights
        )

        total_loss = data_loss + reg_loss
        return total_loss


    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Args:
            X (np.ndarray): Input data of shape (batch_size, num_features).

        Returns:
            np.ndarray: The output of the network after forward propagation.
        """
        self.layer_outputs = []
        current_output = X

        for i, layer in enumerate(self.layers_config):
            z = np.dot(current_output, self.weights[i]) + self.biases[i]
            activation = layer.get('activation', 'relu')
            if activation == 'relu':
                a = self.relu(z)
            elif activation == 'sigmoid':
                a = self.sigmoid(z)
            elif activation == 'softmax':
                a = self.softmax(z)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            self.layer_outputs.append({'z': z, 'a': a, 'activation': activation})
            current_output = a

        return current_output


    def backward_propagation(self, X: np.ndarray, y: np.ndarray, yPred: np.ndarray) -> None:
        """
        Perform backward propagation to update weights and biases.

        Args:
            X (np.ndarray): Input data of shape (batch_size, num_features).
            y (np.ndarray): True labels of shape (batch_size, ).
            yPred (np.ndarray): Predicted output of shape (batch_size, num_classes).
        """
        m = X.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        y = y.astype(int)  # Ensure y is integer type
        y_one_hot = np.zeros_like(yPred)
        y_one_hot[np.arange(m), y] = 1

        delta = yPred - y_one_hot

        for i in reversed(range(len(self.weights))):
            layer_cache = self.layer_outputs[i]
            z = layer_cache['z']
            a = layer_cache['a']
            activation = layer_cache['activation']
            a_prev = self.layer_outputs[i - 1]['a'] if i > 0 else X

            dW = np.dot(a_prev.T, delta) / m + (self.regularization_rate * self.weights[i] / m)
            dB = np.sum(delta, axis=0, keepdims=True) / m

            grads_W[i] = dW
            grads_b[i] = dB

            if i > 0:
                prev_layer_cache = self.layer_outputs[i - 1]
                z_prev = prev_layer_cache['z']
                activation_prev = prev_layer_cache['activation']

                if activation_prev == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_prev)
                elif activation_prev == 'sigmoid':
                    a_prev = prev_layer_cache['a']
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(a_prev)
                else:
                    raise ValueError(f"Unsupported activation function: {activation_prev}")

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_W[i]
            self.biases[i] -= self.learning_rate * grads_b[i]


    def train(self, X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray = None, y_valid: np.ndarray = None,
            epochs: int = None, batch_size: int = 50, patience: int = 50) -> None:
        """
        Train the neural network using mini-batch gradient descent.

        Args:
            X_train (np.ndarray): Training input data of shape (num_samples, num_features).
            y_train (np.ndarray): Training true labels of shape (num_samples, ).
            X_valid (np.ndarray, optional): Validation input data.
            y_valid (np.ndarray, optional): Validation true labels.
            epochs (int, optional): Number of training epochs. Defaults to self.epochs.
            batch_size (int, optional): Size of each mini-batch. Defaults to 32.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped.
        """
        epochs = epochs or self.epochs
        best_loss = float('inf')
        no_improvement_count = 0
   
        input_size = X_train.shape[1]
        self.initialize(input_size)

        num_samples= X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for epoch in range(1, epochs + 1):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                yPred_batch = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, yPred_batch)

            yPred_train = self.forward_propagation(X_train)
            total_loss = self.loss(yPred_train, y_train)
            self.loss_over_epoch.append(total_loss)

            if X_valid is not None and y_valid is not None:
                yPred_valid = self.forward_propagation(X_valid)

                val_loss = self.loss(yPred_valid, y_valid)
                self.validation_loss_over_epoch.append(val_loss)

                val_predictions = self.predict(X_valid)
                val_accuracy = np.mean(val_predictions == y_valid)

                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1                              

                    if no_improvement_count >= patience:
                        print(f"Early stopping triggered after {epoch} epochs. Best val_loss: {best_loss:.4f}")
                        break

                if self.verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}: Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
            else:
                if total_loss < best_loss:
                    best_loss = total_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f"Early stopping triggered after {epoch} epochs. Best loss: {best_loss:.4f}")
                    break

                if self.verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}/{epochs}: Loss = {total_loss:.4f}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Args:
            X (np.ndarray): Input data of shape (num_samples, num_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        probabilities = self.forward_propagation(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions