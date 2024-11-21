import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import precision_score, confusion_matrix


class mlp:
    def __init__(self, seed = False, layers_config: List[dict] = None, epochs = 10000,
            learning_rate = 0.001, regularization_rate = 0.001, verbose = True,
            weights_initializer='heUniform', optimizer='None') -> None:
        self.seed = seed
        self.layers_config = layers_config if layers_config else [
            {'units': 10, 'activation': 'sigmoid'},
            {'units': 10, 'activation': 'sigmoid'},
            {'units': 2, 'activation': 'softmax'}
        ]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.verbose = verbose
        self.weights_initializer = weights_initializer
        self.optimizer=optimizer

        self.weights = None
        self.biases = None
        self.layer_outputs = []
        self.loss_over_epoch = []
        self.validation_loss_over_epoch = []
        self.accuracy_over_epoch = []
        self.validation_accuracy_over_epoch = []


    def initialize(self, input_size: int) -> None:
        """
        Initialize weights and biases for each layer using the specified weights initializer.
        Supports 'heUniform', 'xavierUniform', and default normal initialization.
        Biases are initialized to zeros.
        """
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if self.seed:
            np.random.seed(self.seed)

        self.weights = []
        self.biases = []
        layer_input_size = input_size

        for i, layer in enumerate(self.layers_config):
            units = layer['units']
            activation = layer.get('activation', 'relu')

            if self.weights_initializer == 'heUniform':
                limit = np.sqrt(6 / layer_input_size)
                W = np.random.uniform(-limit, limit, (layer_input_size, units))
            elif self.weights_initializer == 'xavierUniform':
                limit = np.sqrt(6 / (layer_input_size + units))
                W = np.random.uniform(-limit, limit, (layer_input_size, units))
            elif self.weights_initializer == 'heNormal':
                std_dev = np.sqrt(2. / layer_input_size)
                W = np.random.randn(layer_input_size, units) * std_dev
            elif self.weights_initializer == 'xavierNormal':
                std_dev = np.sqrt(2. / (layer_input_size + units))
                W = np.random.randn(layer_input_size, units) * std_dev
            elif self.weights_initializer == 'None':
                W = np.random.randn(layer_input_size, units) * 0.01
            else:
                raise ValueError (f"Unsupported weights_initializer: {self.weights_initializer}, choose between following values: heUniform, xavierUniform, heNormal, xavierNormal, None.")

            b = np.zeros((1, units))

            self.weights.append(W)
            self.biases.append(b)

            layer_input_size = units

        if self.optimizer == 'adam':
            self.m, self.v = self.initialize_adam_cache()
        elif self.optimizer == 'rmsprop':
            self.caches = self.initialize_rmsprop_cache()
        elif self.optimizer == 'nesterov':
            self.velocities = self.initialize_nesterov_cache()


    def sigmoid(self, z):
        return 1 /(1 + np.exp(-z))


    def sigmoid_derivative(self, a):
        return a * (1 - a)
    

    def relu(self, z):
        return np.maximum(0, z)
    

    def relu_derivative(self, z):
        return (z > 0).astype(float)


    def tanh(self, z):
        return np.tanh(z)


    def tanh_derivative(self, a):
        return 1 - np.power(a, 2)


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
            elif activation == 'tanh':
                a = self.tanh(z)
            elif activation == 'softmax':
                a = self.softmax(z)
            else:
                raise ValueError(f"Unsupported activation function: {activation}, choose between following values: relu, sigmoid, tanh and softmax.")

            self.layer_outputs.append({'z': z, 'a': a, 'activation': activation})
            current_output = a

        return current_output


    def backward_propagation(self, X: np.ndarray, y: np.ndarray, yPred: np.ndarray) -> None:
        """
        Perform backward propagation to compute gradients for weights and biases.

        Args:
            X (np.ndarray): Input data of shape (batch_size, num_features).
            y (np.ndarray): True labels of shape (batch_size, ).
            yPred (np.ndarray): Predicted output of shape (batch_size, num_classes).

        Returns:
            List[tuple]: Gradients for weights and biases as a list of tuples [(grad_w, grad_b), ...].
        """
        m = X.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        y = y.astype(int)
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
                elif activation_prev == 'tanh':
                    a_prev = prev_layer_cache['a']
                    delta = np.dot(delta, self.weights[i+1].T) * self.tanh_derivative(a_prev)
                else:
                    raise ValueError(f"Unsupported activation function: {activation_prev}")
    
        gradients = [(grads_W[i], grads_b[i]) for i in range(len(self.weights))]
        
        return gradients


    def gradient_descent(self, gradients, learning_rate) -> None:
        for i in range(len(self.weights)):
            grad_w, grad_b = gradients[i]
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b


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
                gradients = self.backward_propagation(X_batch, y_batch, yPred_batch)

                if self.optimizer == 'adam':
                    self.adam(gradients, self.m, self.v, self.learning_rate, t=epoch)
                elif self.optimizer == 'rmsprop':
                    self.rmsprop(gradients, self.caches, self.learning_rate)
                elif self.optimizer == 'nesterov':
                    self.nesterov_momentum(gradients, self.velocities, self.learning_rate)
                elif self.optimizer == 'None':
                    self.gradient_descent(gradients, self.learning_rate)
                else:
                    raise ValueError (f"Unsupported optimizer: {self.optimizer}, choose between following values: adam, nesterov, rmsprop, None.")

            yPred_train = self.forward_propagation(X_train)
            total_loss = self.loss(yPred_train, y_train)
            self.loss_over_epoch.append(total_loss)

            train_accuracy = np.mean(self.predict(X_train) == y_train)
            self.accuracy_over_epoch.append(train_accuracy)

            if X_valid is not None and y_valid is not None:
                yPred_valid = self.forward_propagation(X_valid)

                val_loss = self.loss(yPred_valid, y_valid)
                self.validation_loss_over_epoch.append(val_loss)

                val_predictions = self.predict(X_valid)
                val_accuracy = np.mean(val_predictions == y_valid)
                self.validation_accuracy_over_epoch.append(val_accuracy)

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

        self.plot_learning_curves()

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


    def nesterov_momentum(self, gradients, velocities, learning_rate, mu=0.9):
        for i in range(len(self.weights)):
            grad_w, grad_b = gradients[i]

            velocities['weights'][i] = mu * velocities['weights'][i] - learning_rate * grad_w
            velocities['biases'][i] = mu * velocities['biases'][i] - learning_rate * grad_b

            self.weights[i] +=velocities['weights'][i]
            self.biases[i] += velocities['biases'][i]

    
    def rmsprop(self, gradients, caches, learning_rate, beta=0.9, epsilon=1e-8):
        for i in range(len(self.weights)):
            grad_w, grad_b = gradients[i]

            caches['weights'][i] = beta * caches['weights'][i] + (1 - beta) * (grad_w ** 2)
            caches['biases'][i] = beta * caches['biases'][i] + (1 - beta) * (grad_b ** 2)

            self.weights[i] -= learning_rate * grad_w / (np.sqrt(caches['weights'][i]) + epsilon)
            self.biases[i] -= learning_rate * grad_b / (np.sqrt(caches['biases'][i]) + epsilon)


    def adam(self, gradients, m, v, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        for i in range(len(self.weights)):
            grad_w, grad_b = gradients[i]

            m['weights'][i] = beta1 * m['weights'][i] + (1 - beta1) * grad_w
            m['biases'][i] = beta1 * m['biases'][i] + (1 - beta1) * grad_b

            v['weights'][i] = beta2 * v['weights'][i] + (1 - beta2) * (grad_w ** 2)
            v['biases'][i] = beta2 * v['biases'][i] + (1 - beta2) * (grad_b ** 2)

            m_hat_w = m['weights'][i] / (1 - beta1 ** t)
            m_hat_b = m['biases'][i] / (1 - beta1 ** t)
            v_hat_w = v['weights'][i] / (1 - beta2 ** t)
            v_hat_b = v['biases'][i] / (1 - beta2 ** t)

            self.weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    def initialize_adam_cache(self):
        """
        Initializes Adam caches for weights and biases.
        Returns two dictionaries (`m` and `v`) for the first and second moments.
        """
        m = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        v = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        return m, v

    def initialize_rmsprop_cache(self):
        """
        Initializes RMSProp caches for weights and biases.
        Returns a dictionary (`caches`) for squared gradients.
        """
        caches = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        return caches

    def initialize_nesterov_cache(self):
        """
        Initializes velocities for weights and biases for Nesterov momentum.
        Returns a dictionary (`velocities`) for velocities.
        """
        velocities = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        return velocities


    def plot_learning_curves(self):
        """
        Plot learning curves for loss and accuracy.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.loss_over_epoch, label='Training Loss')
        if self.validation_loss_over_epoch:
            plt.plot(self.validation_loss_over_epoch, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_over_epoch, label='Training Accuracy')
        if self.validation_accuracy_over_epoch:
            plt.plot(self.validation_accuracy_over_epoch, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def evaluate_model(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the model with additional metrics: precision, recall, F1-score, and confusion matrix.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1-score, and confusion matrix.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        precision = precision_score(y, predictions, average='weighted')
        conf_matrix = confusion_matrix(y, predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "confusion_matrix": conf_matrix
        }