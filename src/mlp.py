import numpy as np
from typing import List

class mlp:
    def __init__(self, seed = False, size: List, epochs: 50, learning_rate = 0.001, regularization_rate = 0.001) -> None:
        self.seed = seed
        self.size = size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.initialize()
        self.weights = None
        self.bias = None
        self.loss_over_epoch = []
        self.test_loss_over_epoch = []

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
        
        self.loss_over_epoch = []
        self.test_loss_over_epoch = []

    def sigmoid(z):
        return 1 /(1 + np.exp(-z))

    def loss(self, yPred: np.ndarray, yTrue: np.ndarray, size: int) -> float:
        cross_entropy_loss = np.mean(-np.log(yPred[np.arange(size), yTrue]))
        reg_loss = 0.5 * self.regularization_rate * sum(
            np.sum(weights**2) for weights in self.weights
        )
        return cross_entropy_loss, reg_loss

    
