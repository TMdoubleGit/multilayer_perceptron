# multilayer_perceptron

42Paris Machine Learning project

A Python implementation of a Multilayer Perceptron (MLP) for solving classification and regression problems. This project provides a flexible and easy-to-use framework to define, train, and evaluate neural networks.

## Features
Support for customizable network architectures.
Configurable activation functions (e.g., sigmoid, ReLU, softmax).
Adjustable hyperparameters such as learning rate, batch size, and epochs.
Evaluation metrics for performance tracking.
Compatible with custom datasets.

## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed on your system.

### Clone the Repository

- git clone git@github.com:TMdoubleGit/multilayer_perceptron.git
- cd multilayer_perceptron

## Usage
### Defining the Network
You can customize your MLP by defining its layers and hyperparameters in the script:

from multilayer_perceptron import model, layers

network = model.createNetwork([
    layers.DenseLayer(input_shape, activation='sigmoid'),
    layers.DenseLayer(24, activation='sigmoid'),
    layers.DenseLayer(output_shape, activation='softmax')
])

### Training the Model
Train the MLP using your dataset:

model.fit(network, data_train, data_valid, 
          loss='binaryCrossentropy', 
          learning_rate=0.0314, 
          batch_size=8, 
          epochs=84)

### Evaluating the Model
After training, evaluate the model's performance:

accuracy = model.evaluate(data_test)
print(f"Test Accuracy: {accuracy:.2f}")
### Examples
1. Binary Classification
For binary classification tasks, use binaryCrossentropy as the loss function and a sigmoid activation in the output layer.

2. Multi-Class Classification
For multi-class problems, use categoricalCrossentropy as the loss function and a softmax activation in the output layer.

## Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository
- Create a new branch (git checkout -b feature/new_feature)
- Commit your changes (git commit -m 'Add new feature')
- Push the branch (git push origin feature/new_feature)
- Create a pull request

## License
This project is licensed under the MIT License.