import numpy as np
import pickle
import os
import argparse
from src.mlp import mlp

def load_data(data_path):
    """
    Load data from the specified .npz file.

    Args:
        data_path (str): Path to the .npz data file.

    Returns:
        X_train, y_train, X_valid, y_valid: Arrays containing the training and validation data.
    """
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_test']
    y_valid = data['y_test']
    return X_train, y_train, X_valid, y_valid


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an MLP model.')
    parser.add_argument('--layers', type=int, nargs='+', default=[10, 10],
                        help='List of units in each hidden layer.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs.')
    parser.add_argument('--loss', type=str, default='binaryCrossentropy', help='Loss function.')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of each mini-batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--activation', type=str, default='sigmoid', help='Activation function.')
    parser.add_argument('--weights_initializer', type=str, default='heUniform', help='Weights initializer.')
    parser.add_argument('--optimizer', type=str, default='None', help='Gradient descent optimizer')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    return parser.parse_args()


def initialize_model(args, input_units):
    layers_config = []
    output_units = 2

    layers_config.append({'units': input_units, 'activation': args.activation})

    for units in args.layers:
        layers_config.append({'units': units, 'activation': args.activation})

    layers_config.append({'units': output_units, 'activation': 'softmax'})

    model = mlp(
        seed=42,
        layers_config=layers_config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        regularization_rate=0.001,
        verbose=True,
        weights_initializer=args.weights_initializer,
        optimizer=args.optimizer
    )
    return model



def train_model(model, X_train, y_train, X_valid, y_valid, args):
    """
    Train the MLP model using the provided training data.

    Args:
        model: The mlp model instance.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_valid (np.ndarray): Validation features.
        y_valid (np.ndarray): Validation labels.

    Returns:
        model: The trained model.
    """
    model.train(
        X_train,
        y_train,
        X_valid,
        y_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )
    return model


def save_model(model, filepath):
    """
    Save the trained model to a file using pickle.

    Args:
        model: The trained mlp model instance.
        filepath (str): Path to save the pickle file.

    Returns:
        None
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to '{filepath}'.")


def main():
    try:
        args = parse_arguments()

        data_path = './datasets/data.npz'
        X_train, y_train, X_valid, y_valid = load_data(data_path)

        model = initialize_model(args, input_units = X_train.shape[1])

        model = train_model(model, X_train, y_train, X_valid, y_valid, args)

        model_path = './models/trained_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        save_model(model, model_path)

        metrics = model.evaluate_model(X_valid, y_valid)
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            if key == "confusion_matrix":
                print(f"{key}:\n{value}")
            else:
                print(f"{key}: {value:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
