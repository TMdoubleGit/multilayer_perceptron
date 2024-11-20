import numpy as np
import pickle
import os
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

def initialize_model():
    """
    Initialize the MLP model with specified parameters.

    Returns:
        model: An instance of the mlp class.
    """
    model = mlp(
        seed=42,
        layers_config=[
            {'units': 10, 'activation': 'sigmoid'},
            {'units': 10, 'activation': 'sigmoid'},
            {'units': 2, 'activation': 'softmax'}
        ],
        epochs=10000,
        learning_rate=0.001,
        regularization_rate=0.001,
        verbose=True
    )
    return model

def train_model(model, X_train, y_train, X_valid, y_valid):
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
    model.train(X_train, y_train, X_valid, y_valid)
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
    data_path = './datasets/data.npz'
    X_train, y_train, X_valid, y_valid = load_data(data_path)

    model = initialize_model()

    model = train_model(model, X_train, y_train, X_valid, y_valid)

    model_path = './models/trained_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    save_model(model, model_path)

if __name__ == "__main__":
    main()
