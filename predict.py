import numpy as np
import pickle
from src.mlp import mlp
import pandas as pd

def load_model(model_path):
    """
    Load the trained model from the specified pickle file.

    Args:
        model_path (str): Path to the pickle file containing the trained model.

    Returns:
        model: The loaded mlp model instance.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_test_data(data_path):
    """
    Load test data from the specified .npz file.

    Args:
        data_path (str): Path to the .npz data file.

    Returns:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels (if available).
    """
    data = np.load(data_path)
    X_test = data['X_test']
    y_test = data['y_test'].astype(int)
    return X_test, y_test


def make_predictions(model, X):
    """
    Use the trained model to make predictions on the input data.

    Args:
        model: The trained mlp model instance.
        X (np.ndarray): Input features for prediction.

    Returns:
        predictions (np.ndarray): Predicted class labels.
    """
    predictions = model.predict(X)
    return predictions


def evaluate_model(predictions, y_true):
    """
    Evaluate the model's predictions against the true labels.

    Args:
        predictions (np.ndarray): Predicted class labels.
        y_true (np.ndarray): True class labels.

    Returns:
        accuracy (float): The accuracy of the model.
    """
    accuracy = np.mean(predictions == y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.

    Args:
        predictions (np.ndarray): Predicted class labels.
        output_path (str): Path to save the predictions.

    Returns:
        None
    """
    np.savetxt(output_path, predictions, delimiter=',', fmt='%d')
    print(f"Predictions saved to '{output_path}'.")


def main():
    try:
        model_path = './models/trained_model.pkl'

        model = load_model(model_path)

        data_path = './datasets/data.npz'
        X_test, y_test = load_test_data(data_path)

        predictions = make_predictions(model, X_test)

        if y_test is not None:
            evaluate_model(predictions, y_test)

        print("Predictions:")
        print(predictions)
        print("True")
        print(y_test)

        output_path = './predictions.csv'
        save_predictions(predictions, output_path)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()