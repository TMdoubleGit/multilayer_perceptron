import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from src.mlp import mlp

def load_data(datapath):
    """
    Load data from the specified .npz file.

    Args:
        datapath (str): Path to the .npz data file.

    Returns:
        X_train, y_train, X_valid, y_valid: Arrays containing the training and validation data.
    """
    data = np.load(datapath)
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_test']
    y_valid = data['y_test']
    return X_train, y_train, X_valid, y_valid


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an MLP model.')
    parser.add_argument('--layers', type=int, nargs='+', default=[10, 10],
                        help='List of units in each hidden layer.')
    parser.add_argument("datapath", type=str,
                        help='Path to the .npz data file.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs.')
    parser.add_argument('--loss', type=str, default='binaryCrossentropy', help='Loss function.')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of each mini-batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--activation', type=str, default='sigmoid', help='Activation function.')
    parser.add_argument('--weights_initializer', type=str, default='heUniform', help='Weights initializer.')
    parser.add_argument('--optimizers', type=str, nargs='+', default=['None', 'nesterov', 'adam', 'rmsprop'], help='List of optimizers to use during training')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    return parser.parse_args()


def initialize_model(args, input_units, optimizer):
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
        optimizer=optimizer
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


def plot_multiple_learning_curves(models_results: dict):
    """
    Plot learning curves for multiple models.

    Args:
        models_results (dict): Dictionary where keys are model names and values are dictionaries with:
                            - 'train_loss': List of training loss values over epochs.
                            - 'valid_loss': List of validation loss values over epochs.
                            - 'train_acc': List of training accuracy values over epochs.
                            - 'valid_acc': List of validation accuracy values over epochs.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for model_name, results in models_results.items():
        plt.plot(results['loss'], label=f'{model_name} Train Loss')
        plt.plot(results['val_loss'], label=f'{model_name} Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for model_name, results in models_results.items():
        plt.plot(results['acc'], label=f'{model_name} Train Accuracy')
        plt.plot(results['val_acc'], label=f'{model_name} Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

        
def main():
    try:
        args = parse_arguments()

        datapath = args.datapath
        X_train, y_train, X_valid, y_valid = load_data(datapath)

        input_units = X_train.shape[1]

        models = {}
        learning_curves = {}

        for optimizer in args.optimizers:
            print(f"Training with optimizer: {optimizer}")
            model = initialize_model(args, input_units, optimizer)
            if ({optimizer} == 'None'):
                model_name = "gradient_descent"
            else:
                model_name = f"{optimizer}"
            trained_model = train_model(model, X_train, y_train, X_valid, y_valid, args)

            model_path = f'./models/{model_name}.pkl'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_model(trained_model, model_path)

            models[model_name] = trained_model
            learning_curves[model_name] = {
                'loss': trained_model.loss_over_epoch,
                'val_loss': trained_model.validation_loss_over_epoch,
                'acc':trained_model.accuracy_over_epoch,
                'val_acc':trained_model.validation_accuracy_over_epoch
            }

            metrics = trained_model.evaluate_model(X_valid, y_valid)
            print(f"Metrics for {model_name}:")
            for key, value in metrics.items():
                if key == "confusion_matrix":
                    print(f"{key}:\n{value}")
                else:
                    print(f"{key}: {value:.4f}")   
        plot_multiple_learning_curves(learning_curves)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
