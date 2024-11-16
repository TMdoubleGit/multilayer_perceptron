import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Prepares data for a multilayer perceptron model and saves it in .npz format.

    Steps:
        1. Load the dataset from the CSV file located at './datasets/data.csv'.
        2. Check for missing values and drop rows containing NaNs.
        3. Rename columns to include 'id', 'diagnosis', and feature names ('feature0', 'feature1', etc.).
        4. Drop the 'id' column if it exists, as it is not a useful feature.
        5. Encode the 'diagnosis' column manually using one-hot encoding:
           - 'B' is encoded as [1, 0]
           - 'M' is encoded as [0, 1]
        6. Separate the features (X) and target labels (y).
        7. Normalize the features (X) by subtracting the mean and dividing by the standard deviation for each feature.
        8. Combine the features and labels into a single dataset and shuffle it using a fixed random seed for reproducibility.
        9. Split the shuffled dataset into 80% training data and 20% testing data.
        10. Save the prepared data (X_train, X_test, y_train, y_test) into a .npz file at './datasets/data.npz'.

    Returns:
        None: The function saves the processed data into the specified .npz file and prints a success message.
    """

    dataset = pd.read_csv("./datasets/data.csv")

    if dataset.isnull().sum().any():
        dataset = dataset.dropna()

    columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(len(dataset.columns) - 2)]
    dataset.columns = columns


    if 'id' in dataset.columns:
        dataset = dataset.drop(columns=['id'])
    
    if 'diagnosis' in dataset.columns:
        dataset['diagnosis'] = dataset['diagnosis'].apply(
            lambda x: [1, 0] if x == 'B' else [0, 1]
        )

    X = dataset.drop(columns=['diagnosis']).values
    y = np.array(dataset['diagnosis'].tolist())

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std

    dataset = np.hstack((X_scaled, y))
    np.random.seed(42)
    np.random.shuffle(dataset)

    split_index = int(0.8 * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    X_train, y_train = train_data[:, :-y.shape[1]], train_data[:, -y.shape[1]:]
    X_test, y_test = test_data[:, :-y.shape[1]], test_data[:, -y.shape[1]:]

    np.savez("./datasets/data.npz", 
             X_train=X_train, 
             X_test=X_test, 
             y_train=y_train, 
             y_test=y_test)

    print(f"Data prepared and saved to './datasets/data.npz'.")

if __name__ == "__main__":
    prepare_data()