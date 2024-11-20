import numpy as np
import pandas as pd

def prepare_data() -> None:
    """
    Prepares data for a multilayer perceptron model and saves it in .npz format.

    Steps:
        1. Load the dataset from the CSV file located at './datasets/data.csv'.
        2. Check for missing values and drop rows containing NaNs.
        3. Rename columns to include 'id', 'diagnosis', and feature names ('feature0', 'feature1', etc.).
        4. Drop the 'id' column if it exists, as it is not a useful feature.
        5. Encode the 'diagnosis' column manually:
           - 'B' is encoded as 1
           - 'M' is encoded as 0
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

    dataset['diagnosis'] = dataset['diagnosis'].apply(
        lambda x: 1 if x == 'B' else 0
    )

    X = dataset.drop(columns=['diagnosis']).values
    y = dataset['diagnosis'].values

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_scaled = (X - mean) / std

    dataset_combined = np.hstack((X_scaled, y.reshape(-1, 1)))
    np.random.seed(42)
    np.random.shuffle(dataset_combined)

    split_index = int(0.8 * len(dataset_combined))
    train_data = dataset_combined[:split_index]
    test_data = dataset_combined[split_index:]

    X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
    X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

    np.savez("./datasets/data.npz",
             X_train=X_train,
             X_test=X_test,
             y_train=y_train,
             y_test=y_test,
             mean=mean,
             std=std)

    print(f"Data prepared and saved to './datasets/data.npz'.")

if __name__ == "__main__":
    prepare_data()
