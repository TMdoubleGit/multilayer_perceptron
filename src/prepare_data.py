import numpy as np
import pandas as pandas
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Prepares data for a multilayer perceptron model and saves it in .npz format.

    Steps:
        1. Load the data from the specified CSV file.
        2. Check and handle missing values by dropping rows with NaNs.
        3. Drop the 'id' column (if it exists) as it is not a useful feature.
        4. Encode the 'diagnosis' column using Label Encoding to convert it into numerical values.
        5. Separate features (X) and target labels (y).
        6. Normalize the features using StandardScaler for better performance in neural networks.
        7. Split the dataset into training and testing sets with an 80-20 ratio.
        8. Save the prepared data (X_train, X_test, y_train, y_test) in .npz format.

    Returns:
        None: The function saves the processed data to the specified .npz file and prints a success message.
    """

    dataset = pd.read_csv("/datasets/data.csv")

    if data.isnull().sum().any():
        data = data.dropna()

    columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(len(dataset.columns) - 2)]
    dataset.columns = columns


    if 'id' in dataset.columns:
        dataset = dataset.drop(columns=['id'])
    
    if 'diagnosis' in data.columns:
        data['diagnosis'] = data['diagnosis'].apply(
            lambda x: [1, 0] if x == 'B' else [0, 1]
        )

    X = data.drop(columns=['diagnosis']).values
    y = np.array(data['diagnosis'].tolist())

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