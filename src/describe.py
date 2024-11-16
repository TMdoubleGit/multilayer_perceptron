import pandas as pd
import math
import sys

def describe():
    """
    This function loads a CSV file, renames the columns,
    and displays statistics for numerical feature columns.
    """
    try:
        dataset = pd.read_csv("./datasets/data.csv", index_col=0)
        columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(len(dataset.columns) - 2)]
        dataset.columns = columns

        feature_columns = [col for col in dataset.columns if col.startswith('feature')]
        dataset_to_analyse = dataset[feature_columns].ffill()

        stats = []
        for colonne in dataset_to_analyse.columns:
            data = dataset_to_analyse[colonne]
            s_data = sorted(data)
            count = len(data)
            mean = round(data.mean(), 2)
            std = round(data.std(), 2)
            minimum = round(data.min(), 2)
            q25 = round(data.quantile(0.25), 2)
            q50 = round(data.median(), 2)
            q75 = round(data.quantile(0.75), 2)
            maximum = round(data.max(), 2)
            skewness = round(data.skew(), 2)
            kurtosis = round(data.kurtosis(), 2)

            stats.append([colonne, count, mean, std, minimum, q25, q50, q75, maximum, skewness, kurtosis])

        columns = ['Feature', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Exc. kurtosis']
        formatted_res = pd.DataFrame(stats, columns=columns).set_index('Feature').transpose()

        print(formatted_res)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    av = sys.argv
    describe()