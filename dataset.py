# This module defines interfaces and classes implementing the datasets
# 

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data:
    """Dataset interface
    Attributes:
        - self.name: Name of the dataset (string)
        - self.size: Number of samples in the data: (int)
        - self.size_train: Number of samples in the train data: (int)
        - self.size_test: Number of samples in the test data: (int)
        - self.x_train: Input feature of the train data (numpy.array)
        - self.y_train: Labels of the train data (numpy.array)
        - self.x_test: Input feature of the test data (numpy.array)
        - self.y_test: Labels of the test data (numpy.array)
        - self.feature_names: Names of the features in 'x' (list)
        - self.label_names: Names of the labels in 'y' (list)
    Methods:
        __init__(): Load and pre-proecess the dataset
    """

    def __init__(self, name):
        """Constructor
        Parameters:
            name: name of the dataset (string)
        """
        self.name = name
        self.size = None
        self.size_train = None
        self.size_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.feature_names: None
        self.label_names = None

    def get_train(self):
        """Get the train part of the data
        Returns:
            [x_train, y_train]
            x_train: Input feature of train data (numpy.array)
            y_train: Label of the train data (numpy.array)
        """
        return [self.x_train, self.y_train]

    def get_test(self):
        """Get the test part of the data
        Returns:
            [x_test, y_test]
            x_test: Input feature of test data (numpy.array)
            y_test: Label of the test data (numpy.array)
        """
        return [self.x_test, self.y_test]


class RiceData(Data):
    """Rice (Cammeo and Osmancik)
    - This class implements loading and processing of 'Rice (Cammeo and Osmancik)' dataset.
    - URL: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    - Output labels: { b"Cammeo": 0, b"Osmancik": 1 }
    - train-test-split: 80-20 split
    """

    def __init__(self, name="Rice (Cammeo and Osmancik)"):
        """Constructor
        Parameters:
            name: Name of the dataset (string)
        """
        super(RiceData, self).__init__(name=name)
        df = pd.DataFrame(arff.loadarff("./data/rice_cammeo_osmancik.arff")[0])
        self.size = len(df)
        # Get features and labels
        x = df[df.columns.difference(["Class"])].values
        y = df["Class"].map(lambda x: '0' if(x == b'Cammeo') else '1').astype(int).values
        # Standardize the input features
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        # Train - Test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=1693854383, shuffle=True, stratify=y
            )
        self.size_train = len(self.x_train)
        self.size_test = len(self.x_test)
        self.feature_names = list(df.columns.difference(["Class"]))
        self.label_names = [b'Cammeo', b"Osmancik"]


class DataStats:
    """Print basic statistics on the dataset.
    Statistics such as name, size, class imbalance, no. of features, feature names, and label names of the dataset.
    """

    def __call__(self, data):
        """Print statistics of the datasets.
        Parameters:
            data: Dataset object for which to print the statistics (Data)
        """
        print(f"\n{data.name}:\n--------------------------")
        print(f"- Dataset size: {data.size}")
        print(f"- Train dataset size: {data.size_train}")
        for idx, _ in enumerate(data.label_names):
            print(f"\t- {_} label count: {np.sum(data.y_train == idx)}")
        print(f"- Test dataset size: {data.size_test}")
        for idx, _ in enumerate(data.label_names):
            print(f"\t- {_} label count: {np.sum(data.y_test == idx)}")
        print(f"- Number of features: {len(data.feature_names)}")
        print(f"- Feature names: {data.feature_names}")
        print(f"- Label names: {data.label_names}\n")



if __name__ == "__main__":

    # Testing code
    print("\nTesting the dataset implementation:\n====================================")

    # Rice (Cammeo and Osmancik) dataset
    data = RiceData()
    DataStats()(data)
