import os

import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def filter_columns(dataframe):
    """
    Filter columns from a DataFrame based on whether they are numeric and have a non-negligible standard deviation.

    Args:
        dataframe (pd.DataFrame): The input DataFrame to be filtered.

    Returns:
        pd.DataFrame: A new DataFrame containing only the columns that meet the criteria.
    """
    result_dataframe = pd.DataFrame()
    for column in dataframe.columns:
        if pd.to_numeric(dataframe[column], errors='coerce').notnull().all():
            standard_deviation = dataframe[column].std(ddof=0)
            if standard_deviation > 1e-10:
                result_dataframe[column] = dataframe[column]
    return result_dataframe


class UCIDataset:
    def __init__(self, dataset_id, cache_path="./cache", test_ratio=0.2, random_seed=42):
        """
        Initialize a UCIDataset object.

        Args:
            dataset_id (int): The ID of the UCI dataset to load.
            cache_path (str): The path to the cache directory.
            test_ratio (float): The ratio of test data for splitting.
            random_seed (int): The random seed for reproducibility.
        """
        self.cache_path = cache_path
        self.dataset_id = dataset_id
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.pickle_filepath = os.path.join(cache_path, f'{dataset_id}.pkl')

        if os.path.exists(self.pickle_filepath):
            self.load(self.pickle_filepath)
            print(f'UCI Dataset ID:{dataset_id} Loaded from {self.pickle_filepath}.')
        else:
            dataset = fetch_ucirepo(id=dataset_id)
            print(f'UCI Dataset ID:{dataset_id} Download Successfully')
            self.raw_features = filter_columns(dataset.data.features)
            self.raw_targets = dataset.data.targets

            self.feature_names = self.raw_features.columns
            self.feature_count = len(self.feature_names)
            self.feature_indices = list(range(self.feature_count))
            self.features = self.raw_features.values

            self.targets = self.raw_targets.values.reshape(-1)
            self.unique_targets = list(set(self.targets))
            self.target_count = len(self.unique_targets)
            self.target_mapping = {target: idx for idx, target in enumerate(self.unique_targets)}
            self.save(self.pickle_filepath)

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and the target of the sample.
        """
        return self.features[index, :], self.targets[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.features)

    def __repr__(self):
        """
        Return a string representation of the UCIDataset object.

        Returns:
            str: A string representation of the object.
        """
        return f'UCIDataset(id={self.dataset_id})'

    def save(self, filepath):
        """
        Save the dataset object to a file using dill.

        Args:
            filepath (str): The path to the file where the object will be saved.
        """
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    def load(self, filepath):
        """
        Load the dataset object from a file using dill.

        Args:
            filepath (str): The path to the file from which the object will be loaded.
        """
        with open(filepath, 'rb') as file:
            loaded_dataset = dill.load(file)
        self.__dict__.update(loaded_dataset.__dict__)

    def train_test_split(self, test_ratio, random_seed):
        """
        Split the dataset into training and test sets.

        Args:
            test_ratio (float): The ratio of test data.
            random_seed (int): The random seed for reproducibility.

        Returns:
            tuple: A tuple containing the training features, test features, training targets, and test targets.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=test_ratio, random_state=random_seed
        )
        y_train = np.array([self.target_mapping[y] for y in y_train])
        y_test = np.array([self.target_mapping[y] for y in y_test])
        return x_train, x_test, y_train, y_test
