import numpy as np
import pandas as pd
from dstz.core.atom import Element
from dstz.core.distribution import Evidence
from scipy.stats import norm
from tqdm import tqdm

def build_mass_distribution(mass_dict):
    """
    Build a mass distribution from a dictionary of mass values.

    Args:
        mass_dict (dict): A dictionary containing mass values for different elements.

    Returns:
        Evidence: An Evidence object representing the mass distribution.
    """
    result = Evidence()
    if not mass_dict:
        return result

    total_mass = sum(mass_dict.values())
    normalized_mass = {key: value / total_mass for key, value in mass_dict.items()}
    sorted_keys = sorted(normalized_mass, key=normalized_mass.get, reverse=True)

    for idx, key in enumerate(sorted_keys):
        result[Element(set(sorted_keys[:idx + 1]))] = normalized_mass[key]

    return result

def convert_data_to_mass(data, gaussian_model):
    """
    Convert input data to mass distributions using a Gaussian model.

    Args:
        data (np.ndarray): Input data to be converted.
        gaussian_model (Gaussianer): A Gaussianer object representing the Gaussian model.

    Returns:
        list: A list of mass distributions for each data point.
    """
    data_mass = []
    feature_indices = gaussian_model.get_feature_indices()
    target_indices = gaussian_model.get_target_indices()
    for data_point in tqdm(data):
        data_point_mass = []
        for feature_index in feature_indices:
            mass = {}
            for target_index in target_indices:
                try:
                    mass_value = gaussian_model.pdf(data_point[feature_index], feature_index, target_index)
                    if mass_value > 0:
                        mass[target_index] = mass_value
                except:
                    continue
            evidence = build_mass_distribution(mass)
            data_point_mass.append(evidence)
        data_mass.append(data_point_mass)
    return data_mass

class Gaussianer:
    def __init__(self, gaussian_df=None):
        """
        Initialize a Gaussianer object.

        Args:
            gaussian_df (pd.DataFrame, optional): A DataFrame containing Gaussian parameters. Defaults to None.
        """
        self.gaussian_df = gaussian_df
        self.feature_indices = None
        self.target_indices = None

    def get_feature_indices(self):
        """
        Get the feature indices.

        Returns:
            list: A list of feature indices.
        """
        return self.feature_indices

    def get_target_indices(self):
        """
        Get the target indices.

        Returns:
            list: A list of target indices.
        """
        return self.target_indices

    def build(self, features, targets):
        """
        Build the Gaussian model from input features and targets.

        Args:
            features (np.ndarray): Input features.
            targets (np.ndarray): Target values.
        """
        data = np.concatenate([features, targets.reshape((-1, 1))], axis=1)
        full_df = pd.DataFrame(data, columns=[i for i in range(features.shape[1])] + ['target'])
        mean_df = full_df.groupby('target').mean().reset_index()
        std_df = full_df.groupby('target').std().reset_index()
        self.feature_indices = [i for i in range(features.shape[1])]
        self.target_indices = list(set(targets))

        for target in mean_df['target']:
            for feature in self.feature_indices:
                mean = mean_df.loc[mean_df['target'] == target, feature].values[0]
                std = std_df.loc[std_df['target'] == target, feature].values[0]

                if self.gaussian_df is not None:
                    self.gaussian_df = pd.concat(
                        [self.gaussian_df,
                         pd.DataFrame({'target': [target], 'feature': [feature], 'mean': [mean], 'std': [std]})],
                        ignore_index=True)
                else:
                    self.gaussian_df = pd.DataFrame(
                        {'target': [target], 'feature': [feature], 'mean': [mean], 'std': [std]})

    def pdf(self, x, feature_index, target_index):
        """
        Calculate the probability density function (PDF) value.

        Args:
            x (float): Input value.
            feature_index (int): Index of the feature.
            target_index (int): Index of the target.

        Returns:
            float: The PDF value.
        """
        params = self.gaussian_df.loc[(self.gaussian_df['feature'] == feature_index) & (self.gaussian_df['target'] == target_index)]
        if params.empty:
            raise ValueError(f"No parameters found for feature {feature_index} and target {target_index}")

        mean = params['mean'].values[0]
        std = params['std'].values[0]

        if std > 0:
            return norm.pdf(x, loc=mean, scale=std)
        else:
            return 1.0 if x == mean else 0.0
