import numpy as np
from dstz.math.stat.moment import deng_entropy, information_var

def calculate_entropy_weights(mass_distributions):
    """
    Calculate weights based on Deng entropy for each mass distribution.

    Args:
        mass_distributions (list): A list of mass distributions.

    Returns:
        np.ndarray: An array of weights calculated from Deng entropy.
    """
    entropy_values = np.array([deng_entropy(m) for m in mass_distributions])
    weights = entropy_values / entropy_values.sum()
    return weights

def calculate_entropy_variance_weights(mass_distributions):
    """
    Calculate weights based on a combination of Deng entropy and information variance.

    Args:
        mass_distributions (list): A list of mass distributions.

    Returns:
        np.ndarray: An array of weights calculated from the combination of entropy and variance.
    """
    entropy_values = np.array([deng_entropy(m) for m in mass_distributions])
    variance_values = np.array([information_var(m) for m in mass_distributions])
    max_variance = variance_values.max()
    max_entropy = entropy_values.max()
    ratio = 0.8

    combined_score = ratio * entropy_values / max_entropy + (1.0 - ratio) * variance_values / max_variance
    exponential_score = np.exp(-6 * combined_score)
    weights = exponential_score / exponential_score.sum()
    return weights
