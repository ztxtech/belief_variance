import random

import numpy as np


def set_global_random_seed(seed_value):
    """
    Set the global random seed for both Python's built-in random module and NumPy.

    Args:
        seed_value (int): The seed value to be used for random number generation.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
