from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from src.data import UCIDataset
from src.fusionf import murphy
from src.gaussian import Gaussianer, convert_data_to_mass
from src.utils import set_global_random_seed
from src.weightf import calculate_entropy_variance_weights, calculate_entropy_weights


# Function to run the experiment with given dataset, seed, weight function, fusion function, and test ratio
def run_experiment(dataset, seed, weight_function, fusion_function, test_ratio):
    """
    Run the experiment with given dataset, seed, weight function, fusion function, and test ratio.

    Args:
        dataset (UCIDataset): The dataset to use for the experiment.
        seed (int): The random seed for reproducibility.
        weight_function (function): The weight function to calculate weights.
        fusion_function (function): The fusion function to perform data fusion.
        test_ratio (float): The ratio of test data.

    Returns:
        tuple: A tuple containing dataset ID, seed, weight function name, and accuracy.
    """
    print(f"{dataset.dataset_id} | {seed} | {weight_function.__name__}")
    set_global_random_seed(seed)
    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = dataset.train_test_split(test_ratio, seed)
    # Initialize Gaussianer object
    gaussian_model = Gaussianer()
    # Build the Gaussian model using training data
    gaussian_model.build(x_train, y_train)
    # Convert test data to mass data
    test_mass_data = convert_data_to_mass(x_test, gaussian_model)
    results = []
    for mass in test_mass_data:
        # Calculate weights using the weight function
        weights = weight_function(mass)
        # Perform data fusion using the fusion function
        current_result = fusion_function(mass, weights)
        results.append(current_result)
    results = np.array(results)
    # Calculate the accuracy of the results
    accuracy = accuracy_score(y_test, results)
    print(f"{dataset.dataset_id} | {seed} | {weight_function.__name__} | {accuracy}")
    return dataset.dataset_id, seed, weight_function.__name__, f"{float(accuracy):.4f}"

if __name__ == "__main__":
    # List of dataset IDs
    dataset_ids = [17, 52, 53, 109, 143, 144, 697, 763, 936]
    # Create a list of UCIDataset objects
    datasets = [UCIDataset(id) for id in dataset_ids]
    # List of random seeds
    random_seeds = [i for i in range(10)]
    # List of weight calculation methods
    weight_calculation_methods = [calculate_entropy_variance_weights, calculate_entropy_weights]
    # Test data ratio
    test_data_ratio = 0.2

    # Run experiments in parallel
    experiment_logs = Parallel(n_jobs=-1, prefer='processes')(
        delayed(run_experiment)(dataset, seed, weight_function, murphy, test_data_ratio)
        for dataset in datasets
        for seed in random_seeds
        for weight_function in weight_calculation_methods
    )

    # Create a DataFrame from the experiment logs
    results_df = pd.DataFrame(experiment_logs, columns=['dataset', 'seed', 'weight', 'accuracy'])
    # Create the output directory if it doesn't exist
    output_path = Path('./out/')
    output_path.mkdir(exist_ok=True)
    # Save the results to a CSV file
    results_df.to_csv('./out/result.csv', index=False)
