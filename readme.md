# Classification Experiments Based on Variance of Belief Information

This project conducts classification experiments centered around the paper "Variance of Belief Information". The aim is to explore the application effect of the variance of belief information in classification tasks. By using UCI datasets, combining Gaussian models and evidence theory, it compares the influence of different weight calculation methods on classification accuracy.

## Project Overview
This project implements a classification experiment based on the variance of belief information in Python. The main steps include data loading, Gaussian model construction, data conversion to mass distributions, evidence fusion, and final classification decision - making. Different weight calculation methods, such as those based on Deng entropy and a combination of Deng entropy and information variance, are used to evaluate their performance in classification tasks.

## Steps to Run
1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the Experiment**:
    ```bash
    python run.py
    ```
Running the `run.py` script will automatically download the required UCI datasets (if not cached) and conduct a series of experiments. The final results will be saved in the `out/result.csv` file.

## Configuration Parameters
- **Datasets**: In `run.py`, you can modify the `dataset_ids` list to select the UCI datasets to be used.
- **Random Seeds**: The `random_seeds` list defines the random seeds for the experiments.
- **Weight Calculation Methods**: The `weight_calculation_methods` list contains the weight calculation methods to be used.
- **Test Set Ratio**: The `test_data_ratio` defines the proportion of the test set in the dataset.

## Explanation of Main Modules
- **`src/data.py`**: The `UCIDataset` class is responsible for loading and processing UCI datasets, supporting data caching and train - test set splitting.
- **`src/gaussian.py`**: The `Gaussianer` class is used to build Gaussian models and convert input data into mass distributions.
- **`src/fusionf.py`**: It contains evidence fusion methods, such as the Murphy method.
- **`src/weightf.py`**: It implements different weight calculation methods, such as those based on Deng entropy and a combination of Deng entropy and information variance.

## Experiment Results
The experiment results will be saved in the `out/result.csv` file, including the dataset ID, random seed, weight calculation method, and classification accuracy.

## Developer Information
- Tianxiang Zhan
- Xingyuan Chen


## License
This project is licensed under the MIT License. Please refer to the `LICENSE` file for specific details.