# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 00:52:06 2023

Dimensional analysis and dimensionless number discovery
This code performs systematic dimensional analysis to identify optimal dimensionless 
representations of the data using polynomial regression and grid search optimization.

@author: Han Xu
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.io


class DatasetHandler:
    '''
    Class for loading, parsing, and splitting datasets
    '''

    def __init__(self, dataset_path, input_features, output_feature):
        """
        Initialize dataset handler with path and feature lists
        :param dataset_path: Path to the CSV dataset file
        :param input_features: List of input feature column names
        :param output_feature: List containing the output feature column name
        """
        self.dataset_path = dataset_path
        self.input_features = input_features
        self.output_feature = output_feature

        self.dataframe = self._load_dataset()
        self.train_data, self.test_data = self._split_dataset()

    def _load_dataset(self):
        '''Load dataset from CSV file'''
        try:
            return pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _split_dataset(self, test_size=0.01, random_state=0):
        '''
        Randomly split dataset into training and testing subsets
        :param test_size: Proportion of data to use for testing
        :param random_state: Seed for reproducibility
        :return: Tuple of (train_dataframe, test_dataframe)
        '''
        return train_test_split(
            self.dataframe, 
            test_size=test_size, 
            random_state=random_state
        )

    def parse_data(self, shuffle_data=True, random_state=0):
        '''
        Convert data to numpy arrays for model training
        :param shuffle_data: Whether to shuffle the training data
        :param random_state: Seed for shuffling reproducibility
        :return: Tuple of (X_train, y_train, X_test, y_test)
        '''
        # Extract features from dataframes
        X_train = self.train_data[self.input_features].to_numpy()
        y_train = self.train_data[self.output_feature].to_numpy().reshape(-1,)

        X_test = self.test_data[self.input_features].to_numpy()
        y_test = self.test_data[self.output_feature].to_numpy().reshape(-1,)

        # Shuffle training data if requested
        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        return X_train, y_train, X_test, y_test


# --------------------------
# Dataset Configuration
# --------------------------
# Note: Users should update this path to their dataset location
dataset_path = 'D:/Project/SINDy4OscillatoryShear/Dimensional_Results/SINDy_Coefficients.csv'

# Input and output feature configuration
input_features = ['p', 'ds', 'rho', 'E', 'T', 'H']
output_feature = ['C1']

# Initialize dataset handler and parse data
data_handler = DatasetHandler(dataset_path, input_features, output_feature)
X_train, y_train, X_test, y_test = data_handler.parse_data()
print(f"Data shapes - Training: {X_train.shape}, Testing: {X_test.shape}")


# --------------------------
# Dimensional Analysis Setup
# --------------------------
# Dimension matrix (input parameters)
D_in = np.array([
    [-1., 1., -3., -1., 0., 1.],
    [-2., 0., 0., -2., 1., 0.],
    [1., 0., 1., 1., 0., 0.],
])

# Dimension matrix (output parameter)
D_out = np.array([
    [0.],
    [0.],
    [0.],
])

# Calculate rank of input dimension matrix
D_in_rank = matrix_rank(D_in)
print(f"Rank of input dimension matrix: {D_in_rank}")


# --------------------------
# Basis Vectors Configuration
# --------------------------
# Basis vectors in columns for scaling matrix
scaling_matrix = np.array([
    [-1., 0.5, 0.],
    [0., -1., -1.],
    [0., -0.5, 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0, 1.]
])

# Extract individual basis vectors
basis1_in = scaling_matrix[:, 0]
basis2_in = scaling_matrix[:, 1]
basis3_in = scaling_matrix[:, 2]

print(f'Basis vector 1: \n{basis1_in}')
print(f'Basis vector 2: \n{basis2_in}')
print(f'Basis vector 3: \n{basis3_in}')


# --------------------------
# Helper Functions
# --------------------------
def calculate_pi(a):
    '''
    Calculate dimensionless numbers (pi terms) using given coefficients
    :param a: Coefficients for combining basis vectors
    :return: Array of dimensionless pi values
    '''
    # Combine basis vectors with coefficients
    pi_coefficients = a[0] * basis1_in + a[1] * basis2_in + a[2] * basis3_in
    
    # Calculate pi values using logarithmic transformation to handle multiplication
    pi_matrix = np.exp(np.log(X_train).dot(pi_coefficients))
    return np.squeeze(np.asarray(pi_matrix))


def calculate_prediction(a, weights):
    '''
    Calculate predictions using polynomial function of dimensionless numbers
    :param a: Coefficients for pi calculation
    :param weights: Polynomial coefficients
    :return: Array of predictions
    '''
    pi = calculate_pi(a)
    # Evaluate polynomial: weights[0] + weights[1]*pi + ... + weights[n]*pi^n
    prediction = np.polyval(weights, pi)
    return prediction


def objective_function(a, weights):
    '''
    Calculate objective function (mean squared error)
    :param a: Coefficients for pi calculation
    :param weights: Polynomial coefficients
    :return: MSE between predictions and actual values
    '''
    return np.square(pi2 - calculate_prediction(a, weights)).mean()


def visualize_relationship(pi1, pi2, iteration=None):
    '''
    Visualize relationship between dimensionless numbers
    :param pi1: First dimensionless number
    :param pi2: Second dimensionless number
    :param iteration: Current iteration number (for title)
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(pi1, pi2, alpha=0.7)
    plt.xlabel(r'$\Pi_1$', fontsize=16)
    plt.ylabel(r'$\Pi_2$', fontsize=16)
    if iteration is not None:
        plt.title(f'Iteration: {iteration}', fontsize=18)
    else:
        plt.title('Dimensionless Relationship', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


# --------------------------
# Main Analysis: Grid Search
# --------------------------
# Configuration parameters
polynomial_degree = 16  # Order of polynomial regression
search_range = np.linspace(-6, 6, 25)  # Search range for coefficients

# Initialize coefficient arrays
a = np.zeros(3)
a[0] = 1  # Fixed first coefficient

# Scale output values
scaler = StandardScaler()
pi2 = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

# Store search history
history = {
    'a1': [],
    'a2': [],
    'r2_score': []
}

# Perform grid search over coefficient space
print("Starting grid search for optimal dimensionless representation...")
for a1 in search_range:
    for a2 in search_range:
        # Set current coefficients
        a[1] = a1
        a[2] = a2
        
        # Calculate dimensionless input (pi1) and scale
        pi1 = calculate_pi(a)
        pi1_scaled = scaler.fit_transform(pi1.reshape(-1, 1)).flatten()
        
        # Fit polynomial regression
        coefficients = np.polyfit(pi1_scaled, pi2, polynomial_degree)
        predictions = np.polyval(coefficients, pi1_scaled)
        
        # Calculate goodness of fit
        r2 = r2_score(pi2, predictions)
        
        # Store results
        history['a1'].append(a1)
        history['a2'].append(a2)
        history['r2_score'].append(r2)

# Organize results for saving
results_array = np.vstack([
    history['a1'],
    history['a2'],
    history['r2_score']
])

# Save results
output_filename = 'grid_search_results.mat'
scipy.io.savemat(output_filename, {'search_results': results_array})
print(f"Grid search completed. Results saved to {output_filename}")

# Example visualization of the last relationship
visualize_relationship(pi1_scaled, pi2)
    