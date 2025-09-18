# -*- coding: utf-8 -*-
"""
Generates comparison plots of polynomial terms vs. R² scores for training and test sets
Fixed parameters: a[1] = 0, a[2] = 1. Contains only core logic.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --------------------------
# Global Settings: Use Arial font consistently
# --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


class Dataset:
    '''Load and split dataset'''
    def __init__(self, dataset_path, input_features, output_feature):
        self.dataset_path = dataset_path
        self.input_features, self.output_feature = input_features, output_feature
        self.dataframe = self._load_dataset()
        self.train_data, self.test_data = self._split_dataset()

    def _load_dataset(self):
        '''Load CSV data'''
        return pd.read_csv(self.dataset_path)
    
    def _split_dataset(self, test_size=0.2, random_state=42):
        '''Split into training and testing sets (20% test data)'''
        return train_test_split(self.dataframe, test_size=test_size, random_state=random_state)

    def parse_data(self, shuffle_data=True, random_state=42):
        '''Convert to numpy arrays'''
        X_train = self.train_data[self.input_features].to_numpy()
        y_train = self.train_data[self.output_feature].to_numpy().reshape(-1,)
        X_test = self.test_data[self.input_features].to_numpy()
        y_test = self.test_data[self.output_feature].to_numpy().reshape(-1,)
        
        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        return X_train, y_train, X_test, y_test


# --------------------------
# 1. Load Data
# --------------------------
dataset_path = 'D:/PhD_work/Code_PhDs/Project/SINDy4OscillatoryShear/Dimensional_Results/SINDy_Coefficients_1048_(725+195+128)_New_addH0+phi(删除T=6和T=8).csv'
input_features = ['p', 'ds', 'rho', 'E', 'T', 'H']
output_feature = ['C3']

data_loader = Dataset(dataset_path, input_features, output_feature)
X_train, y_train, X_test, y_test = data_loader.parse_data()
print(f"Training data shape: {X_train.shape}, Training label shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test label shape: {y_test.shape}")


# --------------------------
# 2. Dimension Matrix and Basis Vectors (original logic preserved)
# --------------------------
D_in = np.array([
    [-1., 1., -3., -1., 0., 1.],
    [-2., 0., 0., -2., 1., 0.],
    [1., 0., 1., 1., 0., 0.],
])
D_in_rank = matrix_rank(D_in)
print(f"Rank of D_in matrix: {D_in_rank}")

scaling_matrix = np.array([
    [-1., 0.5, 0.],
    [0., -1., -1.],
    [0., -0.5, 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0, 1.]],
)
basis1_in = scaling_matrix[:, 0]
basis2_in = scaling_matrix[:, 1]
basis3_in = scaling_matrix[:, 2]


# --------------------------
# 3. Core Helper Functions
# --------------------------
def calculate_pi(a, X_data):
    '''Calculate dimensionless pi numbers (supports training/test sets)'''
    pi_coefficients = a[0] * basis1_in + a[1] * basis2_in + a[2] * basis3_in
    pi_matrix = np.exp(np.log(X_data).dot(pi_coefficients))
    return np.squeeze(np.asarray(pi_matrix))


def plot_terms_vs_r2(train_terms, train_r2, test_r2):
    '''Generate plot of polynomial terms vs. R² scores (training vs. test sets)'''
    plt.figure(figsize=(10, 6))
    
    # Plot training (blue circles) and test (red squares) R² curves
    plt.plot(train_terms, train_r2, 'o-', color='#457B9D', linewidth=2.5, markersize=8, label='Training Set')
    plt.plot(train_terms, test_r2, 's-', color='#E63946', linewidth=2.5, markersize=8, label='Test Set')
    
    # Format plot
    plt.xlabel('Number of Polynomial Terms', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score', fontsize=14, fontweight='bold')
    plt.title('Polynomial Terms vs R² Score (Train vs Test, Fixed a[1]=0, a[2]=1)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', framealpha=0.9)
    
    # Annotate R² values every 3 terms to avoid overcrowding
    for i, (terms, tr2, te2) in enumerate(zip(train_terms, train_r2, test_r2)):
        if i % 3 == 0:
            plt.annotate(f'{tr2:.3f}', (terms, tr2), xytext=(-15, 10), textcoords='offset points', fontsize=10, color='#457B9D')
            plt.annotate(f'{te2:.3f}', (terms, te2), xytext=(-15, -20), textcoords='offset points', fontsize=10, color='#E63946')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('terms_vs_r2_train_test.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nChart 'terms_vs_r2_train_test.png' saved successfully!")


# --------------------------
# 4. Main Logic: Calculate R² and Generate Plot
# --------------------------
# Fixed parameters: a[1] = 0, a[2] = 1
fixed_a = np.array([1, 0, 1])
print(f"\nFixed parameters: a = {fixed_a}")

# Scalers (fit only on training data)
scaler_pi1 = StandardScaler()
scaler_pi2 = StandardScaler()
y_train_scaled = scaler_pi2.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_pi2.transform(y_test.reshape(-1, 1)).flatten()

# Polynomial degree range (0-49, corresponding to 1-50 terms)
degrees_to_test = range(0, 50)
poly_terms_list = [d for d in degrees_to_test]

# Store only R² data needed for plotting
train_r2_scores = []
test_r2_scores = []


# Iterate through each polynomial term count and calculate R²
for degree, terms in zip(degrees_to_test, poly_terms_list):
    print(f"Processing {terms} terms ({degree}-th degree)...", end=" ")
    
    # Training set fitting
    pi1_train = calculate_pi(fixed_a, X_train)
    pi1_train_scaled = scaler_pi1.fit_transform(pi1_train.reshape(-1, 1))
    coefficients = np.polyfit(pi1_train_scaled.flatten(), y_train_scaled, degree)
    
    # Calculate training set R²
    y_pred_train = np.polyval(coefficients, pi1_train_scaled.flatten())
    train_r2 = r2_score(y_train_scaled, y_pred_train)
    
    # Calculate test set R²
    pi1_test = calculate_pi(fixed_a, X_test)
    pi1_test_scaled = scaler_pi1.transform(pi1_test.reshape(-1, 1))
    y_pred_test = np.polyval(coefficients, pi1_test_scaled.flatten())
    test_r2 = r2_score(y_test_scaled, y_pred_test)
    
    # Store core data
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")


# --------------------------
# 5. Generate and Save Target Plot
# --------------------------
plot_terms_vs_r2(poly_terms_list, train_r2_scores, test_r2_scores)

# (Optional) Save core data for plot as CSV (facilitates further analysis)
result_df = pd.DataFrame({
    'polynomial_degree': poly_terms_list,
    'train_r2': train_r2_scores,
    'test_r2': test_r2_scores
})
result_df.to_csv('terms_vs_r2_data.csv', index=False)
print("Core data saved to 'terms_vs_r2_data.csv'")

print("\nAll tasks completed!")
