# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:52:41 2024

The GenerateData class generates and processes experimental data for oscillatory shear analysis.
It supports multiple data generation scenarios including dimensional analysis cases and 
standard oscillatory shear experiments, with functionality for dataset splitting and 
preprocessing for machine learning applications.

@author: Han Xu
"""

import numpy as np
import os
from scipy.io import loadmat
import math

class GenerateData:
    """
    Generates and processes data from oscillatory shear experiments.
    Provides methods for loading experimental data, calculating derived quantities,
    and preparing datasets for machine learning (training/validation splits).
    """
    
    def __init__(self):
        """Initialize with default time arrays for data generation."""
        self.t = np.linspace(0, 10, 1000)           # Standard time array
        self.t_collection = np.linspace(0, 10, 200000)  # High-resolution time array for derivatives
    
    def create_DimensionalCase_data(self, name_list, project_root, collection_multiple, 
                                   directory_path=r'H:/Batch_OscillatoryShear_4/', log_flag=False):
        """
        Generates dimensional data for dimensionless learning coefficient calculations.
        
        Args:
            name_list (list): List of folder names containing experimental data
            project_root (str): Root directory of the project
            collection_multiple (int): Resolution multiplier for high-density time array
            directory_path (str): Base path to data directories
            log_flag (bool): If True, uses log(gamma0) instead of raw gamma0 values
        
        Returns:
            tuple: Contains comprehensive dataset splits and bounds:
                - Data_Total: Complete merged dataset
                - Data_Train: Training subset (80% of data)
                - Data_Validation: Validation subset (20% of data)
                - Data_Collection: Combined dataset for PINN training
                - lb: Lower bounds of input features
                - ub: Upper bounds of input features
                - t: Time array from experimental measurements
        """
        Data_Total = np.array([])
        Data_Collection_Only = np.array([])
        
        for gam_value in name_list:
            # Construct file path
            file_name = f'{gam_value}/post/Tau_Gammat.mat'
            file_path = os.path.join(directory_path, file_name)
                
            # Load .mat file with error handling
            try:
                data = loadmat(file_path)
            except FileNotFoundError:
                print(f'File not found: {file_path}')
                continue
            except Exception as e:
                print(f'Error reading {file_path}: {e}')
                continue
                
            # Extract variables from mat file
            t, gammat, gamma0, rho, P0, T, tau_xx, ds, epsilon_strain = (
                data['t'], data['gammat'], data['gamma0'], data['rho'], data['P0'],
                data['T'], data['tau_xx'], data['ds'], data['epsilon_strain'])
            
            w = 2 * math.pi / T  # Angular frequency
            
            # Define deformation functions
            def gam(t_val):
                return gamma0/(w) * (1 - np.cos(w * t_val))
            
            def gam_t(t_val):
                return gamma0 * np.sin(w * t_val)
            
            def gam_tt(t_val):
                return gamma0 * w * np.cos(w * t_val)
            
            # Prepare time arrays
            t_total = t.flatten()
            t_collection = np.linspace(int(min(t)), int(max(t)), 
                                      collection_multiple * (len(t) - 1) + 1)
            
            # Create dataset for current case
            Data_Total_i = np.zeros((len(t_total), 6))
            Data_Total_i[:, 0] = t_total
            Data_Total_i[:, 1] = gam(t_total)
            Data_Total_i[:, 2] = gam_t(t_total)
            Data_Total_i[:, 3] = gam_tt(t_total)
            Data_Total_i[:, 4] = tau_xx.flatten() / P0  # Normalized stress
            Data_Total_i[:, 5] = np.log(gamma0) if log_flag else gamma0
            
            # Create high-resolution derivative dataset
            data_derivative_i = np.zeros((len(t_collection), 5))
            data_derivative_i[:, 0] = t_collection
            data_derivative_i[:, 1] = gam(t_collection)
            data_derivative_i[:, 2] = gam_t(t_collection)
            data_derivative_i[:, 3] = gam_tt(t_collection)
            data_derivative_i[:, 4] = np.log(gamma0) if log_flag else gamma0

            # Accumulate data
            if Data_Total.size == 0:
                Data_Total = Data_Total_i
            else:
                Data_Total = np.vstack((Data_Total, Data_Total_i))
                
            if Data_Collection_Only.size == 0:
                Data_Collection_Only = data_derivative_i
            else:
                Data_Collection_Only = np.vstack((Data_Collection_Only, data_derivative_i))
        
        # Split into training and validation sets (80/20 split)
        split_ratio = 0.8
        np.random.seed(42)  # Ensure reproducibility
        idx_train = np.random.choice(Data_Total.shape[0], 
                                    int(Data_Total.shape[0] * split_ratio), 
                                    replace=False)
        Data_Train = Data_Total[idx_train, :]
        
        # Create combined collection dataset for PINN training
        # (removes stress column as it's not needed for derivative calculations)
        Data_Train_modified = np.delete(Data_Train, 4, axis=1)
        Data_Collection = np.concatenate((Data_Collection_Only, Data_Train_modified), axis=0)
        
        # Validation set
        idx_val = np.setdiff1d(np.arange(Data_Total.shape[0]), idx_train, assume_unique=True)
        Data_Validation = Data_Total[idx_val, :]
        
        # Feature bounds for normalization
        X_u_meas = np.vstack((Data_Train[:, 5], Data_Train[:, 0])).T
        lb, ub = X_u_meas.min(0), X_u_meas.max(0) 
        
        return Data_Total, Data_Train, Data_Validation, Data_Collection, lb, ub, t
    
    def create_oscillatoryshear_data(self, name_array, directory, 
                                    directory_path=r'H:/Batch_OscillatoryShear_2/', 
                                    file_prefix=r'p10000_gam', log_flag=False):
        """
        Generates data for standard oscillatory shear experiments.
        
        Args:
            name_array (array): Array of gamma values to process
            directory (str): Base directory for results storage
            directory_path (str): Path to experimental data directories
            file_prefix (str): Prefix for data folder naming convention
            log_flag (bool): If True, uses log(gamma0) instead of raw gamma0 values
        
        Returns:
            tuple: Contains comprehensive dataset splits and bounds:
                - Data_Total: Complete merged dataset
                - Data_Train: Training subset (80% of data)
                - Data_Validation: Validation subset (20% of data)
                - Data_Collection: Combined dataset for PINN training
                - lb: Lower bounds of input features
                - ub: Upper bounds of input features
                - t: Time array from experimental measurements
        """
        Data_Total = np.array([])
        Data_Collection_Only = np.array([])
        
        for gam_value in name_array:
            # Format gamma value to match folder naming convention
            gam_str = f'{gam_value:.2f}'
            
            # Construct paths
            file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
            file_path = os.path.join(directory_path, file_name)
            store_path = f'{directory}/results/{file_prefix}{gam_str}'
            
            # Check storage directory status
            if not os.path.exists(store_path):
                print(f"Directory '{store_path}' does not exist. Create it to store results.")
            else:
                print(f"Directory '{store_path}' already exists.")
                
            # Load .mat file with error handling
            try:
                data = loadmat(file_path)
            except FileNotFoundError:
                print(f'File not found: {file_path}')
                continue
            except Exception as e:
                print(f'Error reading {file_path}: {e}')
                continue
                
            # Extract variables from mat file
            t, gammat, gamma0, rho, P0, T, tau_xx, ds, epsilon_strain = (
                data['t'], data['gammat'], data['gamma0'], data['rho'], data['P0'],
                data['T'], data['tau_xx'], data['ds'], data['epsilon_strain'])
            
            w = 2 * math.pi / T  # Angular frequency
            
            # Define deformation functions
            def gam(t_val):
                return gamma0/(w) * (1 - np.cos(w * t_val))
            
            def gam_t(t_val):
                return gamma0 * np.sin(w * t_val)
            
            def gam_tt(t_val):
                return gamma0 * w * np.cos(w * t_val)
            
            # Prepare time arrays
            t_total = t.flatten()
            t_collection = np.linspace(int(min(t)), int(max(t)), 20 * (len(t) - 1) + 1)
            
            # Create dataset for current case
            Data_Total_i = np.zeros((len(t_total), 6))
            Data_Total_i[:, 0] = t_total
            Data_Total_i[:, 1] = gam(t_total)
            Data_Total_i[:, 2] = gam_t(t_total)
            Data_Total_i[:, 3] = gam_tt(t_total)
            Data_Total_i[:, 4] = tau_xx.flatten() / P0  # Normalized stress
            Data_Total_i[:, 5] = np.log(gamma0) if log_flag else gamma0
            
            # Create high-resolution derivative dataset
            data_derivative_i = np.zeros((len(t_collection), 5))
            data_derivative_i[:, 0] = t_collection
            data_derivative_i[:, 1] = gam(t_collection)
            data_derivative_i[:, 2] = gam_t(t_collection)
            data_derivative_i[:, 3] = gam_tt(t_collection)
            data_derivative_i[:, 4] = np.log(gamma0) if log_flag else gamma0

            # Accumulate data
            if Data_Total.size == 0:
                Data_Total = Data_Total_i
            else:
                Data_Total = np.vstack((Data_Total, Data_Total_i))
                
            if Data_Collection_Only.size == 0:
                Data_Collection_Only = data_derivative_i
            else:
                Data_Collection_Only = np.vstack((Data_Collection_Only, data_derivative_i))
        
        # Split into training and validation sets (80/20 split)
        split_ratio = 0.8
        np.random.seed(42)  # Ensure reproducibility
        idx_train = np.random.choice(Data_Total.shape[0], 
                                    int(Data_Total.shape[0] * split_ratio), 
                                    replace=False)
        Data_Train = Data_Total[idx_train, :]
        
        # Create combined collection dataset for PINN training
        Data_Train_modified = np.delete(Data_Train, 4, axis=1)  # Remove stress column
        Data_Collection = np.concatenate((Data_Collection_Only, Data_Train_modified), axis=0)
        
        # Validation set
        idx_val = np.setdiff1d(np.arange(Data_Total.shape[0]), idx_train, assume_unique=True)
        Data_Validation = Data_Total[idx_val, :]
        
        # Feature bounds for normalization
        X_u_meas = np.vstack((Data_Train[:, 5], Data_Train[:, 0])).T
        lb, ub = X_u_meas.min(0), X_u_meas.max(0) 
        
        return Data_Total, Data_Train, Data_Validation, Data_Collection, lb, ub, t

    def create_PNAS_data(self):
        """Placeholder method for generating PNAS-formatted data. Implementation pending."""
        pass
    
    def create_synthetic_data(self, gamma0_values):
        """
        Generates synthetic data for testing and validation purposes.
        
        Args:
            gamma0_values (list): List of gamma0 values for synthetic data generation
        
        Returns:
            list: Dictionaries containing deformation fields and stress for each gamma0 value
        """
        # Define deformation and stress functions
        def gam(t_val, gamma0):
            return gamma0 * np.sin(t_val)

        def gam_t(t_val, gamma0):
            return gamma0 * np.cos(t_val)

        def gam_tt(t_val, gamma0):
            return -gamma0 * np.sin(t_val)
        
        def sigma(t_val, gamma0):
            return gamma0 * (np.sin(t_val) + np.cos(t_val))
        
        # Generate data for each gamma0 value
        results = []
        for gamma0 in gamma0_values:
            results.append({
                'gam': gam(self.t, gamma0),
                'gam_t': gam_t(self.t, gamma0),
                'gam_tt': gam_tt(self.t, gamma0),
                'gam_c': gam(self.t_collection, gamma0),
                'gam_t_c': gam_t(self.t_collection, gamma0),
                'gam_tt_c': gam_tt(self.t_collection, gamma0),
                'sigma': sigma(self.t, gamma0)
            })
        
        return results

if __name__ == "__main__":
    # Example usage (adjust paths according to your directory structure)
    current_dir = os.getcwd()
    data_dir = r'H:/Batch_OscillatoryShear_2/'
    file_prefix = r'p10000_gam'
    
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_generator = GenerateData()
