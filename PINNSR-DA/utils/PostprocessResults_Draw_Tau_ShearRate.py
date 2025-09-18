# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:52:41 2024

Extension of PostprocessResults.py for analyzing tau-shear rate relationships
@author: Han Xu
"""

import numpy as np
import os
from scipy.io import loadmat
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


class PostprocessResults:
    def __init__(self, T=10):
        """
        Initialize the postprocessing class with default parameters
        :param T: Period (default 10s)
        """
        self.gamma0 = 0.10
        self.T = T
        self.w = 2 * math.pi / self.T  # Angular frequency
        
    def gam(self, t):
        """Periodic gamma function"""
        return self.gamma0 / self.w * (1 - np.cos(self.w * t))

    def gam_t(self, t):
        """Derivative of periodic gamma function"""
        return self.gamma0 * np.sin(self.w * t)

    def gam_tt(self, t):
        """Second derivative of periodic gamma function"""
        return self.gamma0 * self.w * np.cos(self.w * t)
    
    def gam_t_quadratic_function(self, t):
        """Quadratic gamma function"""
        return 0.002 * t
    
    def gam_heaviside(self, t):
        """
        Returns piecewise values based on time t:
        0~5s: 0.3
        5~10s: -0.3
        10~15s: 0.3
        15~20s: -0.3
        20~25s: 0.3
        outside range: 0
        """
        # Vectorized implementation to handle arrays
        t = np.asarray(t)
        conditions = [
            (t >= 0) & (t < 5),
            (t >= 5) & (t < 10),
            (t >= 10) & (t < 15),
            (t >= 15) & (t < 20),
            (t >= 20) & (t < 25)
        ]
        values = [0.3, -0.3, 0.3, -0.3, 0.3]
        return np.select(conditions, values, default=0)
    
    # Model obtained from PINN+SINDy
    def deriv(self, t, sigma):
        """
        Computes the derivative for the PINN+SINDy model
        :param t: Time
        :param sigma: Current sigma value
        :return: d_sigma_dt derivative
        """
        gamma_t_val = self.gam_t(t)
        gamma_tt_val = self.gam_tt(t)
        
        # Calculates right-hand side of the differential equation
        d_sigma_dt = (
            -6.056830 * gamma_t_val
            -28.078222 * sigma * np.abs(gamma_t_val)
            -31.903385 * sigma**2 * gamma_t_val
        )
        return d_sigma_dt
    
    # Model obtained from PINN+SINDy for quadratic function
    def deriv_quadratic_function(self, t, sigma):
        """
        Computes the derivative for the quadratic PINN+SINDy model
        :param t: Time
        :param sigma: Current sigma value
        :return: d_sigma_dt derivative
        """
        gamma_t_val = self.gam_t_quadratic_function(t)
        
        # Calculates right-hand side of the differential equation
        d_sigma_dt = (
            -5.963015 * gamma_t_val
            -28.01297 * sigma * np.abs(gamma_t_val)
            -32.43191 * sigma**2 * gamma_t_val
        )
        return d_sigma_dt
    
    # Calculate error metrics between two vectors
    def compute_metrics(self, A, B):
        """
        Calculates error metrics between two datasets
        :param A: Reference data
        :param B: Predicted data
        :return: Dictionary containing MSE, RMSE, MAE, RAE, MAPE, r, and R^2
        """
        # Ensure A and B are numpy arrays
        A = np.asarray(A)
        B = np.asarray(B)
        
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape.")
        
        # Calculate errors
        errors = A - B
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        # Calculate MSE
        mse = np.mean(squared_errors)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        
        # Calculate MAE
        mae = np.mean(absolute_errors)
        
        # Calculate RAE
        mean_absolute_deviation = np.mean(np.abs(A - np.mean(A)))
        rae = mae / mean_absolute_deviation if mean_absolute_deviation > 1e-8 else np.nan
        
        # Calculate MAPE with protection against division by zero
        non_zero_mask = np.abs(A) > 1e-8
        if np.sum(non_zero_mask) == 0:
            mape = np.nan
        else:
            mape = np.mean(absolute_errors[non_zero_mask] / np.abs(A[non_zero_mask])) * 100
        
        # Calculate correlation coefficient
        if np.std(A) < 1e-8 or np.std(B) < 1e-8:
            corr_coef = np.nan
        else:
            corr_coef = np.corrcoef(A, B)[0, 1]
        
        # Calculate coefficient of determination R²
        ss_total = np.sum((A - np.mean(A)) ** 2)
        ss_residual = np.sum(squared_errors)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 1e-8 else np.nan

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'RAE': rae,
            'MAPE': mape,
            'r': corr_coef,
            'R^2': r_squared
        }
    
    # Traditional SINDy model obtained from SINDy-np.arange(0.10, 0.51, 0.02)
    def deriv_tSINDy(self, t, sigma):
        """
        Traditional SINDy model derivative
        :param t: Time
        :param sigma: Current sigma value
        :return: d_sigma_dt derivative
        """
        gamma_t_val = self.gam_t(t)
        
        # Traditional SINDy model coefficients
        coef_σ2γt = 19.012703
        coef_γt = -4.362921
        coef_σ_absγt = -0.972048
        coef_σ = -0.950789
        coef_σγt = -0.645912
        
        # Calculate right-hand side of the differential equation
        d_sigma_dt = (
            coef_σ2γt * sigma**2 * gamma_t_val
            + coef_γt * gamma_t_val
            + coef_σ_absγt * sigma * np.abs(gamma_t_val)
            + coef_σ * sigma
            + coef_σγt * sigma * gamma_t_val
        )
        return d_sigma_dt
    
    # Traditional SINDy model obtained from SINDy-np.arange(0.10, 0.51, 0.10)
    def deriv_tSINDy2(self, t, sigma):
        """
        Alternative traditional SINDy model derivative
        :param t: Time
        :param sigma: Current sigma value
        :return: d_sigma_dt derivative
        """
        gamma_t_val = self.gam_t(t)
        
        # Traditional SINDy model coefficients
        coef_σ2γt = 21.360361
        coef_γt = -4.523539
        coef_σ_absγt = -2.501335
        coef_σγt = -2.015833
        coef_σ2_absγt = -0.004059
        
        # Calculate right-hand side of the differential equation
        d_sigma_dt = (
            coef_σ2γt * sigma**2 * gamma_t_val
            + coef_γt * gamma_t_val
            + coef_σ_absγt * sigma * np.abs(gamma_t_val)
            + coef_σγt * sigma * gamma_t_val
            + coef_σ2_absγt * sigma**2 * np.abs(gamma_t_val)
        )
        return d_sigma_dt


def create_results_directory(directory, file_prefix):
    """
    Creates results directory if it doesn't exist
    :param directory: Base directory
    :param file_prefix: File prefix for results subdirectory
    :return: Path to results directory
    """
    results_path = os.path.join(os.path.dirname(directory), 'results', file_prefix)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results directory: '{results_path}'")
    return results_path


def predict_tau_by_discovered_equation(postprocessor, name_array, directory, 
                                      directory_path=f'H:/Batch_OscillatoryShear_2/', 
                                      file_prefix=f'p10000_gam'):
    """
    Generate plots for the data-driven model predictions
    Starting from t=10s, using μ(t=10) as initial condition, predicting 10~22s
    """
    results_path = create_results_directory(directory, file_prefix)
    
    for gam_value in name_array:
        gam_str = f'{gam_value:.2f}'
        file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
        file_path = os.path.join(directory_path, file_name)
        
        try:
            data = loadmat(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            continue
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue

        # Extract data from mat file
        t = data['t']
        tau_xx = data['tau_xx']
        P0 = data['P0']
        
        postprocessor.gamma0 = gam_value
        sigma0 = tau_xx[0,0]/P0[0,0]  # Initial value of σ
        miu = tau_xx.flatten() / P0
        
        # Time configuration
        t_span = (10, 22)
        t_eval = np.linspace(t_span[0], t_span[1], 12001)

        # Solve differential equation
        sol = solve_ivp(postprocessor.deriv, t_span, [sigma0], method='RK45', t_eval=t_eval)
        
        # Configure plot style
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        
        # Plot tau comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), miu.flatten(), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, sol.y[0], label='Data-driven Model', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\sigma_{xy}$/$P_0$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'tau-{gam_value:.2f}.png'))
        plt.close()
        
        # Plot mu comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), abs(miu.flatten()), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, abs(sol.y[0]), label='Data-driven Model', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\mu$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'mu-{gam_value:.2f}.png'))
        plt.close()


def predict_tau_by_discovered_equation2(postprocessor, name_array, directory, 
                                       directory_path=f'H:/Batch_OscillatoryShear_2/', 
                                       file_prefix=f'p10000_gam'):
    """
    Generate plots for the data-driven model predictions
    Starting from t=0s, using μ(t=10) as initial condition, predicting 0~22s, showing 10~22s
    """
    results_path = create_results_directory(directory, file_prefix)
    
    for gam_value in name_array:
        gam_str = f'{gam_value:.2f}'
        file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
        file_path = os.path.join(directory_path, file_name)
        
        try:
            data = loadmat(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            continue
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue

        # Extract data from mat file
        t = data['t']
        tau_xx = data['tau_xx']
        P0 = data['P0']
        
        postprocessor.gamma0 = gam_value
        sigma0 = tau_xx[0,0]/P0[0,0]  # Initial value of σ
        miu = tau_xx.flatten() / P0
        
        # Time configuration
        t_span = (0, 22)
        t_eval = np.linspace(t_span[0], t_span[1], 2201)

        # Solve differential equation
        sol = solve_ivp(postprocessor.deriv, t_span, [sigma0], method='RK45', t_eval=t_eval)
        
        # Configure plot style
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        
        # Plot tau comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), miu.flatten(), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, sol.y[0], label='Data-driven Model', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\sigma_{xy}$/$P_0$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'2tau-{gam_value:.2f}.png'))
        plt.close()
        
        # Plot mu comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), abs(miu.flatten()), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, abs(sol.y[0]), label='Data-driven Model', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\mu$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'2mu-{gam_value:.2f}.png'))
        plt.close()


def predict_tau_by_tSINDy(postprocessor, name_array, directory, 
                         directory_path=f'H:/Batch_OscillatoryShear_2/', 
                         file_prefix=f'p10000_gam'):
    """Generate plots for traditional SINDy model predictions"""
    results_path = create_results_directory(directory, file_prefix)
    
    for gam_value in name_array:
        gam_str = f'{gam_value:.2f}'
        file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
        file_path = os.path.join(directory_path, file_name)
        
        try:
            data = loadmat(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            continue
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue

        # Extract data from mat file
        t = data['t']
        tau_xx = data['tau_xx']
        P0 = data['P0']
        
        postprocessor.gamma0 = gam_value
        sigma0 = tau_xx[0,0]/P0[0,0]  # Initial value of σ
        miu = tau_xx.flatten() / P0
        
        # Time configuration
        t_span = (10, 22)
        t_eval = np.linspace(t_span[0], t_span[1], 1201)

        # Solve differential equation
        sol = solve_ivp(postprocessor.deriv_tSINDy, t_span, [sigma0], method='RK45', t_eval=t_eval)
        
        # Configure plot style
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        
        # Plot tau comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), miu.flatten(), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, sol.y[0], label='Traditional SINDy', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\sigma_{xy}$/$P_0$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'tau-{gam_value:.2f}.png'))
        plt.close()
        
        # Plot mu comparison
        plt.figure(figsize=(10, 6)) 
        plt.plot(t.flatten(), abs(miu.flatten()), label='DEM simulations', color='blue', linewidth=1.5, alpha=0.99)
        plt.plot(sol.t, abs(sol.y[0]), label='Traditional SINDy', color='red', linewidth=2.5, alpha=0.9)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\mu$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'mu-{gam_value:.2f}.png'))
        plt.close()


def compare_prediction_results(postprocessor, name_array, directory, 
                              directory_path=f'H:/Batch_OscillatoryShear_2/', 
                              file_prefix=f'p10000_gam'):
    """Compare predictions from our model and traditional SINDy models"""
    results_path = create_results_directory(directory, file_prefix)
    Error_pSINDy = []
    Error_tSINDy1 = []
    Error_tSINDy2 = []
    
    for gam_value in name_array:
        gam_str = f'{gam_value:.2f}'
        file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
        file_path = os.path.join(directory_path, file_name)
        
        try:
            data = loadmat(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            continue
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue

        # Extract data from mat file
        t = data['t']
        tau_xx = data['tau_xx']
        P0 = data['P0']
        
        postprocessor.gamma0 = gam_value
        sigma0 = tau_xx[0,0]/P0[0,0]  # Initial value of σ
        miu = tau_xx.flatten() / P0
        
        # Time configuration
        t_span = (10, 22)
        t_eval = np.linspace(t_span[0], t_span[1], 1201)

        # Solve differential equations for all models
        sol_pSINDy = solve_ivp(postprocessor.deriv, t_span, [sigma0], method='RK45', t_eval=t_eval)
        sol_tSINDy1 = solve_ivp(postprocessor.deriv_tSINDy, t_span, [sigma0], method='RK45', t_eval=t_eval)
        sol_tSINDy2 = solve_ivp(postprocessor.deriv_tSINDy2, t_span, [sigma0], method='RK45', t_eval=t_eval)
        
        # Calculate error metrics
        Error_pSINDy.append(postprocessor.compute_metrics(miu.flatten(), sol_pSINDy.y[0]))
        Error_tSINDy1.append(postprocessor.compute_metrics(miu.flatten(), sol_tSINDy1.y[0]))
        Error_tSINDy2.append(postprocessor.compute_metrics(miu.flatten(), sol_tSINDy2.y[0]))
        
        # Configure plot style
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        
        # Plot tau comparison
        plt.figure(figsize=(18, 5))
        plt.plot(t.flatten(), miu.flatten(), label='DEM simulations', 
                 color='black', linestyle='-', linewidth=1.3, alpha=0.8)
        plt.plot(sol_tSINDy1.t, sol_tSINDy1.y[0], label='Traditional SINDy', 
                 color='royalblue', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.plot(sol_pSINDy.t, sol_pSINDy.y[0], label='Our Model', 
                 color='crimson', linestyle='-', linewidth=2.5, alpha=0.8)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\sigma_{xy}$/$P_0$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'c-tau-{gam_value:.2f}.png'))
        plt.close()
        
        # Plot mu comparison
        plt.figure(figsize=(18, 5))
        plt.plot(t.flatten(), abs(miu.flatten()), label='DEM simulations', 
                 color='black', linestyle='-', linewidth=1.3, alpha=0.8)
        plt.plot(sol_tSINDy1.t, abs(sol_tSINDy1.y[0]), label='Traditional SINDy', 
                 color='royalblue', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.plot(sol_pSINDy.t, abs(sol_pSINDy.y[0]), label='Our Model', 
                 color='crimson', linestyle='-', linewidth=2.5, alpha=0.8)
        plt.xlabel('t (s)', fontsize=18)
        plt.ylabel(r'$\mu$', fontsize=18)
        plt.title(f'Validation on unknown dataset ($\gamma_0$={gam_value:.2f})', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(results_path, f'c-mu-{gam_value:.2f}.png'))
        plt.close()
    
    # Save error metrics to Excel
    df_pSINDy = pd.DataFrame(Error_pSINDy)
    df_tSINDy1 = pd.DataFrame(Error_tSINDy1)
    df_tSINDy2 = pd.DataFrame(Error_tSINDy2)
    
    # Add gamma0 values to DataFrames
    df_pSINDy.insert(0, 'γ0', name_array[:len(df_pSINDy)])
    df_tSINDy1.insert(0, 'γ0', name_array[:len(df_tSINDy1)])
    df_tSINDy2.insert(0, 'γ0', name_array[:len(df_tSINDy2)])
    
    # Save to Excel
    excel_path = os.path.join(results_path, 'error_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_pSINDy.to_excel(writer, sheet_name='Error_pSINDy', index=False)
        df_tSINDy1.to_excel(writer, sheet_name='Error_tSINDy1', index=False)
        df_tSINDy2.to_excel(writer, sheet_name='Error_tSINDy2', index=False)
    
    print(f"Error metrics saved to: {excel_path}")
    
    # Return last set of results for potential further analysis
    return t.flatten(), miu.flatten(), sol_tSINDy1.y[0], sol_pSINDy.y[0]


def plot_tau_shear_relationship(postprocessor, name_array, directory, 
                               directory_path=f'H:/Batch_OscillatoryShear_2/', 
                               file_prefix=f'p10000_gam'):
    """Plot tau-shear rate relationship"""
    results_path = create_results_directory(directory, file_prefix)
    
    for gam_value in name_array:
        gam_str = f'{gam_value:.2f}'
        file_name = f'{file_prefix}{gam_str}/post/Tau_Gammat.mat'
        file_path = os.path.join(directory_path, file_name)
        
        try:
            data = loadmat(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            continue
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue

        # Extract data from mat file
        t = data['t']
        tau_xx = data['tau_xx']
        P0 = data['P0']
        
        # Calculate mu from DEM data
        miu = tau_xx.flatten() / P0
        
        # Solve quadratic model
        sigma0 = 0  # Initial value
        t_span = (0, 30)
        t_eval = np.linspace(t_span[0], t_span[1], 1201)
        shear_rates = postprocessor.gam_t_quadratic_function(t_eval)
        sol = solve_ivp(postprocessor.deriv_quadratic_function, t_span, 
                       [sigma0], method='RK45', t_eval=t_eval)
        
        # Configure plot style
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        
        # Plot tau-shear rate relationship
        plt.figure(figsize=(10, 5))
        plt.plot(shear_rates, abs(sol.y[0]), label='Data-driven Model', 
                 color='red', linewidth=2.5, alpha=0.9)
        
        # Add DEM data for comparison
        gammat = np.linspace(0, 0.3, 1201)
        plt.plot(np.abs(gammat), np.abs(miu[:len(gammat)]), 
                 label='DEM simulations', color='black', 
                 linestyle='-', linewidth=1.3, alpha=0.8)
        