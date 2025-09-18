# -*- coding: utf-8 -*-
"""
March 20, 2025
PostprocessResults class: Compares predictions from 8 formulas with DEM simulation results
Core functionalities: Batch solution of 8 differential equations, calculation of error metrics,
                     generation of comparison plots, and saving of Excel analysis reports
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
from datetime import datetime


class PostprocessResults:
    def __init__(self, today_datetime, T=10):
        """
        Initialization: Set time parameters, period, coefficients for 8 formulas, and plotting styles
        :param today_datetime: Current timestamp (used for directory naming)
        :param T: Period (default 10s, consistent with original code)
        """
        self.gamma0 = 0.10  # Initial gamma0 (will be updated based on data)
        self.today_datetime = today_datetime
        self.T = T
        self.w = 2 * math.pi / self.T  # Fixed bug where w was undefined in original code
        
        # -------------------------- Coefficient configuration for 8 formulas --------------------------
        # Each formula format: d_sigma_dt = a*gamma_t_val + b*sigma*|gamma_t_val| + c*sigma²*gamma_t_val
        # Order: Formula 1 → Formula 8, (a, b, c) correspond to respective coefficients
        self.formula_coeffs = [
            (-5.70, -15.91, 0.00),    # Formula 1 (no sigma² term, c=0)
            (-6.00, -28.58, -34.16),  # Formula 2
            (-6.09, -29.25, -35.39),  # Formula 3
            (-6.07, -29.27, -35.57),  # Formula 4
            (-6.02, -29.04, -35.36),  # Formula 5
            (-6.10, -29.35, -35.59),  # Formula 6
            (-6.10, -29.25, -35.33),  # Formula 7
            (-6.10, -29.37, -35.67)   # Formula 8
        ]
        self.formula_labels = [f'Formula {i+1}' for i in range(8)]  # Formula labels (for plotting)
        
        # -------------------------- Plot style configuration --------------------------
        self.plot_colors = plt.cm.Set2(np.linspace(0, 1, 8))  # 8 distinct colors
        self.plot_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Line style combinations


    def gam(self, t):
        """Original periodic gamma function (retained for compatibility)"""
        return self.gamma0 / self.w * (1 - np.cos(self.w * t))


    def gam_t(self, t):
        """Original periodic gamma derivative (retained for compatibility)"""
        return self.gamma0 * np.sin(self.w * t)


    def gam_tt(self, t):
        """Original second derivative of periodic gamma (retained for compatibility)"""
        return self.gamma0 * self.w * np.cos(self.w * t)


    def gam_t_quadratic_function(self, t):
        """Original quadratic gamma function (retained for compatibility)"""
        return 0.002 * t


    def gam_heaviside(self, t):
        """
        Vectorized Heaviside piecewise function (fixed issue where original code couldn't handle arrays)
        Piecewise logic: 0~5s: gamma0 | 5~10s: -gamma0 | 10~15s: gamma0 | 15~20s: -gamma0 | 20~25s: gamma0 | otherwise:0
        :param t: Time (scalar or numpy array)
        :return: gamma_t_val (same shape as t)
        """
        t = np.asarray(t)  # Ensure input is an array
        conditions = [
            (t >= 0) & (t < 5),
            (t >= 5) & (t < 10),
            (t >= 10) & (t < 15),
            (t >= 15) & (t < 20),
            (t >= 20) & (t < 25)
        ]
        values = [self.gamma0, -self.gamma0, self.gamma0, -self.gamma0, self.gamma0]
        return np.select(conditions, values, default=0)


    def deriv_by_formula(self, t, sigma, formula_idx):
        """
        General differential equation solver: Calculate d_sigma_dt using coefficients based on formula index
        :param t: Time
        :param sigma: Current sigma value
        :param formula_idx: Formula index (0~7 corresponding to formulas 1~8)
        :return: d_sigma_dt (derivative of sigma)
        """
        # Validate formula index
        if not (0 <= formula_idx < len(self.formula_coeffs)):
            raise ValueError(f"Formula index must be between 0 and 7, current input: {formula_idx}")
        
        # Get coefficients for current formula
        a, b, c = self.formula_coeffs[formula_idx]
        # Get gamma_t value from Heaviside function
        gamma_t_val = self.gam_heaviside(t)
        # Calculate differential equation (core formula)
        d_sigma_dt = a * gamma_t_val + b * sigma * np.abs(gamma_t_val) + c * (sigma ** 2) * gamma_t_val
        return d_sigma_dt


    def compute_metrics(self, A, B):
        """
        Calculate error metrics between two datasets (fixed error where mean_absolute_error was undefined)
        :param A: Reference data (e.g., DEM)
        :param B: Predicted data (e.g., formula results)
        :return: Dictionary of error metrics (MSE, RMSE, MAE, etc.)
        """
        A = np.asarray(A)
        B = np.asarray(B)
        if A.shape != B.shape:
            raise ValueError(f"Data shapes do not match: A={A.shape}, B={B.shape}")
        
        errors = A - B
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2

        # Calculate metrics (with protection against zero denominators)
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(absolute_errors)  # Correctly defined MAE variable
        
        # RAE (fixed typo: changed mean_absolute_error to mae)
        mean_absolute_deviation = np.mean(np.abs(A - np.mean(A)))
        rae = mae / mean_absolute_deviation if mean_absolute_deviation > 1e-8 else np.nan
        
        # MAPE (avoid division by zero)
        non_zero_mask = np.abs(A) > 1e-8
        if np.sum(non_zero_mask) == 0:
            mape = np.nan
        else:
            mape = np.mean(absolute_errors[non_zero_mask] / np.abs(A[non_zero_mask])) * 100
        
        # Correlation coefficient and R²
        if np.std(A) < 1e-8 or np.std(B) < 1e-8:
            corr_coef = np.nan
            r_squared = np.nan
        else:
            corr_coef = np.corrcoef(A, B)[0, 1]
            ss_total = np.sum((A - np.mean(A)) ** 2)
            ss_residual = np.sum(squared_errors)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 1e-8 else np.nan

        return {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae,
            'RAE': rae, 'MAPE': mape, 'r': corr_coef, 'R^2': r_squared
        }


    def compare_eight_formulas(self, name_array, directory, directory_path=f'H:/Batch_OscillatoryShear_2/', file_prefix=f'p10000_gam'):
        """
        Core method: Compare predictions from 8 formulas with DEM data
        :param name_array: Array of gamma0 values to analyze (e.g., [0.10, 0.15, ...])
        :param directory: Working directory (for compatibility with original parameters)
        :param directory_path: Root path for data files
        :param file_prefix: Prefix for data file names
        """
        # -------------------------- 1. Initialize storage path --------------------------
        store_path = os.path.join(directory_path, f'Eight_Formulas_Results_{self.today_datetime}')
        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f"Created result directory: {store_path}")
        else:
            print(f"Result directory already exists: {store_path}")

        # -------------------------- 2. Initialize data storage --------------------------
        formula_errors = [[] for _ in range(8)]  # Error list for each formula (index = formula index)
        all_raw_data = []  # All raw data (DEM + 8 formula predictions)

        # -------------------------- 3. Global plotting configuration --------------------------
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.serif'] = 'Times New Roman'
        mpl.rcParams['axes.unicode_minus'] = False  # Fix issue with negative sign display
        mpl.rcParams['figure.dpi'] = 100  # Temporary plotting resolution

        # -------------------------- 4. Analyze each gamma0 value --------------------------
        for gam_idx, gam_value in enumerate(name_array):
            gam_str = f'{gam_value:.2f}'
            # Construct MAT data file path
            mat_file = f'{file_prefix}{gam_str}_Heaviside/post/Tau_Gammat.mat'
            mat_path = os.path.join(directory_path, mat_file)

            # -------------------------- 4.1 Read MAT data --------------------------
            try:
                data = loadmat(mat_path)
            except FileNotFoundError:
                print(f"File not found: {mat_path}, skipping this gamma0")
                continue
            except Exception as e:
                print(f"Error reading file {mat_path}: {str(e)}, skipping this gamma0")
                continue

            # Extract key data (flatten to 1D arrays to avoid dimension issues)
            t_dem = data['t'].flatten()  # DEM time series
            tau_xx = data['tau_xx'].flatten()  # DEM tau_xx
            P0 = data['P0'].flatten() if data['P0'].ndim > 1 else data['P0']  # Handle P0 dimensions
            miu_dem = tau_xx / P0  # DEM sigma_xy/P0 (reference data)
            self.gamma0 = gam_value  # Update current gamma0

            # -------------------------- 4.2 Solution configuration --------------------------
            sigma0 = tau_xx[0] / P0[0] if len(P0) > 0 else 0.0  # Initial condition
            t_span = (10, 22)  # Prediction time range (consistent with original code)
            t_eval = np.linspace(t_span[0], t_span[1], 1201)  # Prediction time points (1201 points)

            # Store all data for current gamma0
            current_gam_data = {
                'gamma0': gam_value,
                't_dem': t_dem,
                'miu_dem': miu_dem,
                't_predict': t_eval,
                'miu_predict': []  # Store predictions from 8 formulas
            }

            # -------------------------- 4.3 Solve for each of the 8 formulas --------------------------
            sol_list = []  # Store solution results for 8 formulas
            for formula_idx in range(8):
                # Call solver (RK45 method, consistent with original code)
                sol = solve_ivp(
                    fun=lambda t, sigma: self.deriv_by_formula(t, sigma, formula_idx),
                    t_span=t_span,
                    y0=[sigma0],
                    method='RK45',
                    t_eval=t_eval,
                    atol=1e-8,  # Increased solution accuracy
                    rtol=1e-6
                )
                sol_list.append(sol)
                current_gam_data['miu_predict'].append(sol.y[0])  # Save prediction results

                # -------------------------- 4.4 Calculate errors for current formula --------------------------
                # Interpolate DEM data to prediction time points (ensure consistent data length)
                dem_mask = (t_dem >= t_span[0]) & (t_dem <= t_span[1])
                if np.sum(dem_mask) == 0:
                    print(f"No valid DEM data for gamma0={gam_value:.2f}, skipping error calculation for formula {formula_idx+1}")
                    formula_errors[formula_idx].append(None)
                    continue
                
                # Interpolate DEM to t_eval (linear interpolation)
                miu_dem_interp = np.interp(t_eval, t_dem[dem_mask], miu_dem[dem_mask])
                # Calculate error metrics
                error_dict = self.compute_metrics(miu_dem_interp, sol.y[0])
                # Add metadata
                error_dict.update({
                    'gamma0': gam_value,
                    'formula_idx': formula_idx + 1,  # Formula number (1~8)
                    'formula_name': self.formula_labels[formula_idx]
                })
                formula_errors[formula_idx].append(error_dict)

            # Save current gamma0 data to master list
            all_raw_data.append(current_gam_data)

            # -------------------------- 4.5 Generate comparison plots --------------------------
            fig, ax = plt.subplots(figsize=(12, 8))
            # Plot DEM reference curve (thick black line)
            ax.plot(t_dem, miu_dem, label='DEM simulations', color='black', linestyle='-', linewidth=2.0, alpha=0.9)
            # Plot prediction curves for 8 formulas
            for formula_idx in range(8):
                sol = sol_list[formula_idx]
                ax.plot(
                    sol.t, sol.y[0],
                    label=self.formula_labels[formula_idx],
                    color=self.plot_colors[formula_idx],
                    linestyle=self.plot_linestyles[formula_idx],
                    linewidth=1.5,
                    alpha=0.8
                )
            # Plot configuration
            ax.set_xlabel('t (s)', fontsize=16)
            ax.set_ylabel(r'$\sigma_{xy}/P_0$', fontsize=16)
            ax.set_title(f'Comparison of 8 Formulas (γ₀ = {gam_value:.2f})', fontsize=18, pad=20)
            ax.tick_params(axis='both', which='major', labelsize=12)
            # Place legend on right side (to avoid obscuring curves)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, frameon=True)
            # Adjust layout (ensure legend is fully visible)
            plt.tight_layout()
            # Save image (high resolution 300dpi)
            img_path = os.path.join(store_path, f'Formula_Comparison_gam{gam_str}.png')
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Comparison plot saved: {img_path}")

        # -------------------------- 5. Save results to Excel --------------------------
        excel_path = os.path.join(store_path, 'Eight_Formulas_Analysis.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 5.1 Save error metrics for each formula (one worksheet per formula)
            for formula_idx in range(8):
                valid_errors = [err for err in formula_errors[formula_idx] if err is not None]
                if not valid_errors:
                    continue
                df_error = pd.DataFrame(valid_errors)
                # Adjust column order (for readability)
                col_order = ['gamma0', 'formula_idx', 'formula_name', 'MSE', 'RMSE', 'MAE', 'RAE', 'MAPE', 'r', 'R^2']
                df_error = df_error[col_order]
                # Write to worksheet
                sheet_name = f'Formula{formula_idx+1}_Error'
                df_error.to_excel(writer, sheet_name=sheet_name, index=False)

            # 5.2 Save all raw data (DEM + 8 formula predictions)
            raw_data_rows = []
            for gam_data in all_raw_data:
                gamma0 = gam_data['gamma0']
                # Write DEM data
                for t_val, miu_val in zip(gam_data['t_dem'], gam_data['miu_dem']):
                    raw_data_rows.append({
                        'gamma0': gamma0,
                        'data_type': 'DEM',
                        'formula_idx': None,
                        'formula_name': None,
                        't': t_val,
                        'miu': miu_val
                    })
                # Write prediction data from 8 formulas
                for formula_idx in range(8):
                    formula_name = self.formula_labels[formula_idx]
                    for t_val, miu_val in zip(gam_data['t_predict'], gam_data['miu_predict'][formula_idx]):
                        raw_data_rows.append({
                            'gamma0': gamma0,
                            'data_type': 'Prediction',
                            'formula_idx': formula_idx + 1,
                            'formula_name': formula_name,
                            't': t_val,
                            'miu': miu_val
                        })
            df_raw = pd.DataFrame(raw_data_rows)
            df_raw.to_excel(writer, sheet_name='All_Raw_Data', index=False)

        print(f"Analysis report saved: {excel_path}")
        print("Comparison analysis of 8 formulas completed!")


# -------------------------- Main program entry --------------------------
if __name__ == "__main__":
    # 1. Initialize timestamp (for result folder naming)
    today_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 2. Configure paths and parameters (consistent with original code)
    directory = os.getcwd()
    directory_path = f'H:/Batch_OscillatoryShear_5/Heaviside/'  # Root path for data
    file_prefix = f'p10000_ds0.005_rho2500_nH15_T10_gam'  # Prefix for data file names
    name_array = np.array([0.10])  # gamma0 values to analyze
    T = 10  # Period (consistent with original code)

    # 3. Create instance and execute comparison analysis
    postprocessor = PostprocessResults(today_datetime, T=T)
    postprocessor.compare_eight_formulas(
        name_array=name_array,
        directory=directory,
        directory_path=directory_path,
        file_prefix=file_prefix
    )
