# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:20:53 2024

The ReadFolder class provides functionality to read folder names from a specified directory,
classify them based on naming patterns, and return structured results. It supports various
folder naming conventions and can export classification results to CSV format.

@author: Han Xu
"""

# import numpy as np
import os
import re
from collections import defaultdict
import csv

class ReadFolder:
    """
    A utility class for reading and classifying folder structures based on naming conventions.
    Provides methods to identify specific folder patterns and export classification results.
    """
    
    def read_thermo_log(self, folder_path):
        """
        Reads the content of thermo.log file from a specified folder.
        
        Args:
            folder_path (str): Path to the folder containing thermo.log
            
        Returns:
            list: Lines from thermo.log file, or empty list if file doesn't exist
        """
        thermo_file = os.path.join(folder_path, 'thermo.log')
        if os.path.exists(thermo_file):
            with open(thermo_file, 'r') as file:
                return file.readlines()
        return []
     
    def list_numeric_folders(self, directory):
        """
        Identifies folders whose names start with numeric characters.
        
        Args:
            directory (str): Path to the parent directory
            
        Returns:
            list: Full paths of folders with numeric starting characters
        """
        numeric_folders = []
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            # Check if item is a directory and name starts with a digit
            if os.path.isdir(item_path) and item[0].isdigit():
                numeric_folders.append(item_path)
 
        return numeric_folders
    
    def return_dimensionless_cases(self, directory_path):
        """
        Classifies folders matching the pattern: p{num}_ds{float}_rho{num}_nH{num}_T{num}
        
        Args:
            directory_path (str): Path to the directory containing folders to classify
            
        Returns:
            list: Dictionary entries with 'category', 'count', and 'folders' for each classification
        """
        pattern = re.compile(r'p(\d+)_ds(\d+\.\d+)_rho(\d+)_nH(\d+)_T(\d+)')
        categories = defaultdict(list)

        for folder_name in os.listdir(directory_path):
            match = pattern.match(folder_name)
            if match:
                category = f'p{match.group(1)}_ds{match.group(2)}_rho{match.group(3)}_nH{match.group(4)}_T{match.group(5)}'
                categories[category].append(folder_name)

        return [{'category': cat, 'count': len(folders), 'folders': folders} 
                for cat, folders in categories.items()]
    
    def return_dimensionless_cases_7items(self, directory_path):
        """
        Classifies folders matching the 7-parameter pattern: 
        p{num}_ds{float}_rho{num}_nH{num}_T{num}_Ep{val}_mu{float}
        
        Args:
            directory_path (str): Path to the directory containing folders to classify
            
        Returns:
            list: Dictionary entries with 'category', 'count', and 'folders' for each classification
        """
        pattern = re.compile(r'p(\d+)_ds(\d+\.\d+)_rho(\d+)_nH(\d+)_T(\d+)_Ep([^_]+)_mu(\d+\.\d+)')
        categories = defaultdict(list)

        for folder_name in os.listdir(directory_path):
            match = pattern.match(folder_name)
            if match:
                category = (f'p{match.group(1)}_ds{match.group(2)}_rho{match.group(3)}_nH{match.group(4)}'
                           f'_T{match.group(5)}_Ep{match.group(6)}_mu{match.group(7)}')
                categories[category].append(folder_name)

        return [{'category': cat, 'count': len(folders), 'folders': folders} 
                for cat, folders in categories.items()]
    
    def return_Single_Ep_cases(self, directory_path):
        """
        Classifies folders matching the pattern with single Ep parameter:
        p{num}_ds{float}_rho{num}_nH{num}_T{num}_Ep{val}
        
        Args:
            directory_path (str): Path to the directory containing folders to classify
            
        Returns:
            list: Dictionary entries with 'category', 'count', and 'folders' for each classification
        """
        pattern = re.compile(r'p(\d+)_ds(\d+\.\d+)_rho(\d+)_nH(\d+)_T(\d+)_Ep([^_]+)')
        categories = defaultdict(list)

        for folder_name in os.listdir(directory_path):
            match = pattern.match(folder_name)
            if match:
                category = (f'p{match.group(1)}_ds{match.group(2)}_rho{match.group(3)}_nH{match.group(4)}'
                           f'_T{match.group(5)}_Ep{match.group(6)}')
                categories[category].append(folder_name)

        return [{'category': cat, 'count': len(folders), 'folders': folders} 
                for cat, folders in categories.items()]
    
    def save_categories_to_csv(self, classification_results, output_file):
        """
        Exports classification results to a CSV file containing category information.
        
        Args:
            classification_results (list): Results from classification methods
            output_file (str): Path for the output CSV file
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in classification_results:
                writer.writerow({'category': item['category']})

if __name__ == "__main__":
    directory_path = 'H:/Batch_OscillatoryShear_8_Heaviside/all_cases/'
    
    reader = ReadFolder()
    
    # Get classification results using the 7-parameter method
    classification_results = reader.return_dimensionless_cases_7items(directory_path)
    
    # Print results
    for item in classification_results:
        print(f"Category: {item['category']}, Count: {item['count']}, Folders: {item['folders']}")
    
    # Save categories to CSV
    output_csv = 'classification_categories_100.csv'
    reader.save_categories_to_csv(classification_results, output_csv)
    print(f"Category information saved to {output_csv}")
