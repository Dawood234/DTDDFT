#!/usr/bin/env python3
"""
Enhanced DTDDFT Analysis Script v2.0

Improvements over v1:
- Centralized configuration management  
- Robust data loading with validation
- Enhanced numerical stability
- Better convergence algorithms
- Modular design with clean interfaces
- Comprehensive error handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Configure matplotlib
mpl.rcParams['figure.dpi'] = 100

@dataclass
class AnalysisConfig:
    """Centralized configuration management"""
    # Computational settings
    basis_options: List[str] = None
    basis: str = 'cc-pVDZ'
    functional: str = 'pbe96'
    method: str = 'tddft'
    
    # Physical constants
    au_to_ev: float = 27.2114
    
    # Numerical settings
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    damping_factor: float = 0.1
    
    # File paths (relative to analysis directory)
    data_dir: str = '../'
    reference_data_path: str = '../ex_data.txt'  # User's reference data file
    
    # BLA geometry array
    bla_array: np.ndarray = None
    
    def __post_init__(self):
        if self.basis_options is None:
            self.basis_options = ['def2-SVP', 'def2-TZVP', 'def2-TZVPP', 'cc-pVDZ', 'cc-pVTZ']
        
        if self.bla_array is None:
            self.bla_array = np.array([
                0.125466, 0.110464, 0.0954646, 0.0803673, 0.0652275, 0.0500952,
                0.0349265, 0.0198214, 0.00452051, -0.0107809, -0.0138626, -0.0200338,
                -0.0230615, -0.0262975, -0.0295307, -0.0327402, -0.0359374, -0.0390515,
                -0.0423162, -0.0457886, -0.0490522, -0.0523145, -0.0556133, -0.062,
                -0.075, -0.088, -0.099, -0.111, -0.122, -0.134, -0.144
            ])
    
    @property
    def folder_name(self) -> str:
        return f"{self.method}_{self.basis}_{self.functional}"
    
    @property
    def data_file_path(self) -> str:
        return f"{self.data_dir}data_{self.folder_name}.txt"
    
    @property
    def apb_file_path(self) -> str:
        return f"{self.data_dir}apb_{self.folder_name}"
    
    @property
    def sqrtamb_file_path(self) -> str:
        return f"{self.data_dir}sqrtamb_{self.folder_name}"


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class ConvergenceError(Exception):
    """Custom exception for convergence failures"""
    pass


class DataLoader:
    """Enhanced data loading with validation and error handling"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._vector_data = None
        self._matrix_data_cache = {}
        
    def load_vector_data(self) -> Dict[str, np.ndarray]:
        """Load and validate vector data from standardized data file"""
        if self._vector_data is not None:
            return self._vector_data
            
        data_file = self.config.data_file_path
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading vector data from: {data_file}")
        data_dict = {}
        
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                if '=' in line and 'np.array' in line and not line.strip().startswith('#'):
                    try:
                        var_name = line.split('=')[0].strip()
                        array_str = line.split('np.array(')[1].split(')')[0]
                        array_data = eval(array_str)
                        data_dict[var_name] = np.array(array_data, dtype=float)
                    except Exception as e:
                        warnings.warn(f"Could not parse line {line_num}: {line.strip()}: {e}")
                        continue
                        
        except Exception as e:
            raise DataValidationError(f"Error reading data file {data_file}: {e}")
        
        self._validate_vector_data(data_dict)
        self._vector_data = data_dict
        
        # Debug: show what vectors were found
        print(f"   📋 Found {len(data_dict)} vectors:")
        for key in list(data_dict.keys())[:10]:  # Show first 10
            print(f"      {key} (length: {len(data_dict[key])})")
        if len(data_dict) > 10:
            print(f"      ... and {len(data_dict) - 10} more")
        
        return data_dict
    
    def _validate_vector_data(self, data_dict: Dict[str, np.ndarray]) -> None:
        """Validate loaded vector data"""
        required_vars = ['gs', 'mo14', 'mo15', 'mo16', 'mo17', 'Hq1d', 'Hq2d']
        required_vars.extend([f'ag_{self.config.method}', f'bu_{self.config.method}'])
        
        missing_vars = []
        for var in required_vars:
            # Check with various naming conventions
            found = False
            for key in data_dict.keys():
                if var in key or key in var:
                    found = True
                    break
            if not found:
                missing_vars.append(var)
        
        if missing_vars:
            warnings.warn(f"Missing variables: {missing_vars}")
        
        # Validate array lengths match BLA array
        expected_length = len(self.config.bla_array)
        for var_name, array in data_dict.items():
            if len(array) != expected_length:
                warnings.warn(f"Array {var_name} has length {len(array)}, expected {expected_length}")
    
    def load_matrix_data(self, filename: str) -> Optional[np.ndarray]:
        """Load and validate matrix data with caching"""
        if filename in self._matrix_data_cache:
            return self._matrix_data_cache[filename]
            
        if not os.path.exists(filename):
            warnings.warn(f"Matrix file not found: {filename}")
            return None
        
        print(f"Loading matrix data from: {filename}")
        
        try:
            with open(filename, 'r') as file:
                text = file.read()
            
            if not text or text.strip() == "":
                warnings.warn(f"Empty matrix file: {filename}")
                return None
            
            sections = re.split(r'\n-{3,}\n|\n-{3,}$', text, flags=re.MULTILINE)
            matrices_text = sections[1:]
            num_list = []
            
            for i in range(len(matrices_text)):
                if i % 2 != 0 and i != 0:
                    secc = sections[i].split('\n')
                    if len(secc) >= 5:
                        try:
                            line1, line2 = secc[3], secc[4]
                            numline1, numline2 = line1.split(), line2.split()
                            
                            if len(numline1) >= 4 and len(numline2) >= 4:
                                matrix = np.array([
                                    [float(numline1[2]), float(numline1[3])],
                                    [float(numline2[2]), float(numline2[3])]
                                ])
                                num_list.append(matrix)
                        except (IndexError, ValueError) as e:
                            warnings.warn(f"Could not parse matrix section {i} in {filename}: {e}")
                            continue
            
            if not num_list:
                warnings.warn(f"No valid matrices found in {filename}")
                return None
                
            result = np.array(num_list)
            self._matrix_data_cache[filename] = result
            return result
            
        except Exception as e:
            raise DataValidationError(f"Error reading matrix file {filename}: {e}")
    
    def get_vector_by_names(self, data_dict: Dict[str, np.ndarray], 
                           possible_names: List[str]) -> Optional[np.ndarray]:
        """Get vector data by trying multiple possible names"""
        for name in possible_names:
            if name in data_dict:
                return data_dict[name]
        return None


class EnhancedTDDFTCalculation:
    """Enhanced TDDFT calculation with better error handling and validation"""
    
    _matrix_cache = {}
    
    def __init__(self, index: int, data_loader: DataLoader, config: AnalysisConfig):
        self.index = index
        self.config = config
        self.data_loader = data_loader
        
        # Load and validate data
        self._load_data()
        self._validate_data()
    
    def _load_data(self) -> None:
        """Load all required data for this calculation"""
        vector_data = self.data_loader.load_vector_data()
        
        # Load vector data
        self.gs = self._get_vector(vector_data, [f'gs_{self.config.folder_name}', 'gs'])
        self.mo14 = self._get_vector(vector_data, ['mo14'])
        self.mo15 = self._get_vector(vector_data, ['mo15'])
        self.mo16 = self._get_vector(vector_data, ['mo16'])
        self.mo17 = self._get_vector(vector_data, ['mo17'])
        
        # Load excited state data and convert units
        ag_vec = self._get_vector(vector_data, [f'ag_{self.config.method}', 'ag'])
        bu_vec = self._get_vector(vector_data, [f'bu_{self.config.method}', 'bu'])
        
        self.om_tddft = ag_vec[self.index] / self.config.au_to_ev if ag_vec is not None else None
        self.om_d = 2 * bu_vec[self.index] / self.config.au_to_ev if bu_vec is not None else None
        
        # Load coupling elements
        Hq1d_vec = self._get_vector(vector_data, ['Hq1d'])
        Hq2d_vec = self._get_vector(vector_data, ['Hq2d'])
        
        self.Hq1d = -2 / np.sqrt(2) * Hq1d_vec[self.index] if Hq1d_vec is not None else None
        self.Hq2d = 2 / np.sqrt(2) * Hq2d_vec[self.index] if Hq2d_vec is not None else None
        self.Hdq1 = self.Hq1d
        self.Hdq2 = self.Hq2d
    
    def _get_vector(self, data_dict: Dict[str, np.ndarray], 
                   possible_names: List[str]) -> Optional[np.ndarray]:
        """Helper to get vector data with multiple name options"""
        return self.data_loader.get_vector_by_names(data_dict, possible_names)
    
    def _validate_data(self) -> None:
        """Validate that all required data is available"""
        required_attrs = ['om_tddft', 'om_d', 'mo14', 'mo15', 'mo16', 'mo17', 
                         'Hq1d', 'Hq2d', 'Hdq1', 'Hdq2']
        
        missing_data = []
        for attr in required_attrs:
            value = getattr(self, attr, None)
            # Use 'is None' for safe None checking with arrays
            if value is None:
                missing_data.append(attr)
            elif hasattr(value, '__len__') and len(value) <= self.index:
                missing_data.append(f"{attr}[{self.index}]")
        
        if missing_data:
            raise DataValidationError(f"Missing data for calculation {self.index}: {missing_data}")
    
    def calculate_nu(self) -> Tuple[float, float, float]:
        """Calculate nu parameters with validation"""
        try:
            nu1 = self.mo16[self.index] - self.mo14[self.index]
            nu2 = self.mo17[self.index] - self.mo15[self.index]
            nud = 2 * (self.mo16[self.index] - self.mo15[self.index])
            
            # Validate that nu values are reasonable
            for name, value in [('nu1', nu1), ('nu2', nu2), ('nud', nud)]:
                if not np.isfinite(value):
                    raise ValueError(f"{name} is not finite: {value}")
                if abs(value) < 1e-10:
                    warnings.warn(f"{name} is very small: {value}")
            
            return nu1, nu2, nud
            
        except Exception as e:
            raise DataValidationError(f"Error calculating nu parameters for index {self.index}: {e}")
    
    def get_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load matrix data with caching and validation"""
        cache_key = (self.config.apb_file_path, self.config.sqrtamb_file_path)
        
        if cache_key not in self._matrix_cache:
            apb_data = self.data_loader.load_matrix_data(self.config.apb_file_path)
            
            if self.config.method.lower() == 'tda':
                sqrtamb_data = apb_data  # For TDA, sqrt(A-B) = A+B
            else:
                sqrtamb_data = self.data_loader.load_matrix_data(self.config.sqrtamb_file_path)
            
            self._matrix_cache[cache_key] = (apb_data, sqrtamb_data)
        
        apb_data, sqrtamb_data = self._matrix_cache[cache_key]
        
        # Get matrices for this index
        apb = apb_data[self.index] if apb_data is not None and self.index < len(apb_data) else None
        sqrtamb = sqrtamb_data[self.index] if sqrtamb_data is not None and self.index < len(sqrtamb_data) else None
        
        return apb, sqrtamb
    
    def get_calculation_data(self) -> Dict[str, Any]:
        """Get all calculation data in a structured format"""
        nu1, nu2, nud = self.calculate_nu()
        apb, sqrtamb = self.get_matrices()
        
        return {
            'index': self.index,
            'om_tddft': self.om_tddft,
            'om_d': self.om_d,
            'nu1': nu1,
            'nu2': nu2,
            'nud': nud,
            'apb': apb,
            'sqrtamb': sqrtamb,
            'Hq1d': self.Hq1d,
            'Hq2d': self.Hq2d,
            'Hdq1': self.Hdq1,
            'Hdq2': self.Hdq2,
            'mo_energies': {
                'mo14': self.mo14[self.index],
                'mo15': self.mo15[self.index],
                'mo16': self.mo16[self.index],
                'mo17': self.mo17[self.index]
            }
        }


class MathematicalFunctions:
    """Enhanced mathematical functions with numerical stability"""
    
    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                   fallback: float = 0.0, epsilon: float = 1e-12) -> np.ndarray:
        """Safe division with fallback for near-zero denominators"""
        mask = np.abs(denominator) > epsilon
        result = np.full_like(numerator, fallback, dtype=float)
        result[mask] = numerator[mask] / denominator[mask]
        return result
    
    @staticmethod
    def validate_matrix(matrix: np.ndarray, name: str = "matrix") -> None:
        """Validate matrix properties"""
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.linalg.cond(matrix) > 1e12:
            warnings.warn(f"{name} is ill-conditioned (condition number: {np.linalg.cond(matrix):.2e})")
    
    @staticmethod
    def dress_interaction(om: float, nu1: float, nu2: float, om_d: float,
                         Hq1d: float, Hq2d: float, Hdq1: float, Hdq2: float,
                         epsilon: float = 1e-12) -> np.ndarray:
        """Enhanced dress interaction matrix with numerical stability"""
        
        # Validate inputs
        for name, value in [('om', om), ('nu1', nu1), ('nu2', nu2), ('om_d', om_d)]:
            if not np.isfinite(value):
                raise ValueError(f"{name} is not finite: {value}")
        
        # Check for problematic denominators
        denom_check = om**2 - om_d**2
        if abs(denom_check) < epsilon:
            warnings.warn(f"Small denominator in dress function: {denom_check}")
        
        # Calculate components with numerical stability
        factor1 = (nu1 + om_d)**2 / (om**2 - om_d**2) if abs(om**2 - om_d**2) > epsilon else 0
        factor2 = (nu1 + om_d) * (nu2 + om_d) / (om**2 - om_d**2) if abs(om**2 - om_d**2) > epsilon else 0
        factor3 = (nu2 + om_d)**2 / (om**2 - om_d**2) if abs(om**2 - om_d**2) > epsilon else 0
        
        # Construct matrix
        dress_matrix = np.array([
            [Hq1d * Hdq1 / (4 * np.sqrt(nu1**2)) * (1 + factor1),
             Hq1d * Hdq2 / (4 * np.sqrt(nu1 * nu2)) * (1 + factor2)],
            [Hq2d * Hdq1 / (4 * np.sqrt(nu2 * nu1)) * (1 + factor2),
             Hq2d * Hdq2 / (4 * np.sqrt(nu2**2)) * (1 + factor3)]
        ])
        
        MathematicalFunctions.validate_matrix(dress_matrix, "dress_matrix")
        return dress_matrix
    
    @staticmethod
    def dtda_dress_interaction(om: float, om_d: float, Hq1d: float, Hq2d: float,
                              Hdq1: float, Hdq2: float, epsilon: float = 1e-12) -> np.ndarray:
        """DTDA dress interaction with numerical stability"""
        
        denom = om - om_d
        if abs(denom) < epsilon:
            warnings.warn(f"Small denominator in DTDA dress: {denom}")
            denom = epsilon if denom >= 0 else -epsilon
        
        dress_matrix = np.array([
            [Hq1d * Hdq1 / denom, Hq1d * Hdq2 / denom],
            [Hq2d * Hdq1 / denom, Hq2d * Hdq2 / denom]
        ])
        
        MathematicalFunctions.validate_matrix(dress_matrix, "dtda_dress_matrix")
        return dress_matrix


class ConvergenceSolver:
    """Enhanced convergence algorithms with damping and robust error handling"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def solve_with_damping(self, compute_func, diagonalize_func, initial_guess: float,
                          method_name: str = "unknown") -> Tuple[np.ndarray, np.ndarray]:
        """Solve eigenvalue problem with damped convergence"""
        
        print(f"Starting {method_name} convergence from initial guess: {initial_guess:.6f}")
        
        current_om = initial_guess
        converged = False
        iterations = 0
        convergence_history = [current_om]
        
        while not converged and iterations < self.config.max_iterations:
            try:
                # Compute matrix and diagonalize
                matrix = compute_func(current_om)
                MathematicalFunctions.validate_matrix(matrix, f"{method_name}_matrix")
                
                eigenvalues, eigenvectors = diagonalize_func(matrix)
                new_om = eigenvalues[0]
                
                # Check convergence
                error = abs(new_om - current_om)
                if error < self.config.convergence_tolerance:
                    converged = True
                    print(f"{method_name} converged in {iterations} iterations. Final error: {error:.2e}")
                else:
                    # Apply damping
                    damped_om = (1 - self.config.damping_factor) * new_om + self.config.damping_factor * current_om
                    current_om = damped_om
                    convergence_history.append(current_om)
                
                iterations += 1
                
                # Check for oscillations
                if iterations > 10 and len(convergence_history) >= 4:
                    recent_values = convergence_history[-4:]
                    if abs(max(recent_values) - min(recent_values)) < self.config.convergence_tolerance:
                        print(f"{method_name} detected oscillation, accepting current value")
                        converged = True
                
            except Exception as e:
                raise ConvergenceError(f"{method_name} failed at iteration {iterations}: {e}")
        
        if not converged:
            raise ConvergenceError(f"{method_name} failed to converge after {self.config.max_iterations} iterations")
        
        return eigenvalues, eigenvectors


# Mathematical analysis functions will be continued in the next section...
def renormalized_oscillator_strengths(e1: float, e2: float, F1: np.ndarray, F2: np.ndarray,
                                     Hq1d: float, Hq2d: float, Hdq1: float, Hdq2: float,
                                     nu1: float, nu2: float, om_d: float) -> Tuple[float, float]:
    """Enhanced oscillator strength calculation with validation"""
    
    # Validate inputs
    for name, value in [('e1', e1), ('e2', e2), ('nu1', nu1), ('nu2', nu2), ('om_d', om_d)]:
        if not np.isfinite(value):
            raise ValueError(f"{name} is not finite: {value}")
    
    omega1_squared = e1**2
    omega2_squared = e2**2
    
    # Construct coupling matrix
    beta = np.array([
        [Hq1d * Hq1d, Hq1d * Hq2d],
        [Hq2d * Hdq1, Hdq2 * Hdq2]
    ])
    
    # Energy coupling matrix
    C = np.array([
        [(nu1 + om_d)**2, (nu1 + om_d) * (nu2 + om_d)],
        [(nu2 + om_d) * (nu1 + om_d), (nu2 + om_d)**2]
    ])
    
    D = np.array([[om_d**2, om_d**2], [om_d**2, om_d**2]])
    
    # Calculate renormalization factors with numerical stability
    numerator = beta * C
    
    # Safe division for denominators
    denom1 = (omega1_squared - D)**2
    denom2 = (omega2_squared - D)**2
    
    result_matrix_1 = -MathematicalFunctions.safe_divide(numerator, denom1, fallback=0.0)
    result_matrix_2 = -MathematicalFunctions.safe_divide(numerator, denom2, fallback=0.0)
    
    # Calculate renormalization
    delta1 = np.eye(2) - result_matrix_1
    delta2 = np.eye(2) - result_matrix_2
    
    # Validate delta matrices
    MathematicalFunctions.validate_matrix(delta1, "delta1")
    MathematicalFunctions.validate_matrix(delta2, "delta2")
    
    # Calculate renormalization factors
    try:
        renorm_factor_1 = np.sqrt(F1.T @ delta1 @ F1)
        renorm_factor_2 = np.sqrt(F2.T @ delta2 @ F2)
        
        if abs(renorm_factor_1) < 1e-12 or abs(renorm_factor_2) < 1e-12:
            warnings.warn("Very small renormalization factors detected")
        
        F1_renorm = F1 / renorm_factor_1
        F2_renorm = F2 / renorm_factor_2
        
        return np.linalg.norm(F1_renorm)**2, np.linalg.norm(F2_renorm)**2
        
    except Exception as e:
        raise ValueError(f"Error in oscillator strength calculation: {e}")


class DTDDFTAnalyzer:
    """Enhanced DTDDFT analysis with improved algorithms and error handling"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.math_funcs = MathematicalFunctions()
        self.solver = ConvergenceSolver(config)
        
        print(f"Initialized DTDDFT Analyzer for: {config.folder_name}")
    
    def dtddft_analysis(self, calc_data: Dict[str, Any], use_mazur: bool = True) -> Tuple[Tuple[float, float], np.ndarray]:
        """Enhanced DTDDFT analysis with optional Mazur self-consistency"""
        
        try:
            # Extract data
            om_tddft = calc_data['om_tddft']
            sqrtamb = calc_data['sqrtamb']
            apb = calc_data['apb']
            nu1, nu2 = calc_data['nu1'], calc_data['nu2']
            om_d = calc_data['om_d']
            Hq1d, Hq2d = calc_data['Hq1d'], calc_data['Hq2d']
            Hdq1, Hdq2 = calc_data['Hdq1'], calc_data['Hdq2']
            
            # Safe None checking for arrays
            if sqrtamb is None or apb is None:
                raise ValueError("Missing matrix data for DTDDFT analysis")
            
            if use_mazur:
                return self._mazur_dtddft(om_tddft, sqrtamb, apb, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
            else:
                return self._direct_dtddft(om_tddft, sqrtamb, apb, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
                
        except Exception as e:
            raise ValueError(f"DTDDFT analysis failed: {e}")
    
    def _direct_dtddft(self, om_tddft: float, sqrtamb: np.ndarray, apb: np.ndarray,
                       Hq1d: float, Hq2d: float, Hdq1: float, Hdq2: float,
                       nu1: float, nu2: float, om_d: float) -> Tuple[Tuple[float, float], np.ndarray]:
        """Direct DTDDFT calculation without self-consistency"""
        
        # Calculate dress interaction
        dress_matrix = self.math_funcs.dress_interaction(om_tddft, nu1, nu2, om_d, Hq1d, Hq2d, Hdq1, Hdq2)
        
        # Dressed A+B matrix
        apb_dressed = apb + 2 * dress_matrix
        
        # Construct and diagonalize
        C = sqrtamb @ apb_dressed @ sqrtamb
        self.math_funcs.validate_matrix(C, "DTDDFT_matrix")
        
        eigenvalues, eigenvectors = np.linalg.eig(C)
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Extract first two states
        F1, F2 = sorted_eigenvectors[:, 0], sorted_eigenvectors[:, 1]
        e1, e2 = np.sqrt(sorted_eigenvalues[0]), np.sqrt(sorted_eigenvalues[1])
        
        # Calculate oscillator strengths
        osc_strengths = renormalized_oscillator_strengths(e1, e2, F1, F2, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
        energies = np.array([e1, e2]) * self.config.au_to_ev
        
        return osc_strengths, energies
    
    def _mazur_dtddft(self, om_tddft: float, sqrtamb: np.ndarray, apb: np.ndarray,
                      Hq1d: float, Hq2d: float, Hdq1: float, Hdq2: float,
                      nu1: float, nu2: float, om_d: float) -> Tuple[Tuple[float, float], np.ndarray]:
        """Mazur self-consistent DTDDFT calculation"""
        
        def compute_matrix(om: float) -> np.ndarray:
            dress_matrix = self.math_funcs.dress_interaction(om, nu1, nu2, om_d, Hq1d, Hq2d, Hdq1, Hdq2)
            apb_dressed = apb + 2 * dress_matrix
            return sqrtamb @ apb_dressed @ sqrtamb
        
        def diagonalize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            sorted_indices = np.argsort(eigenvalues)
            sorted_eigenvalues = np.sqrt(eigenvalues[sorted_indices])
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            return sorted_eigenvalues, sorted_eigenvectors
        
        # Solve with enhanced convergence
        eigenvalues, eigenvectors = self.solver.solve_with_damping(
            compute_matrix, diagonalize, om_tddft, "Mazur-DTDDFT"
        )
        
        # Extract results
        F1, F2 = eigenvectors[:, 0], eigenvectors[:, 1]
        e1, e2 = eigenvalues[0], eigenvalues[1]
        
        # Calculate oscillator strengths
        osc_strengths = renormalized_oscillator_strengths(e1, e2, F1, F2, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
        energies = eigenvalues * self.config.au_to_ev
        
        return osc_strengths, energies
    
    def dtda_analysis(self, calc_data: Dict[str, Any], use_mazur: bool = True) -> Tuple[Tuple[float, float], np.ndarray]:
        """Enhanced DTDA analysis"""
        
        try:
            # Extract data
            om_tddft = calc_data['om_tddft']
            apb = calc_data['apb']
            nu1, nu2 = calc_data['nu1'], calc_data['nu2']
            om_d = calc_data['om_d']
            Hq1d, Hq2d = calc_data['Hq1d'], calc_data['Hq2d']
            Hdq1, Hdq2 = calc_data['Hdq1'], calc_data['Hdq2']
            
            # Safe None checking for arrays
            if apb is None:
                raise ValueError("Missing APB matrix data for DTDA analysis")
            
            if use_mazur:
                return self._mazur_dtda(om_tddft, apb, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
            else:
                return self._direct_dtda(om_tddft, apb, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
                
        except Exception as e:
            raise ValueError(f"DTDA analysis failed: {e}")
    
    def _direct_dtda(self, om_tddft: float, apb: np.ndarray, Hq1d: float, Hq2d: float,
                     Hdq1: float, Hdq2: float, nu1: float, nu2: float, om_d: float) -> Tuple[Tuple[float, float], np.ndarray]:
        """Direct DTDA calculation"""
        
        # Calculate dress interaction
        dress_matrix = self.math_funcs.dtda_dress_interaction(om_tddft, om_d, Hq1d, Hq2d, Hdq1, Hdq2)
        
        # Dressed A matrix
        aa_dressed = apb + dress_matrix
        
        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eig(aa_dressed)
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Extract first two states
        F1, F2 = sorted_eigenvectors[:, 0], sorted_eigenvectors[:, 1]
        e1, e2 = sorted_eigenvalues[0], sorted_eigenvalues[1]
        
        # Calculate oscillator strengths
        osc_strengths = renormalized_oscillator_strengths(e1, e2, F1, F2, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
        energies = sorted_eigenvalues * self.config.au_to_ev
        
        return osc_strengths, energies
    
    def _mazur_dtda(self, om_tddft: float, apb: np.ndarray, Hq1d: float, Hq2d: float,
                    Hdq1: float, Hdq2: float, nu1: float, nu2: float, om_d: float) -> Tuple[Tuple[float, float], np.ndarray]:
        """Mazur self-consistent DTDA calculation"""
        
        def compute_matrix(om: float) -> np.ndarray:
            dress_matrix = self.math_funcs.dtda_dress_interaction(om, om_d, Hq1d, Hq2d, Hdq1, Hdq2)
            return apb + dress_matrix
        
        def diagonalize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            sorted_indices = np.argsort(eigenvalues)
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            return sorted_eigenvalues, sorted_eigenvectors
        
        # Solve with enhanced convergence
        eigenvalues, eigenvectors = self.solver.solve_with_damping(
            compute_matrix, diagonalize, om_tddft, "Mazur-DTDA"
        )
        
        # Extract results
        F1, F2 = eigenvectors[:, 0], eigenvectors[:, 1]
        e1, e2 = eigenvalues[0], eigenvalues[1]
        
        # Calculate oscillator strengths
        osc_strengths = renormalized_oscillator_strengths(e1, e2, F1, F2, Hq1d, Hq2d, Hdq1, Hdq2, nu1, nu2, om_d)
        energies = eigenvalues * self.config.au_to_ev
        
        return osc_strengths, energies


class AnalysisInterface:
    """Clean interface for running complete DTDDFT analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer = DTDDFTAnalyzer(config)
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        """Load reference data for comparison"""
        try:
            if os.path.exists(self.config.reference_data_path):
                # Handle both CSV and space-separated formats
                if self.config.reference_data_path.endswith('.csv'):
                    data = pd.read_csv(self.config.reference_data_path)
                else:
                    # Space-separated format like ex_data.txt
                    data = pd.read_csv(self.config.reference_data_path, sep=r'\s+', comment='#')
                    # Rename columns to match expected format
                    if len(data.columns) >= 3:
                        data.columns = ['x', 'bu', 'ag']
                
                print(f"Loaded reference data: {len(data)} points")
                return data
            else:
                warnings.warn(f"Reference data file not found: {self.config.reference_data_path}")
                return None
        except Exception as e:
            warnings.warn(f"Error loading reference data: {e}")
            return None
    
    def run_complete_analysis(self, method_type: str = 'dtddft', use_mazur: bool = True) -> Dict[str, Any]:
        """Run complete analysis for all BLA points"""
        
        print(f"\nRunning {method_type.upper()} analysis ({'Mazur' if use_mazur else 'Direct'}) for {len(self.config.bla_array)} geometries...")
        
        results = {
            'energies_1': [],
            'energies_2': [],
            'oscillator_strengths_1': [],
            'oscillator_strengths_2': [],
            'bla_values': self.config.bla_array,
            'method': method_type,
            'mazur': use_mazur,
            'config': self.config
        }
        
        success_count = 0
        
        for i in range(len(self.config.bla_array)):
            try:
                # Create calculation object for this geometry
                calc = EnhancedTDDFTCalculation(i, self.analyzer.data_loader, self.config)
                calc_data = calc.get_calculation_data()
                
                # Run analysis
                if method_type.lower() == 'dtddft':
                    osc_strengths, energies = self.analyzer.dtddft_analysis(calc_data, use_mazur)
                elif method_type.lower() == 'dtda':
                    osc_strengths, energies = self.analyzer.dtda_analysis(calc_data, use_mazur)
                else:
                    raise ValueError(f"Unknown method type: {method_type}")
                
                # Store results
                results['oscillator_strengths_1'].append(osc_strengths[0])
                results['oscillator_strengths_2'].append(osc_strengths[1])
                results['energies_1'].append(energies[0])
                results['energies_2'].append(energies[1])
                
                success_count += 1
                
                if i % 5 == 0:  # Progress indicator
                    print(f"  Completed {i+1}/{len(self.config.bla_array)} calculations...")
                    
            except Exception as e:
                warnings.warn(f"Analysis failed for geometry {i} (BLA={self.config.bla_array[i]:.6f}): {e}")
                # Add NaN values to maintain array structure
                results['oscillator_strengths_1'].append(np.nan)
                results['oscillator_strengths_2'].append(np.nan)
                results['energies_1'].append(np.nan)
                results['energies_2'].append(np.nan)
        
        # Convert to numpy arrays
        for key in ['energies_1', 'energies_2', 'oscillator_strengths_1', 'oscillator_strengths_2']:
            results[key] = np.array(results[key])
        
        print(f"Analysis complete: {success_count}/{len(self.config.bla_array)} calculations successful")
        return results
    
    def create_comparison_plot(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Create publication-quality comparison plot with line plots
        
        Plot Contents:
        - DTDDFT State 1 & 2: Blue/Red solid lines (main results)
        - Reference Ag/Bu: Black lines with markers (experimental data)  
        - Original TDDFT Ag/Bu: Green/Orange dashed lines (comparison)
        - X-axis: BLA (Bond Length Alternation) in Ångström
        - Y-axis: Energy in eV
        """
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        print("📊 Plotting comparison chart...")
        print("   Lines plotted:")
        print(f"   • DTDDFT State 1 & 2 (main results)")
        print(f"   • Reference Ag/Bu data (if available)")  
        print(f"   • Original TDDFT Ag/Bu (for comparison)")
        
        # Extract data
        bla = results['bla_values']
        e1 = results['energies_1']
        e2 = results['energies_2']
        
        # Load and plot reference data if available
        if self.reference_data is not None:
            bla_ref = self.reference_data['x'].to_numpy()
            ag_ref = self.reference_data['ag'].to_numpy()
            bu_ref = self.reference_data['bu'].to_numpy()
            
            # Reference data as solid lines with markers
            ax.plot(bla_ref, bu_ref, 'k-o', markerfacecolor='None', label='Bu (Ref.)', 
                   markersize=6, linewidth=2, markevery=2)
            ax.plot(bla_ref, ag_ref, 'k-s', label='Ag (Ref.)', 
                   markersize=6, linewidth=2, markerfacecolor='k', markevery=2)
            print(f"   ✅ Reference data plotted ({len(bla_ref)} points)")
        else:
            print(f"   ⚠️  No reference data file found: {self.config.reference_data_path}")
        
        # Plot DTDDFT results as clean lines
        method_label = f"D{results['method'].upper()}"
        if results['mazur']:
            method_label += " (Mazur)"
        
        # Filter out NaN values for plotting
        valid_mask = np.isfinite(e1) & np.isfinite(e2)
        
        if np.any(valid_mask):
            # DTDDFT results as solid lines with occasional markers
            ax.plot(bla[valid_mask], e1[valid_mask], 'b-', label=f'{method_label} State 1', 
                   linewidth=3, marker='o', markevery=3, markersize=8, markerfacecolor='b')
            ax.plot(bla[valid_mask], e2[valid_mask], 'r-', label=f'{method_label} State 2', 
                   linewidth=3, marker='s', markevery=3, markersize=8, markerfacecolor='r')
        
        # Plot original TDDFT for comparison
        try:
            vector_data = self.analyzer.data_loader.load_vector_data()
            
            # More flexible vector name lookup
            ag_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [f'ag_{self.config.method}', 'ag_tddft', 'ag'])
            bu_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [f'bu_{self.config.method}', 'bu_tddft', 'bu'])
            gs_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [
                f'gs_{self.config.folder_name}', f'gs_{self.config.method}', 'gs_tddft', 'gs'
            ])
            
            print(f"   🔍 Looking for original TDDFT vectors:")
            print(f"      ag_vec: {'✅' if ag_vec is not None else '❌'}")
            print(f"      bu_vec: {'✅' if bu_vec is not None else '❌'}")
            print(f"      gs_vec: {'✅' if gs_vec is not None else '❌'}")
            
            if all(x is not None for x in [ag_vec, bu_vec, gs_vec]):
                # Shift energies relative to ground state
                ag_shifted = (ag_vec + gs_vec - gs_vec[0]) * self.config.au_to_ev
                bu_shifted = (bu_vec + gs_vec - gs_vec[0]) * self.config.au_to_ev
                
                # Original TDDFT as dashed lines
                ax.plot(bla, ag_shifted, 'g--', alpha=0.8, label='Ag (TDDFT)', 
                       linewidth=2, marker='^', markevery=4, markersize=6, markerfacecolor='g')
                ax.plot(bla, bu_shifted, 'orange', linestyle='--', alpha=0.8, label='Bu (TDDFT)', 
                       linewidth=2, marker='v', markevery=4, markersize=6, markerfacecolor='orange')
                print(f"   ✅ Original TDDFT data plotted ({len(ag_vec)} points)")
            else:
                print(f"   ⚠️  Missing some original TDDFT vectors - not plotted")
                
        except Exception as e:
            print(f"   ❌ Could not plot original TDDFT data: {e}")
            warnings.warn(f"Could not plot original TDDFT data: {e}")
        
        # Formatting
        ax.invert_xaxis()
        ax.set_xlabel(r'BLA (Å)', fontsize=15)
        ax.set_ylabel('Energy (eV)', fontsize=15)
        ax.set_title(f'{method_label}: {self.config.basis}, {self.config.functional}', fontsize=16)
        ax.legend(fontsize=13, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        ax.minorticks_on()
        
        plt.tight_layout()
        
        # Summary of what was plotted
        plot_summary = []
        if self.reference_data is not None:
            plot_summary.append("✅ Reference data (Ag/Bu experimental)")
        else:
            plot_summary.append("⚠️  No reference data available")
            
        valid_points = np.sum(np.isfinite(results['energies_1']))
        if valid_points > 0:
            plot_summary.append(f"✅ DTDDFT results ({valid_points}/{len(results['bla_values'])} points)")
        else:
            plot_summary.append("❌ No valid DTDDFT results")
            
        try:
            vector_data = self.analyzer.data_loader.load_vector_data()
            ag_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [f'ag_{self.config.method}', 'ag_tddft', 'ag'])
            bu_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [f'bu_{self.config.method}', 'bu_tddft', 'bu'])
            gs_vec = self.analyzer.data_loader.get_vector_by_names(vector_data, [
                f'gs_{self.config.folder_name}', f'gs_{self.config.method}', 'gs_tddft', 'gs'
            ])
            if all(x is not None for x in [ag_vec, bu_vec, gs_vec]):
                plot_summary.append("✅ Original TDDFT data (Ag/Bu comparison)")
            else:
                plot_summary.append("⚠️  Original TDDFT data incomplete")
        except:
            plot_summary.append("❌ Original TDDFT data unavailable")
        
        print("📊 Plot completed! Contents:")
        for item in plot_summary:
            print(f"   {item}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to CSV file"""
        
        if filename is None:
            method_name = results['method']
            mazur_suffix = "_mazur" if results['mazur'] else "_direct"
            filename = f"dtddft_results_{self.config.folder_name}{mazur_suffix}.csv"
        
        # Create DataFrame
        df = pd.DataFrame({
            'bla': results['bla_values'],
            'energy_1_ev': results['energies_1'],
            'energy_2_ev': results['energies_2'],
            'osc_strength_1': results['oscillator_strengths_1'],
            'osc_strength_2': results['oscillator_strengths_2']
        })
        
        # Add metadata
        df.attrs['method'] = results['method']
        df.attrs['mazur'] = results['mazur']
        df.attrs['basis'] = self.config.basis
        df.attrs['functional'] = self.config.functional
        
        # Save
        df.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")
        
        return filename


# Main execution interface
def main():
    """Main function demonstrating the enhanced analysis interface"""
    
    print("=" * 60)
    print("Enhanced DTDDFT Analysis v2.0")
    print("=" * 60)
    
    try:
        # Create configuration
        config = AnalysisConfig(
            basis='cc-pVDZ',
            functional='pbe96',
            method='tddft',
            convergence_tolerance=1e-6,
            max_iterations=50,
            damping_factor=0.1
        )
        
        print(f"Configuration: {config.folder_name}")
        print(f"Data file: {config.data_file_path}")
        print(f"APB file: {config.apb_file_path}")
        print(f"sqrt(A-B) file: {config.sqrtamb_file_path}")
        
        # Check if files exist
        import os
        for name, path in [("Data", config.data_file_path), 
                          ("APB", config.apb_file_path), 
                          ("sqrt(A-B)", config.sqrtamb_file_path)]:
            if os.path.exists(path):
                print(f"✅ {name} file found: {path}")
            else:
                print(f"❌ {name} file missing: {path}")
        
        # Initialize analysis interface
        print("\nInitializing analysis interface...")
        analysis = AnalysisInterface(config)
        
        # Run DTDDFT analysis
        print("\nStarting DTDDFT analysis...")
        results = analysis.run_complete_analysis(method_type='dtddft', use_mazur=True)
        
        # Create comparison plot
        print("\nCreating comparison plot...")
        analysis.create_comparison_plot(results)
        
        # Export results
        print("\nExporting results...")
        analysis.export_results(results)
        
        # Print summary
        valid_points = np.sum(np.isfinite(results['energies_1']))
        print(f"\n✅ Analysis Summary:")
        print(f"  Method: {results['method'].upper()} ({'Mazur' if results['mazur'] else 'Direct'})")
        print(f"  Configuration: {config.basis}/{config.functional}")
        print(f"  Valid calculations: {valid_points}/{len(config.bla_array)}")
        
        if valid_points > 0:
            print(f"  Energy range: {np.nanmin(results['energies_1']):.3f} - {np.nanmax(results['energies_1']):.3f} eV")
        else:
            print("  No valid calculations completed")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 