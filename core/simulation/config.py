"""
simulation/config.pyor all experiments.

Central configuration f
All parameters are defined here for reproducitility.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =====================================================================
# DATA GENERATION PARAMETERS
# =====================================================================

@dataclass
class ProcessConfig:
    """ Ground truth physical process paramameters."""

    # Time series length
    n_timesteps: int = 1000

    # Sensor names
    sensors: List[str] = None

    # AR(1) coefficients for each sensor (temporal correlation)
    ar_coefficients: Dict[str, float] = None

    # Noise levels (measurement error)
    noise_std: Dict[str, float] = None

    # Initial values
    initial_values: Dict[str, float] = None

    # Cross-correlations (which sensors affect each other)
    cross_correlation_pairs: List[tuple] = None
    cross_correlation_strength: float = 0.3

    def __post_init__(self):
        if self.sensors is None:
            self.sensors = ['temperature', 'humidity', 'pollution', 'vibration']

            if self.ar_coefficients is None:
                self.ar_coefficients = {
                    'temperature': 0.85, # High temporal smoothness
                    'humidity': 0.80,
                    'pollution': 0.70,
                    'vibration': 0.60 # More random
                }
            
            if self.noise_std is None:
                self.noise_std = {
                    'temperature': 0.5,
                    "humidity": 1.0,
                    "pollution": 2.0,
                    "vibration": 3.0
                }

            if self.initial_values is None:
                self.initial_values = {
                    'temperature': 20.0, # Celsius
                    'humidity': 60.0, # Percentage
                    'pollution': 50.0, # AQI
                    'vibration': 10.0  # Arbitrary units
                }

            if self.cross_correlation_pairs is None:
                # Temperature affects humitdity, pollution affects vibration
                self.cross_correlation_pairs = [
                    ('temperature', 'humidity'),
                    ('pollution', 'vibration')
                ]

# ================================================================================================
# MISSINGNESS MECHANISM PARAMETERS
# ================================================================================================

@dataclass
class MissingnessConfig:
    """Parameters for MCAR, MAR, MNAR mechanisms"""

    # Which sensor will have missing data
    target_sensor: str = 'pollution'

    # ------------- MCAR parameters ---------------
    mcar_probability: float = 0.3 # 30% missing 

    # ------------- MAR parameters ---------------

    mar_predictor: str = 'humidity' # Missingness depends on this
    mar_alpha: float = 0.1 # Logistic regression coefficient
    mar_intercept: float = -2.0 # Bias term

    # ------------- MNAR parameters ---------------
    mnar_beta: float = 0.05 # Coefficient on target itself
    mnar_intercept: float = -1.5 # Bias term

    # For sensitivity analysis
    missing_rates_to_test: List[float] = None

    def __post_init__(self):
        if self.missing_rates_to_test is None:
            self.missing_rates_to_test = [0.1, 0.2, 0.4, 0.5]

# =====================================================================
# IMPUTATION METHOD PARAMETERS
# =====================================================================

@dataclass
class ImputaionConfig:
    """ Parameters for imputation methods."""

    # Methods to test
    methods: List[str] = None

    # KNN specific
    knn_k: int = 5

    # Regression imputation: which sensors to use as predictors
    regression_predictors: List[str] = None

    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                'mean',
                'forward-fill',
                'linear_interpolation',
                'regression',
                'knn'
            ] 

        if self.regression_predictors is None:
            # Use all sensors except target for regression
            self.regression_predictors = [
                'temperature',
                'humidity',
                'vibration'
            ]
        

# =====================================================================
# EVALUATION METRICS PARAMETERS 
# =====================================================================

@dataclass
class EvaluationConfig:
    """ Metrics to compute"""

    metrics: List[str] = None

    # Confidence level for coverage metrics
    confidence_level: float = 0.95

    # Number of bootstrap samples for unceratainity
    n_bootstap_samples: int = 1000

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'bias',
                'varience_ratio',
                'rmse',
                'kl_divergence',
                'autocorrelation_preservation'
            ]

# =====================================================================
# VISUALIZATION PARAMETERS
# =====================================================================

@dataclass
class VisualizationConfig:
    """ Plot settings"""

    figure_size: tuple = (10, 6)
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: List[str] = None

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#2E86AB', # Blue
                '#A23B72', # Red
                '#F18F01', # Orange
                '#C73E1D', # Dark Red
                '#6A994E', # Green
            ]

# =====================================================================
# MASTER CONFIG
# =====================================================================

class ExperimentConfig:
    """ Master configuration object."""

    def __init__(self):
        self.process = ProcessConfig()
        self.missingness = MissingnessConfig()
        self.imputation = ImputaionConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        self.seed = RANDOM_SEED
    
    def summary(self):
        """ Print configuration summary."""
        print("=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"\n Random Seed: {self.seed}")
        print(f"\n Data Generation:")
        print(f" - Timesteps: {self.process.n_timesteps}")
        print(f" - Sensors: {' , '.join(self.process.sensors)}")
        print(f"\n Missingness Mechanism:")
        print(f" - Target: {self.missingness.target_sensor}")
        print(f" - MCAR rate: {self.missingness.mcar_probability}")
        print(f" - MAR predictor: {self.missingness.mar_predictor}")
        print(f"\n  Imputation:")
        print(f"  - Methods: {', '.join(self.imputation.methods)}")
        print(f"\n  Evaluation:")
        print(f"  - Metrics: {', '.join(self.evaluation.metrics)}")
        print("=" * 60)

# ===============================================================================
# UTILITY FUNCTIONs
# ===============================================================================

def get_config() -> ExperimentConfig:
    """ Factory function to get configuration."""
    return ExperimentConfig()

def save_config(config: ExperimentConfig, filepath: str):
    """Save configuration to file"""
    import json
    from dataclasses import asdict

    config_dict = {
        'process': asdict(config.process),
        'missingness': asdict(config.missingness),
        'imputaion': asdict(config.imputation),
        'evaluation': asdict(config.evaluation),
        'visualization': asdict(config.visualization),
        'seed': config.seed
    }

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Configuration saved to: {filepath}")


# ======================================================================
# TEST
# ======================================================================

if __name__ =="__main__":
    config = get_config()
    config.summary()

    # save for reproductibility

    save_config(config, 'configs/default_config.json')