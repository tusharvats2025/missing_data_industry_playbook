"""
Docstring for core.simulation.missingness

Missing Data Mechanism Implementation

Implementats three fundamental missingness mechanishms:
-MCAR (Missing Completely At Random)
-MAR (Missing At Random)
-MNAR (Missing Not At Random)

Each mechanism is isolated and paramterized for controlled experiments.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.special import expit # Logstic sigmoid function

# ================================================================================
# MCAR: Missing Completely At Random
# ================================================================================

def inject_mcar(
        data: pd.DataFrame,
        target_column: str,
        probability: float,
        seed: int = 42
) -> pd.DataFrame:
    """
    Docstring for inject_mcar
    Inject MCAR missingness: P(R=0) = p

    Missingness is independent of all variables (observed and unobserved).
    This is the simplest mechanism - just flip a biased coin.

    Args:
        data: DataFrame with complete data
        target_column: Column to inject missingness into
        probability: Probability if being missing (0 to 1)
        seed: int
    Return: 
        DataFrame with missing values injected
    """

    np.random.seed(seed)

    n = len(data)

    # Bernaulli indicator: 1 = observed, 0 = missing
    R = np.random.binomial(1, 1 - probability, size=n)

    # Apply missingness
    data_missing = data.copy()
    data_missing.loc[R == 0, target_column] = np.nan

    n_missing = (R == 0).sum()
    actual_rate = n_missing / n

    print(f"  MCAR injected: {n_missing}/{n} missing ({actual_rate:.1%})")

    return data_missing

# =========================================================================
# MAR: Missing at random
# =========================================================================

def inject_mar(
        data: pd.DataFrame,
        target_column: str,
        predictor_column: str,
        alpha: float,
        intercept: float,
        seed: int = 42
) -> pd.DataFrame:
    """
    Docstring for inject_mar
    Inject MAR missingness: P(R=0 | X) = σ(α·X + β)

    Missingness depends on an OBSERVED variable (predictor_column).
    Uses logistic regression to create realistic dependency.

    Args:
        data: DataFrame with complete Data.
        target_column: Column to inject missingness into
        predictor_column: Observed column that affects missingness
        alpha: Coefficient for predictor (slope)
        intercept: Intercept term (bias)
        seed: Random seed
    Returns:
       DatFrame with MAR missingness
    """
    np.random.seed(seed)

    n = len(data)

    # Normalise predictor to [0, 1] range for stability
    X = data[predictor_column].values
    X_norm = (X - X.min()) / (X.max() - X.min())

    # Logistic probability of being missing
    logit = alpha * X_norm + intercept
    prob_missing = expit(logit) # σ(z) = 1 / (1 + exp(-z))

    # Sample missingness indicator
    R = np.random.binomial(1, 1 - prob_missing)

    # Apply missingness
    data_missing = data.copy()
    data_missing.loc[R == 0, target_column] = np.nan

    n_missing = (R == 0).sum()
    actual_rate = n_missing / n

    print(f"  MAR injected: {n_missing}/{n} missing ({actual_rate:.1%})")
    print(f"  Depends on: {predictor_column}")

    return data_missing

# =======================================================================
# MNAR: Missing Not At Random
# =======================================================================

def inject_mnar(
        data: pd.DataFrame,
        target_column: str,
        beta: float,
        intercept: float,
        seed: int = 42
) -> pd.DataFrame:
    """
    Docstring for inject_mnar
    Inject MNAR missingness: P(R=0 | Y) = σ(β·Y + γ)

    Missingness depends on the UNOBSERVED value iteself (target_column).
    This is the hardest case - the missing values are not missing at random.

    Args:
        data: Dataframe with complete data.
        target_column: Column to inject missingness into.
        beta: Coefficient for target itself
        intercept: Intercept term
        seed: Random seed
    Return:
        DataFrame with MNAR missingness
    """

    np.random.seed(seed)

    n = len(data)

    #Normalise target to [0,1] range
    Y = data[target_column].values
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())

    # Logistic probability of being missing
    logit = beta * Y_norm + intercept
    prob_missing = expit(logit)

    # Sample missing indicator
    R = np.random.binomial(1, 1 - prob_missing)

    # Apply missingness
    data_missing = data.copy()
    data_missing.loc[R == 0, target_column] = np.nan

    n_missing = (R == 0).sum()
    actual_rate = n_missing/n

    print(f"  MNAR injected: {n_missing}/{n} missing ({actual_rate:.1%})")
    print(f"  Depends on: {target_column} itself (UNOBSERVED when missing)")


    return data_missing

# ===========================================================================
# ANAYLYSIS FUNCTIONS
# ===========================================================================

def analyze_missingness(
        data_complete: pd.DataFrame,
        data_missing: pd.DataFrame,
        target_column: str
) -> Dict:
    """
    Docstring for analyze_missingness
    Analyze missingness pattern

    Compute basic statistics about the missing data:
    - Missing rate
    - Mean of observed vs complete
    - Distribution comparison

    Args:
      data_complete: Original complete data
      data_missing: Data with missingness
      target_column: Column with missing values

    Returns:
      Dictionary with analysis results
    
    """

    complete_values = data_complete[target_column].values
    observed_mask = ~data_missing[target_column].isna()
    observed_values = data_missing[target_column].dropna().values

    n_total = len(data_complete)
    n_missing = (~observed_mask).sum()
    missing_rate = n_missing / n_total

    # Compare means 
    mean_complete = np.mean(complete_values)
    mean_observed = np.mean(observed_values)
    mean_bias = mean_observed - mean_complete

    # Compare standard deviations
    std_complete = np.std(complete_values)
    std_observed = np.std(observed_values)

    results = {
        'missing_rate': missing_rate,
        'n_missing': n_missing,
        'n_observed': n_total - n_missing,
        'means_complete': mean_complete,
        'mean_observed': mean_observed,
        'std_complete': std_complete,
        'std_observed': std_observed,
        'observed_mask': observed_mask
    }

    return results
