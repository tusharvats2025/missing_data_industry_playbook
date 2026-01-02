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

