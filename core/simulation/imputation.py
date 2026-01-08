"""
core.simulation.imputation
Missing Data Handling Techniques

Implements various imputation strategies:
- Mean imputation (base line)
- Forward fill (time series baseline)
- Linear interpolation (time series)
- Regression imputation (model-based)
- KNN imputation (model-based)

"""
import numpy as np
import pandas as pd
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASELINE METHODS
# ============================================================================

def mean_imputation(
        data: pd.DataFrame,
        target_column: str
) -> pd.DataFrame:
    """
    Replace missing values with the mean of observed values.

    Pros: Simple, unbiased for MCAR.
    Cons: Reduces varience, breaks correlations.

    Args:
        data: DataFrame with missing values
        target_column: Column to impute

    Returns:
        DataFrame with imputed values
    """
    data_imputed = data.copy()

    # Compute mean from observed values only
    observed_mean = data[target_column].mean()

    # Fill missing values
    data_imputed[target_column].fillna(observed_mean, inplace=True)

    n_imputed = data[target_column].isna().sum()
    print(f" Mean imputation: filled {n_imputed} values with {observed_mean:.2f}")

    return data_imputed

def forward_fill(
        data: pd.DataFrame,
        target_column: str
) -> pd.DataFrame:
    """
    Replace missing values with the last observed value (LOCF)

    Pros: Preserves temporal continuity
    Cons: Can propagate state values, biased if trend exists
    
    Args:
        data: DataFrame with missing values (must be time exists)
        target_column: Column to impute
    Returns:
        DataFrame with imputed values
    """
    data_imputed = data.copy()

    # Forward Fill
    data_imputed[target_column].fillna(method='ffill', inplace=True)

    # IF first values are missing, fill with first observed value
    if data_imputed[target_column].isna().any():
        first_valid = data_imputed[target_column].first_valid_index()
        first_value = data_imputed.iloc[first_valid, target_column]
        data_imputed[target_column].fillna(first_value, inplace=True)

    n_imputed = data[target_column].isna().sum()
    print(f"  Forward fill: filled {n_imputed} values using LOCF")

    return data_imputed

def linear_interpolation(
        data: pd.DataFrame,
        target_column: str
) -> pd.DataFrame:
    """
    Replace missing values with linear interpolation between observed points

    Pros: Smooth, reasonable for time Series
    Cons: Assumes linear trends, reduces variance

    Args:
        data: DataFrame with missing values(must be time-ordered)
        target_column: Column to impute
    Returns:
        DataFrame with imputed values
    """
    data_imputed = data.copy()

    # Interpolate
    data_imputed[target_column] = data_imputed[target_column].interpolate(
        method='linear',
        limit_direction='both'
    )

    n_imputed = data[target_column].isna().sum()
    print(f"  Linear intepolation: filled {n_imputed} values")

    return data_imputed



