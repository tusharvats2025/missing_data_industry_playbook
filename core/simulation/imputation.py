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

# ========================================================================
# MODEL-BASED METHODS
# ========================================================================

def regression_imputation(
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: List[str]
)-> pd.DataFrame:
    """
    Impute using Linear regression on other variables

    Fits: y = β₀ + β₁·x₁ + β₂·x₂ + ... + ε

    Pros: Uses relationships between variables.
    Cons: Requires complete predictors, overfits if misspecified.

    Args:
        data: DataFrame with missing values.
        target_column: column to impute.
        predictor_columns: Columns to use as predictors.
    
    Returns:
        DataFrame with imputed values
    """
    data_imputed = data.copy()
    
    # Get observed mask
    observed_mask = ~data[target_column].isna()
    missing_mask = data[target_column].isna()

    # Check is predictiors have missing values
    if data[predictor_columns].isna().any().any():
        print("  Warning: Predictions contain missing values, fillnf with mean")
        for col in predictor_columns:
            data_imputed[col].fillna(data_imputed[col].mean(), inplace=True)

    # Prepare data
    X_train = data_imputed.loc[observed_mask, predictor_columns].values
    y_train = data_imputed.loc[observed_mask, target_column].values
    X_missing = data_imputed.loc[missing_mask, predictor_columns].values

    # Fit linear regression using normal equations: β = (X'X)⁻¹X'y
    # Add intercept term
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    X_missing_aug = np.column_stack([np.ones(len(X_missing)), X_missing]) 


    # Solve:  β = (X'X)⁻¹X'y 
    try:
        beta = np.linalg.solve(X_train_aug.T @ X_train_aug, X_train_aug.T @ y_train)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        beta = np.linalg.pinv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

    # Predict missing values
    y_pred = X_missing_aug @ beta

    # Fill in predictions
    data_imputed.loc[missing_mask, target_column] = y_pred

    n_imputed = missing_mask.sum()
    r2 = 1 - np.sum((y_train - X_train_aug @ beta)**2) / np.sum((y_train - y_train.mean())**2)
    print(f"  Regression imputation: filled {n_imputed} values (R²={r2:.3f})")
    
    return data_imputed