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
    data_imputed[target_column] = (
        data_imputed[target_column].
        ffill()
        .bfill
        )

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

def knn_imputation(
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: List[str],
        k: int = 5
) -> pd.DataFrame:
    """
    Impute using k-nearest neighbors

    For each missing value, find k most similar complete cases and average.

    Pros: Non-parametric, flexible
    Cons: Computationally expensivem sensitive to k
    
    Args:
        data: DataFrame with missing values.
        target_columns: Column to impute.
        predicture_columns: Columns to use for similarity.
        k:  Number of neighbors
    Returns:
        DataFrame with imputes values
    """
    data_imputed = data.copy()

    # Get masks
    observed_mask = ~data[target_column].isna()
    missing_mask = data[target_column].isna()

    # Check predictions
    if data[predictor_columns].isna().any().any():
        print("  Warning: Predictiors contain missing values, filling with mean")
        for col in predictor_columns:
            data_imputed[col].fillna(data_imputed[col].mean(), inplace=True)

    
    # Prepare data
    X_complete = data_imputed.loc[observed_mask, predictor_columns].values
    Y_complete = data_imputed.loc[observed_mask, target_column].values
    X_missing = data_imputed.loc[missing_mask, predictor_columns].values

    # Normalize features for distance calculation
    X_mean = X_complete.mean(axis=0)
    X_std = X_complete.std(axis=0) + 1e-8 #Avoid division by zero

    X_complete_norm = (X_complete - X_mean) / X_std
    X_missing_norm = (X_missing - X_mean) / X_std

    # For each missing point, fins k nearest neighbors
    predictions = []

    for x_miss in X_missing_norm:
        # Compute Euclidean distances
        distances = np.sqrt(np.sum((X_complete_norm - x_miss)**2, axis=1))

        # Find k nearest
        k_nearest_idx = np.argsort(distances)[:k]

        #Average their target values
        pred = np.mean(Y_complete[k_nearest_idx])
        predictions.append(pred)

    # Fill in predictions
    data_imputed.loc[missing_mask, target_column] = predictions

    n_imputed = missing_mask.sum()
    print(f"  KNN imutation: filled {n_imputed} values (k={k})")

    return data_imputed


# ================================================================
# UNIFIED INTERFACE
# ================================================================

def apply_imputation(
        data: pd.DataFrame,
        target_column: str,
        method: str,
        predictor_columns: Optional[List[str]] = None,
        k: int = 5
)-> pd.DataFrame:
    """
    Apply specified imputation method

    Args:
        data: DataFrame with missing values
        target_column: Colume to impute
        method: One of ['mean', 'forward_fill', 'linear_interpolation', 'regression', 'knn']
        predictor_columns: For regression/knn methods,
        k: For KNN method
    Returns:
        DataFrame with imputed values
    """

    if method == 'mean':
        return mean_imputation(data, target_column)
    elif method == 'forward_fill':
        return forward_fill(data, target_column)
    elif method == 'linear_interpolation':
        return linear_interpolation(data, target_column)
    elif method == 'regression':
        if predictor_columns is None:
            raise ValueError("predictor_columns required for regression imputation")
        return regression_imputation(data, target_column, predictor_columns)
    elif method == 'knn':
        if predictor_columns is None:
            raise ValueError("predictor_columns required for KNN imputation")
        return knn_imputation(data, target_column, predictor_columns, k)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def impute_all_methods(
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
)-> dict:
    """
    Apply all imputation methods and return results
    
    Args:
        data: DataFrame with missing values
        target_column: Column to impute
        predictor_colums: For model-based methods
        methods: List of methods to test (None = all)
    Returns:
        Dictionary mapping method name to imputed DataFrame
    """
    if methods is None:
        methods = ['mean', 'forward_fill', 'linear_interpolation']
        if predictor_columns is not None:
            methods.extend(['regression', 'knn'])
    results = {}

    for method in methods:
        print(f"\n Applying {method}...")
        try:
            results[method] = apply_imputation(
                data.copy(),
                target_column,
                method,
                predictor_columns
            )
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = None

    return results

# ===========================================================================
# TESTING
# ===========================================================================

if __name__ == "__main__":
    from config import get_config
    from evaluation import compute_bias, compute_rmse
    import pandas as pd
    
    print("="*60)
    print("IMPUTATION METHODS TEST")
    print("="*60)
    
    config = get_config()
    
    # Load data
    print("\n📂 Loading data...")
    data_true = pd.read_csv('data/raw/ground_truth.csv')
    data_mcar = pd.read_csv('data/processed/mcar_data.csv')
    
    target = config.missingness.target_sensor
    predictors = config.imputation.regression_predictors
    
    print(f"  Target: {target}")
    print(f"  Predictors: {predictors}")
    print(f"  Missing rate: {data_mcar[target].isna().mean():.1%}")
    
    # Test all methods
    print("\n" + "="*60)
    print("TESTING ALL IMPUTATION METHODS")
    print("="*60)
    
    results = impute_all_methods(data_mcar, target, predictors)
    
    # Evaluate each method
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Bias':>10} {'RMSE':>10}")
    print("-"*42)
    
    for method, data_imputed in results.items():
        if data_imputed is not None:
            bias = compute_bias(data_true[target], data_imputed[target])
            rmse = compute_rmse(data_true[target], data_imputed[target])
            print(f"{method:<20} {bias:>+10.4f} {rmse:>10.4f}")
            
            # Save
            output_path = f'data/processed/mcar_{method}_imputed.csv'
            data_imputed.to_csv(output_path, index=False)
    
    print("\n✅ All imputation methods tested!")