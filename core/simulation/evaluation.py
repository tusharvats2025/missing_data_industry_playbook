"""
core.simulation.evaluation.py
Evaluation Metrics for Imputation Quality

Measures statistical correctness of imputation methods:
- Bias (systematic error)
- RMSE (overall error)
- Variance ratio (variance preservation)
- Distribution metrics (KL divergence)
- Temporal metrics (autocorrelation preservation)

All metrics are transparent and interpretable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats

# =====================================================
# BASIC METRICS
# =====================================================

def compute_bias(
        true_values: np.ndarray,
        imputed_values: np.ndarray
)-> float:
    """
    Compute bias: E[X_imputed] -E[X_true]

    Measures systematic over/under-estimation.

    Args:
        true_values: Ground truth values
        imputed_values: Imputed values
    Returns:
        Bias (positive = overestimation, negative = underestimation)
    """
    bias = np.mean(imputed_values) - np.mean(true_values)
    return bias

def compute_rmse(
        true_values: np.ndarray,
        imputed_values: np.ndarray
)-> float:
    """
    Compute Root Mean Squared Error

    Measures overall prediction accuracy.

    Args:
        true_values: Ground truth values
        imputed_values: Imputed values
    
    Returns:
        RMSE (always positive, lower is better)
    """

    mse = np.mean((true_values - imputed_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def compute_variance_ratio(
        true_values: np.ndarray,
        imputed_values: np.ndarray
)-> float:
    """
    Compute ration of variences: Var(imputed) / Var(true)

    Measures varience preservation.
    Ratio < 1 indicates varience shrinkage (comman problem).

    Args:
        true_values: Ground truth values
        imputed_values: Imputed values
    Returns:
        Varience ratio(ideal = 1.0)
    """
    var_true = np.var(true_values)
    var_imputed = np.var(imputed_values)

    if var_true == 0:
        return np.nan
    return var_imputed / var_true

def compute_mae(
        true_values: np.ndarray,
        imputed_values: np.ndarray
)-> float:
    """
    Compute Mean Absoulte Error

    More robust to outliners than RMSE.

    Args:
        true_values: Ground truth values
        imputed_values: Imputed Values
    
    Return:
        MAE (always positive, lower is better)
    """
    mae = np.mean(np.abs(true_values - imputed_values))
    return mae

# ========================================================================
# DISTRIBUTION METRICS
# ========================================================================

def compare_kl_divergence(
        true_values: np.ndarray,
        imputed_values: np.ndarray,
        n_bins: int = 50
)-> float:
    """
    Compute KL divergence between distributions

    Measures how much the imputed distribution differs from true distribution.
    KL(P||Q) = sum P(x) * log(P(x)/Q(x))

    Args:
        true_values: Ground truth values
        imputed_values: Imputed values
        n_bins: Number of bins for histogram
    Returns:
        KL divergence (always >= 0, lower is better)
    """

    # Create common bins
    combined = np.concatenate([true_values, imputed_values])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

    # Compute histograms
    hist_true, _ = np.histogram(true_values, bins=bins)
    hist_imputed, _ = np.histogram(imputed_values, bins=bins)

    # Normalise to probabilities
    p = hist_true / (hist_true.sum() + 1e-10)
    q = hist_imputed / (hist_imputed.sum() + 1e-10)

    # Add small constant to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return kl

def compute_ks_statistic(
        true_values: np.ndarray,
        imputed_values: np.ndarray
)-> Dict[str, float]:
    """
    Compute Kolomogrov-Smirnov test statistics.

    Measures maximum difference between comulative distributions.

    Args:
        true_values: Ground truth values
        imputed_values: Imputed values
    Returns:
        Dictionary with statistic and p-value
    """
    statistic, pvalue = stats.ks_2samp(true_values, imputed_values)

    return {
        'statistic': statistic,
        'pvalue': pvalue
    }

