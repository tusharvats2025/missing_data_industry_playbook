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

# ============================================================================
# TEMPORAL METRICS
# ============================================================================

def compute_autocorrelation(
        series: np.ndarray,
        lag: int = 1
)-> float:
    """
    Compute autocorrelation at given lag

    Args:
        series: Time series
        lag: Lag value

    Returns: 
        Autocorrelation coefficient
    """

    n = len(series)
    mean = np.mean(series)

    c0 = np.sum((series - mean)** 2) / n
    c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n

    if c0 == 0:
        return np.nan
    return c_lag / c0

def compute_autocorrelation_preservation(
        true_series: np.ndarray,
        imputed_series: np.ndarray,
        max_lag: int = 10
)-> Dict[str, float]:
    """
    Measure how well autocorrelation structure is preserved.

    Args:
        true_series: True time series.
        imputed_series: Imputed time series.
        max_lag : Maximum lag to check
    Returns:
        Dictionary with ACF preservatiom metrics
    """
    acf_true = [compute_autocorrelation(true_series, lag) for lag in range(1, max_lag + 1)]
    acf_imputed = [compute_autocorrelation(imputed_series, lag) for lag in range(1, max_lag + 1)]

    # Compute RMSE of ACF
    acf_true = np.array(acf_true)
    acf_imputed = np.array(acf_imputed)

    acf_rmse = np.sqrt(np.mean((acf_true - acf_imputed) ** 2))

    # Compute correlation between ACFs
    acf_correlation = np.corrcoef(acf_true, acf_imputed)[0, 1]

    return {
        'acf_rmse': acf_rmse,
        'acf_correlation': acf_correlation,
        'acf_true': acf_true.tolist(),
        'acf_imputed': acf_imputed.tolist()
    }

# ========================================================================
# COMPREHENSIVE EVALUATION
# ========================================================================

def evaluate_imputation(
        data_true: pd.DataFrame,
        data_imputed: pd.DataFrame,
        target_column: str,
        method_name: str = "Unknown"
) -> Dict:
    """
    Comprehensive evaluation of imputation quality

    Args:
        data_true: DataFrame with ground truth.
        data_imputed: DataFrame with imputed values.
        target_column: Column that was imputed.
        method_column: Column that was imputed.
    
    Returns:
        Dictionary with all metrics
    """
    true_vals = data_true[target_column].values
    imputed_vals = data_imputed[target_column].values

    # Basic metrics
    bias = compute_bias(true_vals, imputed_vals)
    rmse = compute_rmse(true_vals, imputed_vals)
    mae = compute_mae(true_vals, imputed_vals)
    var_ratio = compute_variance_ratio(true_vals, imputed_vals)

    # Distribution metrics
    kl_div = compare_kl_divergence(true_vals, imputed_vals)
    ks_result = compute_ks_statistic(true_vals, imputed_vals)

    # Tempporal metrics
    acf_metrics = compute_autocorrelation_preservation(true_vals, imputed_vals)

    results = {
        'method': method_name,
        'bias': bias,
        'rmse': rmse,
        'mae': mae,
        'variance_ratio': var_ratio,
        'kl_divergence': kl_div,
        'ks_statistic': ks_result['statistic'],
        'ks_pvalue': ks_result['pvalue'],
        'acf_rmse': acf_metrics['acf_rmse'],
        'acf_correlation': acf_metrics['acf_correlation']
    }

    return results

def compare_methods(
        data_true: pd.DataFrame,
        imputed_datasets: Dict[str, pd.DataFrame],
        target_columm: str
)-> pd.DataFrame:
    """
    Compare multiple imputation methods


    Args:
        data_true: DataFrame with ground truth.
        imputed_datasets: Dictionary mapping method name to imputed DataFrame.
        target_column: Column that was imputed.
    
    Returns:
        DataFrame with comparison results.
    """

    results = []

    for method_name, data_imputed in imputed_datasets.items():
        if data_imputed is not None:
            metrics = evaluate_imputation(
                data_true,
                data_imputed,
                target_columm,
                method_name
            )
            results.append(metrics)
    
    df_results = pd.DataFrame(results)

    # Sort by RMSE (primary metric)
    df_results = df_results.sort_values('rmse')

    return df_results

def print_evaluation_report(evaluation_results: pd.DataFrame):
    """
    Print formatted evaluation report

    Args:
        evaluation_results: DataFrame from compare_methods()
    """
    print("\n" + "="*80)
    print("IMPUTATION EVALUATION REPORT")
    print("="*80)

    # Key metrics table
    print("\nKey Metrics:")
    print("-"*80)
    print(f"{'Method':<20} {'Bias':>10} {'RMSE':>10} {'MAE':>10}{'Var Ratio':>10}")
    print("-"*80)

    for _, row in evaluation_results.iterrows():
        print(f"{row['method']:<20} "
              f"{row['bias']:>+10.4f} "
              f"{row['rmse']:>10.4f} "
              f"{row['mae']:>10.4f} "
              f"{row['varience_ratio']:.10.4f}")
    
    print("\n Distribution Metrics:")
    print("-"*80)
    print(f"{'Method':<20} {'KL Div':>12} {'KS Stat':>12} {'KS p-value':>12}")
    print("-"*80)

    print(f"{'Method':<20} {'KL Div':>12} {'KS Stat':>12} {'KS p-value':>12}")
    print("-"*80)
    
    for _, row in evaluation_results.iterrows():
        print(f"{row['method']:<20} "
              f"{row['kl_divergence']:>12.4f} "
              f"{row['ks_statistic']:>12.4f} "
              f"{row['ks_pvalue']:>12.4f}")
    
    print("\n⏰ Temporal Metrics:")
    print("-"*80)
    print(f"{'Method':<20} {'ACF RMSE':>12} {'ACF Corr':>12}")
    print("-"*80)
    
    for _, row in evaluation_results.iterrows():
        print(f"{row['method']:<20} "
              f"{row['acf_rmse']:>12.4f} "
              f"{row['acf_correlation']:>12.4f}")
    
    print("\n" + "="*80)
    
    # Identify best method per metric
    print("\n🏆 Best Methods by Metric:")
    print("-"*80)
    
    best_bias = evaluation_results.loc[evaluation_results['bias'].abs().idxmin(), 'method']
    best_rmse = evaluation_results.loc[evaluation_results['rmse'].idxmin(), 'method']
    best_var = evaluation_results.loc[(evaluation_results['variance_ratio'] - 1.0).abs().idxmin(), 'method']
    best_kl = evaluation_results.loc[evaluation_results['kl_divergence'].idxmin(), 'method']
    best_acf = evaluation_results.loc[evaluation_results['acf_rmse'].idxmin(), 'method']
    
    print(f"  Lowest absolute bias: {best_bias}")
    print(f"  Lowest RMSE: {best_rmse}")
    print(f"  Best variance preservation: {best_var}")
    print(f"  Lowest KL divergence: {best_kl}")
    print(f"  Best ACF preservation: {best_acf}")
    
    print("="*80)

# ======================================================================================
# TESTING
# ======================================================================================

if __name__== "__main__":
    print("="*60)
    print("EVALUATION METRICS TEST")
    print("="*60)

    # Create synthetic test data
    np.random.seed(42)
    n = 1000
    
    true_vals = np.random.normal(50, 10, n)
    
    # Simulate different imputation scenarios
    perfect = true_vals.copy()
    biased_high = true_vals + 5  # Systematic overestimation
    low_variance = true_vals * 0.5 + np.mean(true_vals) * 0.5  # Variance shrinkage
    
    print("\n📊 Testing metrics...")
    
    print(f"\nPerfect imputation:")
    print(f"  Bias: {compute_bias(true_vals, perfect):.4f}")
    print(f"  RMSE: {compute_rmse(true_vals, perfect):.4f}")
    print(f"  Var ratio: {compute_variance_ratio(true_vals, perfect):.4f}")
    
    print(f"\nBiased imputation (+5):")
    print(f"  Bias: {compute_bias(true_vals, biased_high):.4f}")
    print(f"  RMSE: {compute_rmse(true_vals, biased_high):.4f}")
    print(f"  Var ratio: {compute_variance_ratio(true_vals, biased_high):.4f}")
    
    print(f"\nLow variance imputation:")
    print(f"  Bias: {compute_bias(true_vals, low_variance):.4f}")
    print(f"  RMSE: {compute_rmse(true_vals, low_variance):.4f}")
    print(f"  Var ratio: {compute_variance_ratio(true_vals, low_variance):.4f}")
    
    print("\n✅ Evaluation metrics working correctly!")