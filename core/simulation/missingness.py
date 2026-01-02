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

def visualize_missingness(
        data_complete: pd.DataFrame,
        data_missing: pd.DataFrame,
        target_column: str,
        mechanism: str,
        output_path: Optional[str] = None
):
    """
    Create diagnostic plots for missingness pattern

    Plots:
    1. Time series with missing gaps highlighted
    2. Distribution comparison (complete vs observed)
    3. Missingness pattern over time

    Args:
       data_complete: Original complete data
       data_missing: Data with missingness
       target_column: Column with missing values
       mechanism: "MCAR", "MAR", or "MNAR"
       output_path: Where to save figure(optional)
    
    """
    observed_mask = ~data_missing[target_column].isna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Time Series with gaps
    ax = axes[0]
    timestamps = data_complete['timestamp'].values

    # Plot compelete data in light color
    ax.plot(timestamps[:500], data_complete[target_column].values[:500],
            'o-', makersize=2, linewidth=0.5, alpha=0.3, label='Complete', color='grey')
    

    # Plot observed data in bold
    observed_times = timestamps[observed_mask][:500]
    observed_vals = data_missing[target_column].dropna().values[:min(500, observed_mask.sum())]
    ax.plot(observed_times, observed_vals,
            'o-', markersize=3, linewidth=1, label='Observed', color='blue')
    
    # Highlight missing regions
    missing_times = timestamps[~observed_mask][:500]
    if len(missing_times) > 0:
        for t in missing_times:
            ax.axvline(t, color='red', alpha=0.1, linewidth=2)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel(target_column.capitalize())
    ax.set_title(f'{mechanism}: Time Series with Missing data (First 500 steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution comparison
    ax = axes[1]
    bins = np.linspace(
        min(data_complete[target_column].min(), data_missing[target_column].min()),
        max(data_complete[target_column].max(), data_missing[target_column].max()),
        50
    )

    ax.hist(data_complete[target_column], bins=bins, alpha=0.5,
            label='Complete', color='gray', edgecolor='black')
    ax.hist(data_missing[target_column].dropna(), bins=bins, alpha=0.7,
            label='Observed', color='blue', edgecolor='black')
    
    # Add mean lines
    ax.axvline(data_complete[target_column].mean(), color='gray',
               linestyle='--', linewidth=2, label='Complete Mean')
    ax.axvline(data_missing[target_column].mean(), color='blue',
               linestyle='--', linewidth=2, label='Observed Mean')
    
    ax.set_xlabel(target_column.capitalize())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{mechanism}: Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot 3: missingness heatmap over time
    ax = axes[2]

    # Create binary indicator (1=observed, 0=missing)
    inidcator = observed_mask.astype(int)

    # Reshape into blocks for visualization (e.g., 100 blocks of 100 timestamps)
    n_blocks = 100
    block_size = len(inidcator) // n_blocks
    blocked = inidcator[:n_blocks * block_size].reshape(n_blocks, block_size)

    # Plot as heatmap
    im = ax.imshow(blocked.T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Time Block')
    ax.set_ylabel('Within-Block Timestep')
    ax.set_title(f'{mechanism}: Missingness Pattern Heatmap (Green=Observed, Red=Missing)')
    plt.colorbar(im , ax=ax, label='Observed (1) vs Missing (0)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f" Saved: {output_path}")
    else:
        plt.show()
    plt.close()


