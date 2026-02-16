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
        'mean_complete': mean_complete,
        'mean_bias' : mean_bias,
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
            'o-', markersize=2, linewidth=0.5, alpha=0.3, label='Complete', color='grey')
    

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
    indicator = observed_mask.astype(int).values

    # Reshape into blocks for visualization (e.g., 100 blocks of 100 timestamps)
    n_blocks = 100
    block_size = len(indicator) // n_blocks
    blocked = indicator[:n_blocks * block_size].reshape(n_blocks, block_size)

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

def compare_mechanisms(
        data_complete: pd.DataFrame,
        target_column: str,
        configs: Dict
) -> pd.DataFrame:
    """
    Compare all three mechanisms side-by-side
    Args:
       data_complete: Original complete data.
       target_column: Column to inject missingness into.
       configs: Dictonary with keys 'mcar', 'mar', 'mnar' containing parameters.
    Returns:
       DataFrame with comparison statistics 
    """

    results = []

    # MCAR
    if 'mcar' in configs:
        data_mcar = inject_mcar(
            data_complete,
            target_column,
            configs['mcar']['probability'],
            seed=42
        )
        analysis = analyze_missingness(data_complete, data_mcar, target_column)
        results.append({
            'mechanism': 'MCAR',
            'missing_rate': analysis['missing_rate'],
            'mean_bias': analysis['mean_bias'],
            'std_complete': analysis['std_complete'],
            'std_observed': analysis['std_observed']
        })

    # MAR
    if 'mar' in configs:
        data_mar = inject_mar(
            data_complete,
            target_column,
            configs['mar']['predictor'],
            configs['mar']['alpha'],
            configs['mar']['intercept'],
            seed=42
        )
        analysis = analyze_missingness(data_complete, data_mar, target_column)
        results.append({
            'mechanism': 'MAR',
            'missing_rate': analysis['missing_rate'],
            'mean_bias': analysis['mean_bias'],
            'std_complete': analysis['std_complete'],
            'std_observed': analysis['std_observed']
        })

    # MNAR
    if 'mnar' in configs:
        data_mnar = inject_mnar(
            data_complete,
            target_column,
            configs['mnar']['beta'],
            configs['mnar']['intercept'],
            seed=42
        )
        analysis = analyze_missingness(data_complete, data_mnar, target_column)
        results.append({
            'mechanism': 'MNAR',
            'missing_rate': analysis['missing_rate'],
            'mean_bias': analysis['mean_bias'],
            'std_complete': analysis['std_complete'],
            'std_observed': analysis['std_observed']
        })

    return pd.DataFrame(results)

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from .config import get_config

    print("="*60)
    print("MISSINGNESS MECHANISMS TEST")
    print("="*60)

    # Load ground truth
    data = pd.read_csv('data/raw/ground_truth.csv')
    config = get_config()
    target = config.missingness.target_sensor

    print(f"\n Testing on sensor: {target}")
    print(f"  Complete data shape: {data.shape}")

    # Test MCAR
    print("\n" + "-"*60)
    print("Testing MCAR...")
    print("-"*60)
    data_mcar = inject_mcar(data, target, 0.3, seed=42)
    analysis_mcar = analyze_missingness(data, data_mcar, target)
    print(f"  Mean bias: {analysis_mcar['mean_bias']:+.4f}")

    # Save
    data_mcar.to_csv('data/processed/mcar_data.csv', index=False)
    print(" Saved to: data/processed/mcar_data.csv")

    #Visualize
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_missingness(
        data, data_mcar, target, 'MCAR',
        output_path=str(output_dir / 'phase2_mcar_analysis.png')
    )

    # Test MAR
    print("/n" + "-"*60)
    print("Testing MAR...")
    print("-"*60)
    data_mar = inject_mar(
        data, target,
        config.missingness.mar_predictor,
        config.missingness.mar_alpha,
        config.missingness.mar_intercept,
        seed=42
    )

    analysis_mar = analyze_missingness(data, data_mar, target)
    print(f" Mean bias: {analysis_mar['mean_bias']:+.4f}")

    data_mar.to_csv('data/processed/mar_data.csv', index= False)
    print("  Saved to: data/processed/mar_data.csv")

    visualize_missingness(
        data, data_mar, target, 'MAR',
        output_path=str(output_dir / 'phase2_mar_analysis.png')
    )

    # Test MNAR
    print("\n" + "-"*60)
    print("Testing MNAR...")
    print("-"+60)
    data_mnar = inject_mnar(
        data, target,
        config.missingness.mnar_beta,
        config.missingness.mnar_intercept,
        seed=42
    )
    analysis_mnar = analyze_missingness(data, data_mnar, target)
    print(f"  Mean bias: {analysis_mnar['mean_bias']:+.4f}")

    data_mnar.to_csv('data/processed/mnar_data.csv', index=False)
    print("  Saved to: data/processed/mnar_data.csv")

    visualize_missingness(
        data, data_mnar, target, 'MNAR',
        output_path=str(output_dir / 'phase2_mnar_analysis.png')
    )

    # Comparison
    print("/n" + "="*60)
    print("MECHANISM COMPARISON")
    print("="*60)

    comparison_configs= {
        'mcar': {'probability':0.3},
        'mar': {
            'predictor': config.missingness.mar_predictor,
            'alpha': config.missingness.mar_alpha,
            'intercept': config.missingness.mar_intercept
        },
        'mnar':{
            'beta': config.missingness.mnar_beta,
            'intercept': config.missingness.mnar_intercept
        }
    }

    comparison = compare_mechanisms(data, target, comparison_configs)
    print("\n", comparison.to_string(index=False))

    comparison.to_csv('results/metrics/phase2_mechanism_comparison.csv', index=False)
    print("\n  Comaparison saved to: results/metrics/phase2_mechanism_comparison.csv")




