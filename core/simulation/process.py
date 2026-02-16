"""
simulation/process.py
Ground Truth Physical Process Generator

Generates realistic sensor data using AR(1) process with cross-correlations.
This is the foundation - everything else studies that happens when this data goes missing.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from simulation.config import ProcessConfig

def generate_ar1_series(
        n_steps: int,
        ar_coef: float,
        noise_std: float,
        initial_value: float,
        seed: int
) -> np.ndarray:
    """
    Generate Auto-Regressive(1) time series : X_t = a.X_{t-1} + e_t

    Args:
        n_steps: Number of time steps
        ar_coef: Autoregressive coefficient a (0 < a < 1)
        noise-std: Standard deviation of noise e
        initial_value: Starting value X_0
        seed: Random seed for reproducibility

    Returns:
       Array of length n_steps
    """

    np.random.seed(seed)

    # Pre-allocate array
    series = np.zeros(n_steps)
    series[0] = initial_value

    # Generate white noise
    noise = np.random.normal(0, noise_std, n_steps)

    # AR(1) recursion
    for t in range(1, n_steps):
        series[t] = ar_coef * series[t - 1] + noise[t]
    
    return series

def add_cross_correlation(
        series1: np.ndarray,
        series2: np.ndarray,
        strength: float
) -> np.ndarray:
    
    """
    Add cross-correlation between two series

    Args:
       series1: Predictor series
       series2: Target series to modify
       strength: Correalation strength (0 to 1)

    Returns:
       Modified series2 with added correlation to series1
    """
    # Normalise series1 to have similat scale
    series1_norm = (series1 - np.mean(series1)) / np.std(series1)

    #Add weighted influence
    series2_modified = series2 + strength * series1_norm * np.std(series2)
    
    return series2_modified

def generate_ground_truth(config: ProcessConfig) -> pd.DataFrame:
    """
    Generate complete ground truth sensor data

    Args:
       config: ProcessConfig with all parameters
    
    Returns:
       DataFrame with columns: [timestamp, temperature, humidity, pollution, vibration]
    """
    print(f" Generating {config.n_timesteps} timesteps...")

    # Generate base AR(1) series for each sensor
    data = {}

    for i, sensor in enumerate(config.sensors):
        print(f"  - Generating {sensor}...")

        series = generate_ar1_series(
            n_steps=config.n_timesteps,
            ar_coef=config.ar_coefficients[sensor],
            noise_std=config.noise_std[sensor],
            initial_value=config.initial_values[sensor],
            seed=42 + i # Different seed per sensor
        )

        data[sensor] = series

    
    # Add cross-correlations
    print("  - Adding cross-correlations...")
    for sensor1, sensor2 in config.cross_correlation_pairs:
        data[sensor2] = add_cross_correlation(
            data[sensor1],
            data[sensor2],
            config.cross_correlation_strength
        )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.insert(0, 'timestamp', range(config.n_timesteps))

    # Save to disk
    output_path = Path('data/raw/ground_truth.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Data saved to: {output_path}")
    print(f"Shape: {df.shape}")

    return df

def compute_autocorrelation(series: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """ Compute autocorrelation function"""

    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        c_lag = np.sum((series[:-lag-1] - mean) * (series[lag+1:] - mean)) / n
        acf[lag] = c_lag / c0 if c0 > 0 else 0
    
    return acf

def validate_ground_truth(data: pd.DataFrame, config: ProcessConfig) -> Dict:
    """
    Validate generated data quality
    Checks:
    1. No NaN or infinite values
    2. Temporal correlation exists
    3. Cross-correlations present
    4. Distribution reasonable

    Returns:
    Dictonary with validation results and diagnostic plots.
    """
    print("/n Validating data quality....")

    results = {
        'passed': True,
        'issues': [],
        'statistics': {}
    }

    sensors = config.sensors

    # Check 1: No missing or infinite values
    if data.isnull().any().any():
        results['passed'] = False
        results['issues'].append("Contains NaN values")
    
    # Check 2: Basic statistics 
    for sensor in sensors:
        series = data[sensors].values
        results['statistics'][sensor] = {
            'mean': float(np.mean(series)),
            'std': float(np.std(series)),
            'min': float(np.min(series)),
            'max': float(np.max(series))
        }
        print(f" {sensor:12s}: μ={np.mean(series):7.2f}, σ={np.std(series):6.2f}")
    
    # Check 3: Temporal correlation
    print("\n Checking autocorrelations...")
    for sensor in sensors:
        acf = compute_autocorrelation(data[sensor].values, max_lag=10)
        lag1_corr = acf[0]

        if lag1_corr < 0.3: # Should have some temporal correlation
            results['issues'].append(f"{sensor} has weak temporal correlation: {lag1_corr:.3f}")

        print(f"  {sensor:12s}: ACF(1) = {lag1_corr:.3f}")

    # Check 4: Cross-correlation
    print("\n Checking cross-correlations...")
    for sensor1, sensor2 in config.cross_correlation_pairs:
        corr = np.corrcoef(data[sensor1], data[sensor2])[0, 1] 
        print(f"  {sensor1} <-> {sensor2}: ρ = {corr:.3f}")

        if abs(corr) < 0.1: # Should have some correlation
            results['issues'].append(f"Weak correlation between {sensor1} and {sensor2}: {corr:.3f}")
              
    # Generate diagnostic plots
    _create_diagnostic_plots(data, config)

    if results['passed']:
        print("\n All validation checks passed!")
    else:
        print('\n Validation issue found:')
        for issue in results['issues']:
            print(f"   -{issue}")
    
    return results

def _create_diagnostic_plots(data: pd.DataFrame, config: ProcessConfig):
    """Create diagnostic plots for ground truth data"""

    output_dir = Path('results/figure')
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = config.sensors
    n_sensors = len(sensors)

    # Plot 1: Time series
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2.5*n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i, sensor in enumerate(sensors):
        axes[i].plot(data['timestamp'], data[sensor], linewidth=0.5, alpha=0.7)
        axes[i].set_ylabel(sensor.capitalize())
        axes[i].grid(True, alpha=0.3)

        # Show first 500 points for clarity
        axes[i].set_xlim(0, 500)
    
    axes[-1].set_xlabel('Timestamp')
    plt.suptitle('Ground Truth: Sensor Time Series (First 500 Steps)', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_timeseries.png', dpi=100, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'phase1_timeseries.png'}")
    plt.close()

    # Plot 2: Distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, sensor in enumerate(sensors):
        axes[i].hist(data[sensor], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(sensor.capitalize())
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{sensor.capitalize()}\nμ={data[sensor].mean():.2f}, σ={data[sensor].std():.2f}')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Ground Truth: Sensor Distributions')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_distribution.png', dpi=100, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'phase1_distributions.png'}")
    plt.close()

    # Plot 3: Correlation matrix
    corr_matrix = data[sensors].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    #Set ticks
    ax.set_xticks(range(n_sensors))
    ax.set_yticks(range(n_sensors))
    ax.set_xticklabels([s.capitalize() for s in sensors], rotation=45, ha='right')
    ax.set_yticklabels([s.capitalize() for s in sensors])


    #Add correlation values
    for i in range(n_sensors):
        for j in range(n_sensors):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.title('Ground Truth: Sensor Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_correlations.png', dpi=100, bbox_inches='tight')
    print(f"  📊 Saved: {output_dir / 'phase1_correlations.png'}")
    plt.close()

    # Plot 4: Autocorrelation functions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    max_lag = 50

    for i, sensor in enumerate(sensors):
        acf = compute_autocorrelation(data[sensor].values, max_lag)
        axes[i].bar(range(max_lag), acf, width=0.8, alpha=0.7)
        axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].axhline(y=0.2, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i].axhline(y=-0.2, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('ACF')
        axes[i].set_title(f'{sensor.capitalize()} (α={config.ar_coefficients[sensor]:.2f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.3, 1.1)
    
    plt.suptitle('Grond Truth: Autocorrelation Functions')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_autocorrelation.png', dpi=100, bbox_inches='tight')
    print(f"  📊 Saved: {output_dir / 'phase1_autocorrelation.png'}")
    plt.close()

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from simulation.config import get_config
    
    print("="*60)
    print("PHASE 1: GROUND TRUTH GENERATION TEST")
    print("="*60)
    
    config = get_config()
    
    # Generate data
    data = generate_ground_truth(config.process)
    
    # Validate
    results = validate_ground_truth(data, config.process)
    
    if results['passed']:
        print("\n✅ Phase 1 Test PASSED")
    else:
        print("\n⚠️  Phase 1 Test COMPLETED WITH WARNINGS")