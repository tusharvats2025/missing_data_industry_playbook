"""
main.py
core.main

Orchestrator for phase-by-phase execution

Usage:
     python main.py ---phase 0 # setup
     python main.py ---phase 1 # Ground truth generation
     python main.py ---phase 2 # MCAR mechanism
     ... etc
"""
import argparse
import sys
from pathlib import Path

# add simulation to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.config import get_config, save_config

STUDY_NAME = "CONTROLLED MISSING DATA MECHANISM STUDY"
STUDY_VERSION = "v1.0 (Locked Reference Experiment)"

def phase_1_ground_truth():
    """Phase 1: Generate ground truth data."""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 1: GROUND TRUTH GENERATION")
    print("="*60 + "\n")

    try:
        from simulation.process import generate_ground_truth, validate_ground_truth

        config = get_config()

        print(" Generating sensor data...")
        data = generate_ground_truth(config.process)

        print(" Validating data quality...")
        validation_results = validate_ground_truth(data, config.process)

        if validation_results['passed']:
            print("\n Step 1 Complete: Ground Truth Generated and Validated")
            print(f" Data saved to : data/raw/ground_truth.csv")
            print(" Next: Experiment Step 2: (MCAR Missingness)")
            return True
        else:
            print("\n Validation failed!")
            print("Issues:", validation_results['issues'])
            return False
    except ImportError:
        print(" process.py not implemented yet")
        print(" Next: Implement simulation/process.py")
        return False
    
def phase_2_mcar():
    """Phase 2: MCAR mechanism"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 2: MCAR (Missing Completely At Random)")
    print("="*60 + "\n")

    try:
        from simulation.missingness import inject_mcar, analyze_missingness, visualize_missingness
        import pandas as pd
        from pathlib import Path

        config = get_config()

        print(" Loading ground truth...")
        data = pd.read_csv('data/raw/ground_truth.csv')

        print("\n Injecting MCAR missingness...")
        data_mcar = inject_mcar(
            data.copy(),
            target_column=config.missingness.target_sensor,
            probability=config.missingness.mcar_probability,
            seed=config.seed
        )

        print("\n |Analyzing missingness pattern...|")
        analysis = analyze_missingness(data, data_mcar, config.missingness.target_sensor)

        print(f"  |-Missing rate: {analysis['missing_rate']:.1%}|")
        print(f"  |-Mean(complete): {analysis['mean_complete']:.2f}|")
        print(f"  |-Mean(observed): {analysis['mean_observed']:.2f}|")
        print(f"  |-Mean bias: {analysis['mean_bias']:+.4f}")

        # Save data
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        data_mcar.to_csv(output_dir / 'mcar_data.csv', index=False)
        
        # Create visualizations
        print("\n |Creating diagnostic plots...")
        fig_dir = Path('results/figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        visualize_missingness(
            data, data_mcar, config.missingness.target_sensor, 'MCAR',
            output_path=str(fig_dir / 'phase2_mcar_analysis.png')
        )

        print("\n Step 2 Complete: MCAR Missingness Injected and Analyzed")
        print(f"Data Saved to: {output_dir / 'mcar_data.csv'}")
        print(f"Missing rate: {analysis['missing_rate']:.1%}")
        print(f"Mean bias: {analysis['mean_bias']:+.4f}")
        print(f"Next: Experiment Step 3: (Imputation under MCAR)")
        return True
    except ImportError as e:
        print(f"Required module not found: {e}")
        print("Make sure simulation/missingness.py exists")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def phase_3_imputation():
    """Phase 3: Imputation baseline on MCAR"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 3 : IMPUTATION PERFORMANCE under MCAR")
    print("="*60 + "\n")


    try:
        from simulation.imputation import impute_all_methods
        from simulation.evaluation import compare_methods, print_evaluation_report
        import pandas as pd
        from pathlib import Path

        config = get_config()
        data_true = pd.read_csv('data/raw/ground_truth.csv')
        data_mcar = pd.read_csv('data/processed/mcar_data.csv')

        target = config.missingness.target_sensor
        predictors = config.imputation.regression_predictors
        
        print(f" Target; {target}")
        print(f"  Predictors: {predictors}")
        print(f"  Missing rate: {data_mcar[target].isna().mean():.1%}")

        # Apply all imputations methods
        print("\n" + "-"*60)
        print("APPLYING IMPUTATION METHODS")
        print("-"*60)
        
        imputed_datasets = impute_all_methods(
            data_mcar,
            target,
            predictors,
            methods=['mean', 'forward_fill', 'linear_interpolation', 'regression', 'knn']
        )

        # Evaluate and compare
        print("\n" + "-"*60)
        print("EVALUATING IMPUTATION QUALITY")
        print("-"*60)
        
        comparison = compare_methods(data_true, imputed_datasets, target)
        
        # Save results
        metrics_dir = Path('results/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(metrics_dir / 'phase3_imputation_comparison.csv', index=False)
        
        # Save imputed datasets
        for method, data_imputed in imputed_datasets.items():
            if data_imputed is not None:
                output_path = Path('data/processed') / f'mcar_{method}_imputed.csv'
                data_imputed.to_csv(output_path, index=False)
        
        # Print report
        print_evaluation_report(comparison)
        
        print("\n Step 3 Completed: MCAR Imputation Evaluation Finished")
        print(f" Results saved to: {metrics_dir / 'phase3_imputation_comparison.csv'}")
        print("Next: Experiment Step 4 (MAR Missingness)")
        
        return True
    
    except ImportError as e:
        print(f"Required module not found: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_4_mar():
    """Phase 4: MAR mechanism"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 4: MAR (Missing At Random)")
    print("="*60 + "\n")
    
    try:
        from simulation.missingness import inject_mar, analyze_missingness, visualize_missingness
        from simulation.imputation import impute_all_methods
        from simulation.evaluation import compare_methods, print_evaluation_report
        import pandas as pd
        from pathlib import Path
        
        config = get_config()
        
        print(" Loading ground truth...")
        data = pd.read_csv('data/raw/ground_truth.csv')
        
        target = config.missingness.target_sensor
        predictor = config.missingness.mar_predictor
        
        print(f"\n Injecting MAR missingness...")
        print(f"  Target: {target}")
        print(f"  Predictor: {predictor}")
        
        data_mar = inject_mar(
            data.copy(),
            target_column=target,
            predictor_column=predictor,
            alpha=config.missingness.mar_alpha,
            intercept=config.missingness.mar_intercept,
            seed=config.seed
        )
        
        print("\n |Analyzing missingness pattern...")
        analysis = analyze_missingness(data, data_mar, target)
        
        print(f"  |-Missing rate: {analysis['missing_rate']:.1%}")
        print(f"  |-Mean bias: {analysis['mean_bias']:+.4f}")
        
        # Save data
        output_dir = Path('data/processed')
        data_mar.to_csv(output_dir / 'mar_data.csv', index=False)
        
        # Visualize
        print("\n Creating diagnostic plots...")
        fig_dir = Path('results/figures')
        visualize_missingness(
            data, data_mar, target, 'MAR',
            output_path=str(fig_dir / 'phase4_mar_analysis.png')
        )
        
        # Apply imputation methods
        print("\n" + "-"*60)
        print("APPLYING IMPUTATION METHODS TO MAR DATA")
        print("-"*60)
        
        predictors = config.imputation.regression_predictors
        imputed_datasets = impute_all_methods(
            data_mar,
            target,
            predictors,
            methods=['mean', 'forward_fill', 'linear_interpolation', 'regression', 'knn']
        )
        
        # Evaluate
        print("\n" + "-"*60)
        print("EVALUATING IMPUTATION QUALITY ON MAR")
        print("-"*60)
        
        comparison = compare_methods(data, imputed_datasets, target)
        
        # Save results
        metrics_dir = Path('results/metrics')
        comparison.to_csv(metrics_dir / 'phase4_mar_imputation_comparison.csv', index=False)
        
        # Save imputed datasets
        for method, data_imputed in imputed_datasets.items():
            if data_imputed is not None:
                output_path = output_dir / f'mar_{method}_imputed.csv'
                data_imputed.to_csv(output_path, index=False)
        
        # Print report
        print_evaluation_report(comparison)
        
        print("\n Step 4 Complete: MAR Missingness and Imputation Analysis Finished")
        print(f" MAR data saved to: {output_dir / 'mar_data.csv'}")
        print(f" Results saved to: {metrics_dir / 'phase4_mar_imputation_comparison.csv'}")
        print(" Next: Experiment Step 5 (MNAR Missingness)")
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_5_mnar():
    """Phase 5: MNAR mechanism"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 5: MNAR (Missing Not At Random)")
    print("="*60 + "\n")
    
    try:
        from simulation.missingness import inject_mnar, analyze_missingness, visualize_missingness
        from simulation.imputation import impute_all_methods
        from simulation.evaluation import compare_methods, print_evaluation_report
        import pandas as pd
        from pathlib import Path
        
        config = get_config()
        
        print("Loading ground truth...")
        data = pd.read_csv('data/raw/ground_truth.csv')
        
        target = config.missingness.target_sensor
        
        print(f"\n Injecting MNAR missingness...")
        print(f"  Target: {target}")
        print(f"    Missingness depends on UNOBSERVED {target} values")
        
        data_mnar = inject_mnar(
            data.copy(),
            target_column=target,
            beta=config.missingness.mnar_beta,
            intercept=config.missingness.mnar_intercept,
            seed=config.seed
        )
        
        print("\n Analyzing missingness pattern...")
        analysis = analyze_missingness(data, data_mnar, target)
        
        print(f"  Missing rate: {analysis['missing_rate']:.1%}")
        print(f"  Mean bias: {analysis['mean_bias']:+.4f}")
        
        # Save data
        output_dir = Path('data/processed')
        data_mnar.to_csv(output_dir / 'mnar_data.csv', index=False)
        
        # Visualize
        print("\n Creating diagnostic plots...")
        fig_dir = Path('results/figures')
        visualize_missingness(
            data, data_mnar, target, 'MNAR',
            output_path=str(fig_dir / 'phase5_mnar_analysis.png')
        )
        
        # Apply imputation methods
        print("\n" + "-"*60)
        print("APPLYING IMPUTATION METHODS TO MNAR DATA")
        print("-"*60)
        
        predictors = config.imputation.regression_predictors
        imputed_datasets = impute_all_methods(
            data_mnar,
            target,
            predictors,
            methods=['mean', 'forward_fill', 'linear_interpolation', 'regression', 'knn']
        )
        
        # Evaluate
        print("\n" + "-"*60)
        print("EVALUATING IMPUTATION QUALITY ON MNAR")
        print("-"*60)
        
        comparison = compare_methods(data, imputed_datasets, target)
        
        # Save results
        metrics_dir = Path('results/metrics')
        comparison.to_csv(metrics_dir / 'phase5_mnar_imputation_comparison.csv', index=False)
        
        # Save imputed datasets
        for method, data_imputed in imputed_datasets.items():
            if data_imputed is not None:
                output_path = output_dir / f'mnar_{method}_imputed.csv'
                data_imputed.to_csv(output_path, index=False)
        
        # Print report
        print_evaluation_report(comparison)
        
        print("\n Step 5 Complete: MNAR Missingness and Imputation Analysis Finished")
        print(f" MNAR data saved to: {output_dir / 'mnar_data.csv'}")
        print(f" Results saved to: {metrics_dir / 'phase5_mnar_imputation_comparison.csv'}")
        print(" Next: Experiment Step 6 (Cross-Mechanism Comparison)")
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_6_advanced():
    """Phase 6: Cross-mechanism comparison and visualization"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 6: CROSS-MECHANISM COMPARISON")
    print("="*60 + "\n")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np
        
        print("Loading evaluation results...")
        
        metrics_dir = Path('results/metrics')
        
        # Load all comparison results
        mcar_results = pd.read_csv(metrics_dir / 'phase3_imputation_comparison.csv')
        mar_results = pd.read_csv(metrics_dir / 'phase4_mar_imputation_comparison.csv')
        mnar_results = pd.read_csv(metrics_dir / 'phase5_mnar_imputation_comparison.csv')
        
        # Add mechanism column
        mcar_results['mechanism'] = 'MCAR'
        mar_results['mechanism'] = 'MAR'
        mnar_results['mechanism'] = 'MNAR'
        
        # Combine
        all_results = pd.concat([mcar_results, mar_results, mnar_results], ignore_index=True)
        
        # Save combined results
        all_results.to_csv(metrics_dir / 'phase6_all_mechanisms_comparison.csv', index=False)
        print(f"   Combined results saved")
        
        # Create comparison visualizations
        print("\n Creating comparison visualizations...")
        fig_dir = Path('results/figures')
        
        # Get unique methods
        methods = all_results['method'].unique()
        mechanisms = ['MCAR', 'MAR', 'MNAR']
        
        # Plot 1: Bias comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, mech in enumerate(mechanisms):
            data = all_results[all_results['mechanism'] == mech]
            biases = [data[data['method'] == m]['bias'].values[0] for m in methods]
            ax.bar(x + i*width, biases, width, label=mech, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Imputation Method')
        ax.set_ylabel('Bias')
        ax.set_title('Bias Comparison Across Mechanisms')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase6_bias_comparison.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase6_bias_comparison.png")
        plt.close()
        
        # Plot 2: RMSE comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, mech in enumerate(mechanisms):
            data = all_results[all_results['mechanism'] == mech]
            rmses = [data[data['method'] == m]['rmse'].values[0] for m in methods]
            ax.bar(x + i*width, rmses, width, label=mech, alpha=0.8)
        
        ax.set_xlabel('Imputation Method')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison Across Mechanisms')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase6_rmse_comparison.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase6_rmse_comparison.png")
        plt.close()
        
        # Plot 3: Variance ratio comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, mech in enumerate(mechanisms):
            data = all_results[all_results['mechanism'] == mech]
            var_ratios = [data[data['method'] == m]['variance_ratio'].values[0] for m in methods]
            ax.bar(x + i*width, var_ratios, width, label=mech, alpha=0.8)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Ideal (1.0)')
        ax.set_xlabel('Imputation Method')
        ax.set_ylabel('Variance Ratio')
        ax.set_title('Variance Preservation Across Mechanisms')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase6_variance_comparison.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase6_variance_comparison.png")
        plt.close()
        
        # Plot 4: Heatmap of RMSE by method and mechanism
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = all_results.pivot(index='method', columns='mechanism', values='rmse')
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(mechanisms)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(mechanisms)
        ax.set_yticklabels(methods)
        
        # Add values
        for i in range(len(methods)):
            for j in range(len(mechanisms)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('RMSE Heatmap: Method × Mechanism')
        plt.colorbar(im, ax=ax, label='RMSE (lower is better)')
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase6_rmse_heatmap.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase6_rmse_heatmap.png")
        plt.close()
        
        # Summary statistics
        print("\n" + "="*60)
        print("CROSS-MECHANISM ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n Best method per mechanism (by RMSE):")
        for mech in mechanisms:
            data = all_results[all_results['mechanism'] == mech]
            best = data.loc[data['rmse'].idxmin()]
            print(f"  {mech:6s}: {best['method']:20s} (RMSE={best['rmse']:.4f})")
        
        print("\n Most robust method across all mechanisms:")
        avg_rmse = all_results.groupby('method')['rmse'].mean().sort_values()
        print(f"  {avg_rmse.index[0]:20s} (Avg RMSE={avg_rmse.values[0]:.4f})")
        
        print("\n Method performance degradation (MCAR → MNAR):")
        for method in methods:
            mcar_rmse = all_results[(all_results['mechanism']=='MCAR') & (all_results['method']==method)]['rmse'].values[0]
            mnar_rmse = all_results[(all_results['mechanism']=='MNAR') & (all_results['method']==method)]['rmse'].values[0]
            degradation = ((mnar_rmse - mcar_rmse) / mcar_rmse) * 100
            print(f"  {method:20s}: {degradation:+6.1f}%")
        
        print("\n Step 6 Complete: Cross-Mechanism Comparison Finished")
        print(f" Combined results: {metrics_dir / 'phase6_all_mechanisms_comparison.csv'}")
        print(f" Generated 4 comparison plots")
        print(" Next: Experiment Step 7 (Sensitivity Analysis)")
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_7_evaluation():
    """Phase 7: Sensitivity analysis"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 7: Sensitivity Analysis Across Missing Rates")
    print("="*60 + "\n")
    
    try:
        from simulation.missingness import inject_mcar, inject_mar, inject_mnar
        from simulation.imputation import apply_imputation
        from simulation.evaluation import compute_bias, compute_rmse, compute_variance_ratio
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np
        
        config = get_config()
        
        print("Loading ground truth...")
        data = pd.read_csv('data/raw/ground_truth.csv')
        target = config.missingness.target_sensor
        predictors = config.imputation.regression_predictors
        
        # Test different missing rates
        missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        methods_to_test = ['mean', 'forward_fill', 'linear_interpolation', 'regression']
        mechanisms = ['MCAR', 'MAR', 'MNAR']
        
        print(f"\n Testing {len(missing_rates)} missing rates...")
        print(f"   Methods: {methods_to_test}")
        print(f"   Mechanisms: {mechanisms}")
        
        results_list = []
        
        for mechanism in mechanisms:
            print(f"\n{'='*60}")
            print(f"Testing {mechanism}")
            print(f"{'='*60}")
            
            for rate in missing_rates:
                print(f"\n  Missing rate: {rate:.1%}")
                
                # Inject missingness
                if mechanism == 'MCAR':
                    data_missing = inject_mcar(data.copy(), target, rate, seed=42)
                elif mechanism == 'MAR':
                    # Adjust intercept to achieve desired rate
                    alpha = config.missingness.mar_alpha
                    intercept = -2.0 + (rate - 0.3) * 5  # Rough adjustment
                    data_missing = inject_mar(
                        data.copy(), target,
                        config.missingness.mar_predictor,
                        alpha, intercept, seed=42
                    )
                else:  # MNAR
                    beta = config.missingness.mnar_beta
                    intercept = -1.5 + (rate - 0.3) * 5
                    data_missing = inject_mnar(
                        data.copy(), target, beta, intercept, seed=42
                    )
                
                actual_rate = data_missing[target].isna().mean()
                
                # Test each method
                for method in methods_to_test:
                    try:
                        data_imputed = apply_imputation(
                            data_missing.copy(), target, method, predictors
                        )
                        
                        bias = compute_bias(data[target].values, data_imputed[target].values)
                        rmse = compute_rmse(data[target].values, data_imputed[target].values)
                        var_ratio = compute_variance_ratio(data[target].values, data_imputed[target].values)
                        
                        results_list.append({
                            'mechanism': mechanism,
                            'target_rate': rate,
                            'actual_rate': actual_rate,
                            'method': method,
                            'bias': bias,
                            'rmse': rmse,
                            'variance_ratio': var_ratio
                        })
                        
                    except Exception as e:
                        print(f"    ⚠️  {method} failed: {e}")
        
        # Create results DataFrame
        sensitivity_results = pd.DataFrame(results_list)
        
        # Save results
        metrics_dir = Path('results/metrics')
        sensitivity_results.to_csv(metrics_dir / 'phase7_sensitivity_analysis.csv', index=False)
        print(f"\n Sensitivity data saved")
        
        # Visualizations
        print("\n Creating sensitivity plots...")
        fig_dir = Path('results/figures')
        
        # Plot 1: RMSE vs Missing Rate for each mechanism
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, mechanism in enumerate(mechanisms):
            ax = axes[idx]
            data_mech = sensitivity_results[sensitivity_results['mechanism'] == mechanism]
            
            for method in methods_to_test:
                data_method = data_mech[data_mech['method'] == method]
                ax.plot(data_method['actual_rate'], data_method['rmse'], 
                       'o-', label=method, linewidth=2, markersize=6)
            
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{mechanism}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('RMSE vs Missing Rate by Mechanism', y=1.02)
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase7_rmse_vs_rate.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase7_rmse_vs_rate.png")
        plt.close()
        
        # Plot 2: Bias vs Missing Rate
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, mechanism in enumerate(mechanisms):
            ax = axes[idx]
            data_mech = sensitivity_results[sensitivity_results['mechanism'] == mechanism]
            
            for method in methods_to_test:
                data_method = data_mech[data_mech['method'] == method]
                ax.plot(data_method['actual_rate'], data_method['bias'], 
                       'o-', label=method, linewidth=2, markersize=6)
            
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Bias')
            ax.set_title(f'{mechanism}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Bias vs Missing Rate by Mechanism', y=1.02)
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase7_bias_vs_rate.png', dpi=100, bbox_inches='tight')
        print(f"   Saved: phase7_bias_vs_rate.png")
        plt.close()
        
        # Plot 3: Variance Ratio vs Missing Rate
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, mechanism in enumerate(mechanisms):
            ax = axes[idx]
            data_mech = sensitivity_results[sensitivity_results['mechanism'] == mechanism]
            
            for method in methods_to_test:
                data_method = data_mech[data_mech['method'] == method]
                ax.plot(data_method['actual_rate'], data_method['variance_ratio'], 
                       'o-', label=method, linewidth=2, markersize=6)
            
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Ideal')
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Variance Ratio')
            ax.set_title(f'{mechanism}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Variance Preservation vs Missing Rate by Mechanism', y=1.02)
        plt.tight_layout()
        plt.savefig(fig_dir / 'phase7_variance_vs_rate.png', dpi=100, bbox_inches='tight')
        print(f"  Saved: phase7_variance_vs_rate.png")
        plt.close()
        
        # Summary findings
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n Performance at 70% missing rate:")
        high_missing = sensitivity_results[sensitivity_results['actual_rate'] >= 0.65]
        for mechanism in mechanisms:
            data_mech = high_missing[high_missing['mechanism'] == mechanism]
            if len(data_mech) > 0:
                best = data_mech.loc[data_mech['rmse'].idxmin()]
                print(f"  {mechanism:6s}: {best['method']:20s} (RMSE={best['rmse']:.4f})")
        
        print("\n Methods that maintain RMSE < 5.0 up to 50% missing:")
        mid_missing = sensitivity_results[
            (sensitivity_results['actual_rate'] >= 0.45) & 
            (sensitivity_results['actual_rate'] <= 0.55)
        ]
        good_methods = mid_missing[mid_missing['rmse'] < 5.0]
        if len(good_methods) > 0:
            for _, row in good_methods.iterrows():
                print(f"  {row['mechanism']:6s} - {row['method']:20s} (RMSE={row['rmse']:.4f})")
        else:
            print("  None - all methods degrade significantly")
        
        print("\n Step 7 Complete: Sensitivity Curves and Breaking Points Identified")
        print(f" Results: {metrics_dir / 'phase7_sensitivity_analysis.csv'}")
        print(f" Generated 3 sensitivity plots")
        print(" Next: Experiment Step 8 (Final Synthesis)")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_8_visualization():
    """Phase 8: Final visualization suite and summary report"""
    print("\n" + "="*60)
    print("EXPERIMENT STEP 8: Final Synthesis and Study Report")
    print("="*60 + "\n")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np
        
        metrics_dir = Path('results/metrics')
        fig_dir = Path('results/figures')
        
        print(" Loading all results...")
        
        # Load key results
        all_mechanisms = pd.read_csv(metrics_dir / 'phase6_all_mechanisms_comparison.csv')
        sensitivity = pd.read_csv(metrics_dir / 'phase7_sensitivity_analysis.csv')
        
        # Create final summary report
        print("\n Generating final summary report...")
        
        report = []
        report.append("="*80)
        report.append("MISSING DATA MECHANISMS: CASE STUDY FINAL REPORT")
        report.append("="*80)
        report.append("")
        report.append("PROJECT OBJECTIVE")
        report.append("-"*80)
        report.append("Rigorous study of MCAR, MAR, and MNAR missing data mechanisms using")
        report.append("simulated real-time sensor data, proving mathematically and empirically")
        report.append("how different handling techniques succeed or fail.")
        report.append("")
        
        report.append("="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        report.append("")
        
        # Finding 1: Best method per mechanism
        report.append("1. BEST IMPUTATION METHOD BY MECHANISM (30% missing)")
        report.append("-"*80)
        for mech in ['MCAR', 'MAR', 'MNAR']:
            data = all_mechanisms[all_mechanisms['mechanism'] == mech]
            best = data.loc[data['rmse'].idxmin()]
            report.append(f"   {mech:6s}: {best['method']:20s}")
            report.append(f"           RMSE: {best['rmse']:.4f}, Bias: {best['bias']:+.4f}, Var Ratio: {best['variance_ratio']:.4f}")
        report.append("")
        
        # Finding 2: Most robust method
        report.append("2. MOST ROBUST METHOD ACROSS ALL MECHANISMS")
        report.append("-"*80)
        avg_rmse = all_mechanisms.groupby('method').agg({
            'rmse': 'mean',
            'bias': lambda x: np.mean(np.abs(x)),
            'variance_ratio': 'mean'
        }).sort_values('rmse')
        best_overall = avg_rmse.index[0]
        report.append(f"   {best_overall}")
        report.append(f"   Average RMSE: {avg_rmse.loc[best_overall, 'rmse']:.4f}")
        report.append(f"   Average |Bias|: {avg_rmse.loc[best_overall, 'bias']:.4f}")
        report.append("")
        
        # Finding 3: Mechanism difficulty
        report.append("3. MECHANISM DIFFICULTY RANKING")
        report.append("-"*80)
        mech_difficulty = all_mechanisms.groupby('mechanism')['rmse'].mean().sort_values()
        report.append("   (Lower RMSE = easier to handle)")
        for mech, rmse in mech_difficulty.items():
            report.append(f"   {mech:6s}: {rmse:.4f}")
        report.append("")
        
        # Finding 4: Sensitivity to missing rate
        report.append("4. BREAKING POINTS (when RMSE doubles from 10% → 70% missing)")
        report.append("-"*80)
        methods = sensitivity['method'].unique()
        for method in methods:
            breaking_points = []
            for mech in ['MCAR', 'MAR', 'MNAR']:
                data = sensitivity[(sensitivity['method']==method) & (sensitivity['mechanism']==mech)]
                if len(data) > 0:
                    baseline = data[data['actual_rate'] <= 0.15]['rmse'].mean()
                    doubled = data[data['rmse'] >= baseline * 2]
                    if len(doubled) > 0:
                        breaking_rate = doubled['actual_rate'].min()
                        breaking_points.append(f"{mech}:{breaking_rate:.0%}")
            if breaking_points:
                report.append(f"   {method:20s}: {', '.join(breaking_points)}")
        report.append("")
        
        # Finding 5: Variance shrinkage
        report.append("5. VARIANCE SHRINKAGE ANALYSIS")
        report.append("-"*80)
        report.append("   Methods that preserve variance (ratio ≈ 1.0):")
        good_variance = all_mechanisms[
            (all_mechanisms['variance_ratio'] >= 0.9) & 
            (all_mechanisms['variance_ratio'] <= 1.1)
        ]
        for _, row in good_variance.iterrows():
            report.append(f"   {row['mechanism']:6s} - {row['method']:20s} ({row['variance_ratio']:.3f})")
        report.append("")
        
        report.append("="*80)
        report.append("PRACTICAL RECOMMENDATIONS")
        report.append("="*80)
        report.append("")
        
        # Determine best method for each scenario
        mcar_best = all_mechanisms[all_mechanisms['mechanism']=='MCAR'].loc[
            all_mechanisms[all_mechanisms['mechanism']=='MCAR']['rmse'].idxmin(), 'method'
        ]
        mar_best = all_mechanisms[all_mechanisms['mechanism']=='MAR'].loc[
            all_mechanisms[all_mechanisms['mechanism']=='MAR']['rmse'].idxmin(), 'method'
        ]
        mnar_best = all_mechanisms[all_mechanisms['mechanism']=='MNAR'].loc[
            all_mechanisms[all_mechanisms['mechanism']=='MNAR']['rmse'].idxmin(), 'method'
        ]
        
        report.append("1. If data is MCAR (random missingness):")
        report.append(f"   → Use {mcar_best}")
        report.append("")
        
        report.append("2. If data is MAR (depends on observed variables):")
        report.append(f"   → Use {mar_best}")
        report.append("   → Incorporate predictor variables")
        report.append("")
        
        report.append("3. If data is MNAR (depends on unobserved values):")
        report.append(f"   → Use {mnar_best} but expect bias")
        report.append("   → Consider sensitivity analysis")
        report.append("   → Document assumptions explicitly")
        report.append("")
        
        report.append("4. General guidelines:")
        report.append("   → Missing rate < 30%: Most methods work reasonably")
        report.append("   → Missing rate 30-50%: Use model-based methods")
        report.append("   → Missing rate > 50%: Results will be unreliable")
        report.append("   → Always validate with domain knowledge")
        report.append("")
        
        report.append("="*80)
        report.append("TECHNICAL DETAILS")
        report.append("="*80)
        report.append("")
        report.append(f"Data generation: AR(1) processes with cross-correlations")
        report.append(f"Timesteps: 1000")
        report.append(f"Sensors: temperature, humidity, pollution, vibration")
        report.append(f"Target sensor: pollution")
        report.append(f"Random seed: 42 (fully reproducible)")
        report.append("")
        report.append("Imputation methods tested:")
        report.append("  - Mean imputation (baseline)")
        report.append("  - Forward fill (time series baseline)")
        report.append("  - Linear interpolation")
        report.append("  - Regression imputation (model-based)")
        report.append("  - KNN imputation (model-based)")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = Path('results') / 'FINAL_REPORT.txt'
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(report_text)
        
        # Create final summary figure
        print("\n Creating final summary visualization...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: RMSE by mechanism (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        methods = all_mechanisms['method'].unique()
        x = np.arange(len(methods))
        width = 0.25
        for i, mech in enumerate(['MCAR', 'MAR', 'MNAR']):
            data = all_mechanisms[all_mechanisms['mechanism'] == mech]
            rmses = [data[data['method']==m]['rmse'].values[0] for m in methods]
            ax1.bar(x + i*width, rmses, width, label=mech, alpha=0.8)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Bias by mechanism (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        for i, mech in enumerate(['MCAR', 'MAR', 'MNAR']):
            data = all_mechanisms[all_mechanisms['mechanism'] == mech]
            biases = [data[data['method']==m]['bias'].values[0] for m in methods]
            ax2.bar(x + i*width, biases, width, label=mech, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Bias')
        ax2.set_title('Bias Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Variance ratio (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        for i, mech in enumerate(['MCAR', 'MAR', 'MNAR']):
            data = all_mechanisms[all_mechanisms['mechanism'] == mech]
            var_ratios = [data[data['method']==m]['variance_ratio'].values[0] for m in methods]
            ax3.bar(x + i*width, var_ratios, width, label=mech, alpha=0.8)
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Variance Ratio')
        ax3.set_title('Variance Preservation')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4-6: Sensitivity curves (bottom row)
        for idx, mech in enumerate(['MCAR', 'MAR', 'MNAR']):
            ax = fig.add_subplot(gs[1, idx])
            data_mech = sensitivity[sensitivity['mechanism'] == mech]
            for method in ['mean', 'forward_fill', 'linear_interpolation', 'regression']:
                data_method = data_mech[data_mech['method'] == method]
                if len(data_method) > 0:
                    ax.plot(data_method['actual_rate']*100, data_method['rmse'], 
                           'o-', label=method, linewidth=2, markersize=4)
            ax.set_xlabel('Missing Rate (%)')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{mech} Sensitivity')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Plot 7-9: Distribution metrics (third row)
        metrics_to_plot = ['kl_divergence', 'ks_statistic', 'acf_rmse']
        titles = ['KL Divergence', 'KS Statistic', 'ACF RMSE']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = fig.add_subplot(gs[2, idx])
            for i, mech in enumerate(['MCAR', 'MAR', 'MNAR']):
                data = all_mechanisms[all_mechanisms['mechanism'] == mech]
                values = [data[data['method']==m][metric].values[0] for m in methods]
                ax.bar(x + i*width, values, width, label=mech, alpha=0.8)
            ax.set_xlabel('Method')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x + width)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Missing Data Mechanisms: Complete Analysis Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(fig_dir / 'phase8_final_summary.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: phase8_final_summary.png")
        plt.close()
        
        print("\n EXPERIMENT STEP 8: Final Syntesis and Study Report")
        print(f" Final report: {report_path}")
        print(f" Final summary figure: {fig_dir / 'phase8_final_summary.png'}")
        print("\n ALL PHASES COMPLETE!")
        print("\n" + "="*60)
        print("STUDY EXECUTION COMPLETE - REFERENCE RESULTS GENERATED")
        print("="*60)
        print(f"\nResults location:")
        print(f"  Data: data/")
        print(f"  Figures: results/figures/")
        print(f"  Metrics: results/metrics/")
        print(f"  Final report: results/FINAL_REPORT.txt")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Missing Data Mechanisms Case Study - Phase Executor'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=range(0, 9),
        help='Phase number to execute (0-8)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all phases sequentially'
    )
    
    args = parser.parse_args()
    
    phases = {
        1: phase_1_ground_truth,
        2: phase_2_mcar,
        3: phase_3_imputation,
        4: phase_4_mar,
        5: phase_5_mnar,
        6: phase_6_advanced,
        7: phase_7_evaluation,
        8: phase_8_visualization
    }

    print("\n" + "=" * 80)
    print(STUDY_NAME)
    print(f"Version: {STUDY_VERSION}")
    print("Focus: MCAR * MAR * MNAR")
    print("=" * 80 + "\n")
    
    # If --all flag is set, run all phases
    if args.all:
        print("\n" + "🚀"*30)
        print("RUNNING FULL REFERENCE STUDY (ALL EXPERIMENT STEPS)")
        print("🚀"*30 + "\n")
        
        for phase_num in sorted(phases.keys()):
            print(f"\n{'='*60}")
            print(f"STARTING PHASE {phase_num}")
            print(f"{'='*60}\n")
            
            phase_func = phases[phase_num]
            success = phase_func()
            
            if not success:
                print(f"\n Phase {phase_num} failed. Stopping execution.")
                sys.exit(1)
            
            print(f"\n Phase {phase_num} completed successfully!")
        
        print("\n" + "🎉"*30)
        print("ALL PHASES COMPLETED SUCCESSFULLY!")
        print("🎉"*30)
        print("\nCheck results/ directory for outputs")
        sys.exit(0)
    
    # Otherwise run single phase
    if args.phase is None:
        parser.print_help()
        print("\n Error: Either --phase or --all must be specified")
        sys.exit(1)
    
    phase_func = phases[args.phase]
    success = phase_func()
    
    if not success:
        print("\n  Phase did not complete successfully")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
    


