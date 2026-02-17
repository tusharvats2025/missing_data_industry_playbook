# Missing Data Industry Playbook - v1
This repository presents a systematic simulation-based evaluation of missing data imputation strategies for multivariate time-series sensor data, inspired by industrial IoT, SDN telemetry, any EV/V2G monitoring environments

The framework explicitly models:
 -- Realistic temporal dynamics.
 -- Cross-sensor dependencies.
 -- Three fundamental missingness mechanism: MCAR, MAR, MNAR

## 1️⃣ Ground Truth Generation & Validation
Before evaluating any imputation strategy, it is critical to validate that the ground truth data itself is statistically and physically plausible.
This phase establishes the baseline signal properties that all later imputation results depend on.

### Ground Truth Time Series (All Sensors)

![Ground Truth Time Series](results/figures/phase1_timeseries.png)

#### What this shows
 -- Smooth but non-trivial temporal evolution
 -- Different volatility regimes across sensors
 -- Absence of artificial discontinuties.

#### What this matters
 -- Time-aware imputers (linear interpolation, forward fill) implicitly assume temporal continuity

 -- High-frequency variables(e.g., vibration) create a stress test for naive methods.


### Ground Truth Distributions

![Ground Truth Distributions](results/figures/phase1_distribution.png)

#### What this shows

 -- Non-identical marginal distributions across variables.
 -- Different variances and tail behaviors.

#### Why this matters

 -- Mean imputation collapses variance.
 -- Distributional metrics (KL divergence, KS test) are necessary alongside RMSE.

### Cross-Sensor Correlation Heatmap

![Correlation Heatmap](results/figures/phase1_correlations.png)

#### What this shows
 -- Strong inter-variable dependencies (e.g.,     temperature–humidity).
 -- Moderate correlations across other sensors.

#### Why this matters
 -- Enables regression-based imputation under MAR.
 -- Demonstrates that the dataset is not a collection of independent signals.

### Autocorrelation Structure (ACF)

![Autocorrelation](results/figures/phase1_autocorrelation.png)

#### What this shows

 -- High lag-1 autocorrelation across all sensors.
 -- Gradual decay rather than white noise.

#### Why this matters

 -- Justifies linear interpolation effectiveness.
 -- Explains why forward fill can appear stable but fail during regime shifts.

## 2️⃣ MCAR Missingness Injection & Diagnostics
This phase evaluates imputation under Missing Completely At Random (MCAR) — the most benign but still non-trivial missingness mechanism.

### MCAR Missingness Analysis

![MCAR Missingness Analysis](results/figures/phase2_mcar_analysis.png)

#### Key characteristics
 -- Missingness is uniformly random.
 -- Independent of observed and unobserved values.
 -- No systematic bias introduced by the missingness process itself.

#### Implication
 -- Any degradation in performance is due to the imputation method, not the mechanism

## 3️⃣ Imputation Performance under MCAR
This phase benchmarks multiple imputation strategies under MCAR using:
 -- RMSE
 -- MAE
 -- Bias
 -- Variance ratio
 -- Distributional similarity

#### Observed behaviour (from metrics & CSVs)
 -- Linear interpolation consistently acheives the lowest RMSE.
 -- Forward fill degrades as missing gaps lengthen
 -- Mean imputaion introduces strong variance collapse. 
 -- Regression ans kNN provide no advantage under MCAR

#### Key Takeway
 -----> Under MCAR, exploiting temporal continuity is more effective than exploiting cross-variable correlation. 

## 4️⃣ MAR Missingness & Conditional Bias
MAR introduces missingness that depends on observed varaibles, creating structured bias.
### MAR Diagnostic Plot
![MAR Missingness Analysis](results/figures/phase4_mar_analysis.png)

#### What this shows

 -- Missingness probability increases conditionally(e.g., based on temperature or humidity).
 -- The dataset remains recoverable in principle

### MAR Imputation Comparison

#### Observed behavior
 -- Regression-based imputation improves relative to MCAR.
 -- Linear interpolation remains strong.
 -- Forward fill continues to lag during volatile periods.

#### Key Takeway
 MAR rewards methods that leverage cross-variable relationships.

## 5️⃣ MNAR Missingness & Structural Bias
MNAR represents the most challenging and realistic failure mode.
### MNAR Diagnostic Visualisation
![MNAR Missingness Analysis](results/figures/phase5_mnar_analysis.png)

#### What this shows

 -- Missingness depends on the unobserved value itself.
 -- Introduces selection bias.
 -- Violates standard recoverability assumptions

#### Observed behavior

 -- Regression performance degrades.
 -- Linear interpolation remains comparatively robust.
 -- Mean imputation severely underestimates extremes

#### Key takeaway

 -----> Under MNAR, no method can recover truth without additional assumptions.

## 6️⃣ Cross-Mechanism Comparison
This phase consolidates results across MCAR, MAR, and MNAR.

### Bias Comparison
![Bias Comparison](results/figures/phase6_bias_comparison.png)

 -- Mean imputation shows systematic bias.
 -- Forward fill bias increases with mechanism severity.

### RMSE Comparison
![RMSE Comparsion](results/figures/phase6_rmse_comparison.png)
 
 -- No universally optimal method.
 -- Linear interpolation shows strongest robustness.

### Varience Preservation
![Varience Comparison](results/figures/phase6_variance_comparison.png)

 -- Mean imputation collapse variance.
 -- Forward fill inflates or collapse variance depending on gap structure.

### RMSE Heatmap
![RMSE Heatmap](results/figures/phase6_rmse_heatmap.png)

#### Key Insight
 -----> Performance depends jointly on method x missingness mechansim, not method alone


## 7️⃣ Sensitivity Analysis Across Missing Rates
This phase studies how performance changes as missingness increases.

### RMSE vs Missing Rate
![RMSE Sensitivity](results/figures/phase7_rmse_vs_rate.png)

 -- Forward fill error grows non-linearly.
 -- Linear interpolation degrades gracefully.

### Bias vs Missing Rate
![Bias Sensitivity](results/figures/phase7_bias_vs_rate.png)
 
 -- Bias accumulation becomes dominant at high missigness.

### Varience vs Missingness Rate
![Vairence Sensitivity](results/figures/phase7_variance_vs_rate.png)

#### Critical insight
 -----> Forward fill creates long flat segements, causing instability during regime changes.

## 8️⃣ Final Synthesis & Key Insights
![Final Summary](results/figures/phase8_final_summary.png)

#### What v1 is Capable of
 -- Mechanism-aware imputation evaluation.
 -- Distributional and temporal diagnostics.
 -- Failure-mode analysis (not just accuracy ranking).

#### Known Limitations (Expilicitly Acknowledged)
 -- Single imputation only (no uncertainty quantification).
 -- No confidence intervals.
 -- Ground truth schema is fixed.

#### v2 Roadmap(Plug-and-Play Extension)

Planned enhancements:
 -- CSV schema adapter to map arbitrary datasets → ground_truth.csv.
 -- Automated preprocessing (scaling, alignment, stationarity checks).
 -- Multiple Imputation (MI) with uncertainty bounds.
 -- Domain-specific templates (SDN, IoT, EV/V2G).
