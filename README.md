# Forecasting and Interpreting Urban Air Pollution Under Weather, Traffic and Airport Influenc

Last updated: `2026-05-17`

## Reproducibility and Execution Guide

This repository is organized so that the analyses can be reproduced from the
merged hourly dataset and, when available, from the original raw data sources.
The complete raw-data reconstruction requires access to all source datasets,
including the airport operation records obtained within the S4C project. Since
the airport records are not public, the most practical reproducibility entry
point is the merged dataset:

- `Datasets_Raw/hourly_merged_2023_2025.csv`

The expected merged dataset contains hourly observations indexed by `datetime`,
with pollutant targets, BLQ airport activity, urban traffic loop counts,
meteorological variables, and station-level context. The current reference
dataset has:

- `9,792` hourly rows;
- `61` columns;
- time span `2024-05-29 00:00:00` -> `2025-07-10 23:00:00`.

### Environment

The software environment is specified in:

- `environment.yml`

It defines a Python `3.11` conda environment with the packages required for
forecasting, ablation, SHAP, plotting, document generation, and figure
conversion.

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate aira-local
```

If the environment already exists and needs to be updated:

```bash
conda env update -f environment.yml --prune
conda activate aira-local
```

The main scientific dependencies are:

- `pandas`, `numpy`, `scipy`;
- `scikit-learn`;
- `xgboost`;
- `shap`;
- `matplotlib`, `seaborn`;
- `cairosvg`, `pillow`;
- `tqdm`, `openpyxl`, `python-docx`.

### Reproducibility Levels

There are three practical levels of reproducibility.

1. **Full raw-data reconstruction.** This starts from the original pollutant,
   traffic, meteorological, and airport sources and rebuilds the merged hourly
   table. This level requires access to all original sources, including the
   non-public airport operation records.
2. **Computational reproduction from the merged dataset.** This starts from
   `Datasets_Raw/hourly_merged_2023_2025.csv` and reruns the modelling,
   ablation, interpretation, wind-regime, and cross-pollutant analyses. This is
   the recommended route for reproducing the results in the repository.
3. **Inspection-only reproduction.** This uses the existing CSV outputs under
   `Analysis/` and the generated plots to inspect all metrics, ablation deltas,
   SHAP summaries, wind contrasts, and paper figures without rerunning the full
   compute pipeline.

### Data Preparation Scripts

The following scripts are used to build intermediate data sources or the merged
hourly table:

- `merge_blq_traffic.py`
- `merge_hourly_datasets.py`
- `merge_meteo.py`
- `merge_porta_san_felice_pollutants.py`
- `merge_spire_flow.py`

These scripts are relevant when the original raw data are available. For most
analysis reruns, the required starting point is already the merged file
`Datasets_Raw/hourly_merged_2023_2025.csv`.

### Main Analysis Order

The analyses have dependencies. The recommended order is:

1. run the forecasting, ablation, SHAP, and model-comparison pipeline;
2. run the wind-regime and spatial-gradient analysis;
3. run the airport-response descriptive analysis;
4. run the cross-pollutant synthesis after the forecasting outputs exist;
5. generate the paper figures after the relevant analysis outputs exist.

The reason for this order is that `cross_pollutant_analysis.py` does not refit
models. It reads and reorganizes outputs produced by the `explain` and
`upwind/downwind` analyses. Therefore, it must be executed after the upstream
CSV files have been generated.

### Local Execution

From the repository root, after activating the conda environment, the main
scripts can be run locally as follows:

```bash
python explain_pollutants_by_feature_groups.py
python upwind_downwind_analysis.py
python airport_response_analysis.py
python cross_pollutant_analysis.py
python prepare_paper_figures.py
```

For a lighter local run or debugging session, it is advisable to run one
component at a time and inspect the corresponding output directory before moving
to the next step.

### SLURM Execution

For the full analyses, the repository includes SLURM scripts:

- `explain_pollutants.slurm`
- `upwind_downwind.slurm`
- `airport_response_analysis.slurm`
- `cross_pollutant_analysis.slurm`
- `prepare_paper_figures.slurm`

The standard submission commands are:

```bash
sbatch explain_pollutants.slurm
sbatch upwind_downwind.slurm
sbatch airport_response_analysis.slurm
sbatch cross_pollutant_analysis.slurm
sbatch prepare_paper_figures.slurm
```

Since the cross-pollutant analysis depends on the outputs of the explain
pipeline, the repository also provides:

- `submit_explain_then_cross.sh`

This script submits `cross_pollutant_analysis.slurm` after
`explain_pollutants.slurm` has completed successfully. It should be used when
the full workflow is run on a cluster and dependency ordering must be enforced.

To submit the available analyses together, the repository also includes:

- `submit_all_analyses.sh`

### Expected Output Directories

The main generated outputs are written under `Analysis/`:

- `Analysis/slurm_full_explain/`
- `Analysis/slurm_full_upwind/`
- `Analysis/airport_response_full/`
- `Analysis/cross_pollutant/`

Each directory contains CSV files with metrics and summaries, plus a `plots/`
subdirectory with figures. The README tables are summaries of these outputs, not
independent manually curated results.

### Determinism and Validation Design

The main forecasting analysis uses expanding-window temporal cross-validation.
Rows are interpreted as forecast origins, and labels are generated by shifting
the target by the forecast horizon. Lagged and rolling features are shifted so
that they use information available at or before the forecast origin. Missing
predictor values are imputed within each temporal fold using training-set
medians, and the same medians are applied to the corresponding test fold.

This design is intended to avoid temporal leakage. Because the evaluation is
time ordered, results may still vary slightly if model implementations, package
versions, or default numerical behaviour change. The `environment.yml` file is
therefore part of the reproducibility record.

### Regenerating Paper Figures

The paper figures are prepared from analysis outputs with:

```bash
python prepare_paper_figures.py
```

or, on SLURM:

```bash
sbatch prepare_paper_figures.slurm
```

The script expects the relevant analysis outputs to be present under
`Analysis/`. If those directories are missing, rerun the upstream analyses first.

### Cleaning and Rerunning

To reproduce a complete run from scratch, remove or archive the generated
analysis-output directories under `Analysis/`, keep the input dataset under
`Datasets_Raw/`, and rerun the analysis scripts in the order described above.
The datasets should not be deleted unless the raw-data merge step is also being
repeated.

### Result Inspection

The most important CSV files for verification are:

- `Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_delta_summary.csv`
- `Analysis/slurm_full_explain/advanced_group_shap.csv`
- `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_matched_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_bootstrap_effects.csv`
- `Analysis/cross_pollutant/cross_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_family_overview.csv`

These files are sufficient to verify the main claims about predictive
performance, ablation hierarchy, SHAP interpretation, and wind-regime contrasts.

## 1. Purpose of the Repository

The objective of this repository is not only to obtain accurate predictive
models, but also to understand **which type of information** makes pollutant
concentrations predictable in an urban context influenced by:

- urban traffic measured by loop detectors;
- BLQ airport activity;
- meteorology;
- temporal memory in the pollutant time series;
- multi-station and multi-pollutant context.

The distinction between forecasting and interpretation is central. A model may
predict a target accurately simply because it exploits the persistence of the
series. This is operationally useful, but it is not sufficient to support a
physical or source-specific interpretation. For this reason, the repository
separates four analytical layers:

1. single-target forecasting with and without target autoregression;
2. multi-target forecasting to assess whether shared information exists across
   stations and pollutants;
3. `upwind/downwind` analysis and spatial gradients to test the physical
   consistency of airport-related signals;
4. an explicit `cross_pollutant` synthesis to compare chemical families and
   targets in a structured manner.

## 2. Key Result of the Current Run

The most robust result of the current run is not simply which model obtains the
highest `R2`, but **which information blocks remain important when they are
systematically removed**.

In the extended single-target `xgboost` ablation, the strongest average
contributions are:

- `meteo`: `mean delta R2 = +0.087`
- `other_pollutants`: `+0.026`
- `other_pollutants_porta_san_felice`: `+0.021`
- `rolling_features`: `+0.017`

By contrast, the more detailed airport-related blocks have much smaller average
contributions:

- `airport`: `+0.001`
- `station_wind_bools`: `-0.001`
- `airport_service_type`: `-0.002`
- `airport_wind_interaction`: `-0.003`

The correct interpretation is therefore not that the airport is irrelevant, but
a more precise statement:

- the airport contribution exists;
- it does not dominate the whole target-horizon matrix on average;
- it emerges mainly in **selective cases**, rather than as a uniform driver;
- the average structure of the problem remains primarily governed by
  meteorology, multi-pollutant context, and recent aggregated temporal memory.

## 3. Current Repository Structure

The repository contains four distinct analytical components.

### 3.1. `explain` Component

Script:

- `explain_pollutants_by_feature_groups.py`

Purpose:

- multi-horizon single-target forecasting;
- model comparison;
- comparison between setups with and without target autoregression;
- targeted and extended ablations;
- group-level SHAP;
- native `XGBoost` feature importance;
- multioutput `XGBoost` comparison.

Main outputs:

- `Analysis/slurm_full_explain/advanced_temporal_cv_scores.csv`
- `Analysis/slurm_full_explain/advanced_temporal_cv_predictions.csv`
- `Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv`
- `Analysis/slurm_full_explain/advanced_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_feature_sets.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_scores.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_fold_deltas.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_delta_summary.csv`
- `Analysis/slurm_full_explain/advanced_group_shap.csv`
- `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances.csv`
- `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_scores.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_feature_sets.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_scores.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_fold_deltas.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_delta_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_group_shap.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances_summary.csv`
- `Analysis/slurm_full_explain/advanced_runtime_profile.csv`
- `Analysis/slurm_full_explain/pollutant_station_reference_stats.csv`
- `Analysis/slurm_full_explain/plots/`

### 3.2. `upwind/downwind` Component

Script:

- `upwind_downwind_analysis.py`

Purpose:

- classification into `downwind`, `upwind`, `crosswind`, and `calm`;
- descriptive comparisons across wind regimes;
- regressions with `BLQ x downwind` interactions;
- `downwind/upwind` matching;
- block bootstrap;
- threshold sensitivity analysis;
- spatial gradients and multi-station DID analysis;
- SHAP by wind regime.

Main outputs:

- `Analysis/slurm_full_upwind/upwind_downwind_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_blq_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_distributed_lag_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_regression_coefficients.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_distributed_lag_coefficients.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_matched_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_matched_pairs.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_bootstrap_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_threshold_sensitivity.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_classified_hours.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_blq_quantile_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_group_shap_by_regime.csv`
- `Analysis/slurm_full_upwind/multistation_did_summary.csv`
- `Analysis/slurm_full_upwind/multistation_did_regression_coefficients.csv`
- `Analysis/slurm_full_upwind/multistation_panel_long.csv`
- `Analysis/slurm_full_upwind/multistation_spatial_gradients.csv`
- `Analysis/slurm_full_upwind/multistation_spatial_gradient_summary.csv`
- `Analysis/slurm_full_upwind/multistation_station_wind_features.csv`
- `Analysis/slurm_full_upwind/plots/`

### 3.3. `airport_response` Component

Script:

- `airport_response_analysis.py`

Purpose:

- empirical target-versus-BLQ curves by regime;
- partial dependence profiles;
- event windows;
- exceedance probabilities;
- descriptive multi-station gradients.

Main outputs:

- `Analysis/airport_response_full/blq_empirical_response_curves.csv`
- `Analysis/airport_response_full/blq_partial_dependence_model_metrics.csv`
- `Analysis/airport_response_full/blq_partial_dependence_profiles.csv`
- `Analysis/airport_response_full/blq_event_windows_summary.csv`
- `Analysis/airport_response_full/blq_event_windows_long.csv`
- `Analysis/airport_response_full/blq_exceedance_probabilities.csv`
- `Analysis/airport_response_full/blq_spatial_gradient_response.csv`
- `Analysis/airport_response_full/plots/`

This component is **descriptive and explanatory**, not causal. Its purpose is to
make the relationship among BLQ activity, wind, pollutant targets, and urban
context more interpretable.

### 3.4. `cross_pollutant` Component

Script:

- `cross_pollutant_analysis.py`

Purpose:

- explicit comparison among targets and chemical families;
- synthesis of multi-horizon predictability;
- synthesis of dominant ablation groups;
- standardized synthesis of wind-regime contrasts.

Main outputs:

- `Analysis/cross_pollutant/cross_pollutant_predictability_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_predictability_target_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_predictability_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_ablation_group_matrix.csv`
- `Analysis/cross_pollutant/cross_pollutant_ablation_top_groups.csv`
- `Analysis/cross_pollutant/cross_pollutant_targeted_ablation_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_group_shap_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_wind_response_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_wind_response_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_family_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_runtime_profile.csv`
- `Analysis/cross_pollutant/plots/`

This fourth analysis does not refit the base models. It reorganizes and
summarizes the results already produced by the other components.

## 4. Dataset

File:

- `Datasets_Raw/hourly_merged_2023_2025.csv`

Characteristics:

- `9,792` hourly rows
- `61` columns
- time span: `2024-05-29 00:00:00` -> `2025-07-10 23:00:00`
- temporal key: `datetime`

This file represents the common temporal intersection among all data blocks used
in the analysis. This choice is methodologically appropriate because it avoids
training models on periods in which one of the main information blocks is
entirely unavailable.

### 4.1. Analysed Targets

The targets in the current run are:

- `NO2_porta_san_felice`
- `CO_porta_san_felice`
- `C6H6_porta_san_felice`
- `NO2_giardini_margherita`
- `NO2_via_chiarini`
- `O3_giardini_margherita`
- `O3_via_chiarini`

Descriptive statistics in the unified dataset:

| target                      | unit      |   minimum |       mean |     maximum |
| --------------------------- | --------- | --------: | ---------: | ----------: |
| `NO2_porta_san_felice`    | `ug/m3` | `2.000` | `26.586` |  `96.000` |
| `CO_porta_san_felice`     | `mg/m3` | `0.000` |  `0.468` |   `2.500` |
| `C6H6_porta_san_felice`   | `ug/m3` | `0.100` |  `0.961` |   `6.100` |
| `NO2_giardini_margherita` | `ug/m3` | `0.000` | `13.928` |  `63.000` |
| `NO2_via_chiarini`        | `ug/m3` | `0.000` | `15.560` |  `82.000` |
| `O3_giardini_margherita`  | `ug/m3` | `0.000` | `50.426` | `188.000` |
| `O3_via_chiarini`         | `ug/m3` | `0.000` | `45.810` | `213.000` |

Useful interpretation:

- `NO2` is available at three stations and is therefore the natural candidate
  for spatial comparisons;
- `O3` is available at the two external stations and is the most suitable target
  for studying meteorological and background dynamics;
- `CO` and `C6H6` are observed at Porta San Felice.

### 4.2. Information Blocks

The dataset integrates:

- BLQ airport traffic, including the `SERVICE_TYPE_CODE` decomposition;
- urban traffic from loop detectors, kept as separate columns;
- meteorology from two sources, `_aero` and `_centro`;
- other pollutants as multi-station and multi-pollutant context.

The loop-detector selection retains `20` unique sensors:

- `5` closest to BLQ;
- `5` closest to `Porta San Felice`;
- `5` closest to `Giardini Margherita`;
- `5` closest to `Via Chiarini`.

### 4.3. Temporal and Derived Features

The pipeline constructs:

- calendar features:
  - `hour`, `dayofweek`, `month`, `is_weekend`
  - `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
- lags:
  - `_lag_1h`, `_lag_2h`, `_lag_3h`, `_lag_6h`, `_lag_12h`, `_lag_24h`
- differences:
  - `_diff_1h`
- rolling means:
  - `_rolling_3h_mean`, `_rolling_6h_mean`, `_rolling_12h_mean`, `_rolling_24h_mean`
- rolling standard deviations:
  - `_rolling_3h_std`, `_rolling_6h_std`, `_rolling_12h_std`, `_rolling_24h_std`
- wind interactions relative to the airport-to-station geometry.

This is technically important because the problem is not a static tabular
regression task: the targets depend on lagged effects, recent accumulation, local
variability, and transport regimes.

## 5. Methods and Analytical Logic

### 5.1. Single-Target Forecasting

Question:

- how accurately can each target be predicted at `1h`, `3h`, `6h`, `12h`, and
  `24h`?

Design:

- `5`-fold expanding-window temporal validation;
- metrics: `R2`, `MAE`, `RMSE`, `MAPE`;
- compared models:
  - `ridge`
  - `decision_tree`
  - `random_forest`
  - `extra_trees`
  - `adaboost`
  - `xgbrf`
  - `xgboost`

The two views are:

- `no_target_*`: the model does not use the target's past values;
- `with_target_*`: the model also uses the target's past values.

The first view is more interpretative. The second is more predictive.

### 5.2. Multioutput Forecasting

Question:

- does shared context across stations and pollutants provide useful information?

Method:

- `MultiOutputRegressor(XGBoost)`

The purpose is not to replace the single-target setup, but to assess whether a
measurable gain appears when targets are predicted within a common multioutput
framework.

### 5.3. Ablations

Question:

- how much does the model lose when a coherent feature block is removed?

Levels used:

- targeted ablations on `service_type`, `station_wind_bools`, and their joint
  removal;
- extended ablation on broader groups such as `meteo`, `urban_traffic`,
  `other_pollutants`, `rolling_features`, `airport`, `airport_service_type`,
  and `wind_transport`.

Ablation answers a different question from SHAP:

- SHAP indicates what the model uses;
- ablation indicates how much predictive performance is lost when a block is
  absent.

### 5.4. SHAP and Native Feature Importance

Files used:

- `advanced_group_shap.csv`
- `advanced_multioutput_group_shap.csv`
- `advanced_xgboost_native_feature_importances_summary.csv`
- `advanced_multioutput_xgboost_native_feature_importances_summary.csv`

Purpose:

- to understand which groups or individual features drive the predictions;
- to distinguish the roles of autoregression, meteorology, traffic, airport
  activity, and multi-pollutant context.

### 5.5. `upwind/downwind` Analysis

Question:

- is the signal associated with BLQ consistent with a physical transport
  hypothesis toward the monitoring stations?

Method:

- classification of hours into `downwind`, `upwind`, `crosswind`, and `calm`;
- descriptive comparisons by wind regime;
- regressions with the terms:
  - `blq_activity`
  - `downwind_flag`
  - `upwind_flag`
  - `blq_x_downwind`
  - `blq_x_upwind`
- `downwind/upwind` matching;
- block bootstrap;
- threshold sensitivity analysis.

`downwind` means that the air mass moves from BLQ toward the station. It does
not simply mean "favourable wind".

### 5.6. Spatial Gradients

Question:

- is the BLQ-related signal more visible in absolute levels or in gradients
  between stations?

Method:

- construction of gradients such as:
  - `NO2_psf_minus_chiarini`
  - `NO2_psf_minus_giardini`
  - `O3_chiarini_minus_giardini`
- multi-station DID regressions on the gradients.

### 5.7. `airport_response` Analysis

Question:

- are there empirically readable patterns between BLQ activity, wind, and
  pollutant targets that are more intuitive than regression coefficients alone?

This component includes:

- target-versus-BLQ curves by regime;
- partial dependence;
- event windows;
- high-threshold exceedance;
- descriptive gradients by BLQ class.

It is an **explanatory** component, not a causal demonstration.

### 5.8. `cross_pollutant` Analysis

Question:

- in what respects are `NO2`, `CO`, `C6H6`, and `O3` empirically similar, and in
  what respects do they differ?

This component re-aggregates previous outputs along three axes:

- predictability;
- dependence on feature groups;
- response to wind regimes.

## 6. Predictive Results: View Without Target History

This is the most important section if the objective is to understand **how much
external information** is available.

The following table reports, for each target and horizon, the **best single-target
setup without autoregression**. It therefore does not contain all 1,960 raw
results, but the best model and feature-set combination for each case.

| target                      |      h | model           | feature set                                              |         R2 |        MAE |       RMSE |       MAPE |
| --------------------------- | -----: | --------------- | -------------------------------------------------------- | ---------: | ---------: | ---------: | ---------: |
| `C6H6_porta_san_felice`   |  `1` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.578` |  `0.231` |  `0.320` |  `25.51` |
| `C6H6_porta_san_felice`   |  `3` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.401` |  `0.270` |  `0.379` |  `30.48` |
| `C6H6_porta_san_felice`   |  `6` | `xgboost`     | `no_target_without_station_wind_bools`                 |  `0.305` |  `0.299` |  `0.409` |  `34.73` |
| `C6H6_porta_san_felice`   | `12` | `xgboost`     | `no_target_without_station_wind_bools`                 |  `0.299` |  `0.303` |  `0.411` |  `36.43` |
| `C6H6_porta_san_felice`   | `24` | `xgboost`     | `no_target_without_service_type`                       |  `0.179` |  `0.331` |  `0.445` |  `38.85` |
| `CO_porta_san_felice`     |  `1` | `xgboost`     | `no_target_autoregressive`                             |  `0.417` |  `0.111` |  `0.142` |  `23.32` |
| `CO_porta_san_felice`     |  `3` | `xgboost`     | `no_target_without_service_type`                       |  `0.272` |  `0.124` |  `0.162` |  `25.67` |
| `CO_porta_san_felice`     |  `6` | `xgboost`     | `no_target_without_service_type`                       |  `0.199` |  `0.132` |  `0.171` |  `27.38` |
| `CO_porta_san_felice`     | `12` | `xgboost`     | `no_target_autoregressive`                             |  `0.147` |  `0.141` |  `0.180` |  `29.43` |
| `CO_porta_san_felice`     | `24` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.087` |  `0.143` |  `0.184` |  `30.15` |
| `NO2_giardini_margherita` |  `1` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.526` |  `3.781` |  `5.104` |  `38.41` |
| `NO2_giardini_margherita` |  `3` | `xgboost`     | `no_target_without_service_type`                       |  `0.239` |  `4.940` |  `6.592` |  `53.61` |
| `NO2_giardini_margherita` |  `6` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.116` |  `5.471` |  `7.094` |  `66.10` |
| `NO2_giardini_margherita` | `12` | `xgboost`     | `no_target_without_service_type`                       |  `0.023` |  `5.681` |  `7.314` |  `73.28` |
| `NO2_giardini_margherita` | `24` | `extra_trees` | `no_target_without_service_type_or_station_wind_bools` | `-0.027` |  `5.921` |  `7.570` |  `82.94` |
| `NO2_porta_san_felice`    |  `1` | `xgboost`     | `no_target_without_service_type`                       |  `0.206` |  `8.519` | `10.841` |  `34.61` |
| `NO2_porta_san_felice`    |  `3` | `xgboost`     | `no_target_without_service_type`                       |  `0.070` |  `9.254` | `11.723` |  `38.43` |
| `NO2_porta_san_felice`    |  `6` | `xgboost`     | `no_target_without_station_wind_bools`                 | `-0.007` |  `9.689` | `12.140` |  `40.91` |
| `NO2_porta_san_felice`    | `12` | `adaboost`    | `no_target_autoregressive`                             | `-0.062` | `10.086` | `12.495` |  `45.41` |
| `NO2_porta_san_felice`    | `24` | `xgboost`     | `no_target_autoregressive`                             | `-0.003` |  `9.523` | `12.025` |  `42.71` |
| `NO2_via_chiarini`        |  `1` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.293` |  `5.717` |  `7.974` |  `30.33` |
| `NO2_via_chiarini`        |  `3` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.123` |  `6.567` |  `8.895` |  `37.28` |
| `NO2_via_chiarini`        |  `6` | `xgboost`     | `no_target_without_service_type`                       |  `0.057` |  `6.868` |  `9.167` |  `41.64` |
| `NO2_via_chiarini`        | `12` | `xgboost`     | `no_target_without_service_type`                       |  `0.059` |  `6.918` |  `9.194` |  `42.57` |
| `NO2_via_chiarini`        | `24` | `xgboost`     | `no_target_autoregressive`                             |  `0.023` |  `6.989` |  `9.372` |  `43.79` |
| `O3_giardini_margherita`  |  `1` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.813` |  `7.061` |  `9.210` |  `44.02` |
| `O3_giardini_margherita`  |  `3` | `xgboost`     | `no_target_without_service_type`                       |  `0.614` | `10.045` | `13.053` |  `66.62` |
| `O3_giardini_margherita`  |  `6` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.427` | `12.330` | `15.882` |  `82.95` |
| `O3_giardini_margherita`  | `12` | `xgboost`     | `no_target_without_station_wind_bools`                 |  `0.274` | `14.041` | `17.720` | `112.20` |
| `O3_giardini_margherita`  | `24` | `adaboost`    | `no_target_autoregressive`                             |  `0.173` | `16.052` | `19.251` | `147.65` |
| `O3_via_chiarini`         |  `1` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.766` |  `8.906` | `11.421` |  `46.94` |
| `O3_via_chiarini`         |  `3` | `xgboost`     | `no_target_without_service_type_or_station_wind_bools` |  `0.623` | `11.171` | `14.580` |  `65.55` |
| `O3_via_chiarini`         |  `6` | `xgboost`     | `no_target_without_service_type`                       |  `0.439` | `13.707` | `17.761` |  `84.37` |
| `O3_via_chiarini`         | `12` | `xgboost`     | `no_target_without_station_wind_bools`                 |  `0.422` | `14.250` | `18.067` | `102.97` |
| `O3_via_chiarini`         | `24` | `xgboost`     | `no_target_without_service_type`                       |  `0.342` | `15.414` | `19.417` | `119.03` |

Main interpretation:

- `O3` and `C6H6` are the most readable targets from external features;
- `CO` is intermediate;
- `NO2`, especially at Porta San Felice, remains difficult without target
  history;
- `ExtraTrees` and `AdaBoost` emerge as best models only in a few specific
  cases, mainly at long horizons.

## 7. Predictive Results: View With Target History

This view measures the operational predictive potential when target persistence
is made available to the model.

| target                      |      h | model           | feature set                                                |        R2 |        MAE |       RMSE |       MAPE |
| --------------------------- | -----: | --------------- | ---------------------------------------------------------- | --------: | ---------: | ---------: | ---------: |
| `C6H6_porta_san_felice`   |  `1` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.677` |  `0.190` |  `0.279` |  `19.83` |
| `C6H6_porta_san_felice`   |  `3` | `xgboost`     | `with_target_autoregressive`                             | `0.448` |  `0.261` |  `0.368` |  `28.96` |
| `C6H6_porta_san_felice`   |  `6` | `xgboost`     | `with_target_autoregressive`                             | `0.336` |  `0.291` |  `0.401` |  `33.74` |
| `C6H6_porta_san_felice`   | `12` | `xgboost`     | `with_target_autoregressive`                             | `0.310` |  `0.299` |  `0.407` |  `35.65` |
| `C6H6_porta_san_felice`   | `24` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.221` |  `0.320` |  `0.435` |  `36.86` |
| `CO_porta_san_felice`     |  `1` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.717` |  `0.066` |  `0.095` |  `12.87` |
| `CO_porta_san_felice`     |  `3` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.505` |  `0.099` |  `0.134` |  `19.60` |
| `CO_porta_san_felice`     |  `6` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.419` |  `0.111` |  `0.146` |  `22.26` |
| `CO_porta_san_felice`     | `12` | `xgboost`     | `with_target_without_service_type`                       | `0.361` |  `0.116` |  `0.154` |  `23.03` |
| `CO_porta_san_felice`     | `24` | `xgboost`     | `with_target_autoregressive`                             | `0.320` |  `0.120` |  `0.159` |  `23.89` |
| `NO2_giardini_margherita` |  `1` | `xgboost`     | `with_target_autoregressive`                             | `0.781` |  `2.422` |  `3.561` |  `21.47` |
| `NO2_giardini_margherita` |  `3` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.369` |  `4.457` |  `6.119` |  `44.25` |
| `NO2_giardini_margherita` |  `6` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.166` |  `5.316` |  `6.954` |  `61.81` |
| `NO2_giardini_margherita` | `12` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.089` |  `5.541` |  `7.154` |  `70.99` |
| `NO2_giardini_margherita` | `24` | `extra_trees` | `with_target_autoregressive`                             | `0.004` |  `5.840` |  `7.491` |  `81.72` |
| `NO2_porta_san_felice`    |  `1` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.788` |  `4.192` |  `5.723` |  `15.42` |
| `NO2_porta_san_felice`    |  `3` | `xgboost`     | `with_target_autoregressive`                             | `0.503` |  `6.644` |  `8.706` |  `26.49` |
| `NO2_porta_san_felice`    |  `6` | `ridge`       | `with_target_without_service_type_or_station_wind_bools` | `0.342` |  `7.632` |  `9.810` |  `34.04` |
| `NO2_porta_san_felice`    | `12` | `xgboost`     | `with_target_autoregressive`                             | `0.325` |  `7.770` |  `9.928` |  `33.70` |
| `NO2_porta_san_felice`    | `24` | `xgboost`     | `with_target_without_service_type`                       | `0.368` |  `7.596` |  `9.720` |  `34.12` |
| `NO2_via_chiarini`        |  `1` | `xgboost`     | `with_target_without_service_type`                       | `0.748` |  `3.210` |  `4.761` |  `18.06` |
| `NO2_via_chiarini`        |  `3` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.381` |  `5.406` |  `7.451` |  `32.82` |
| `NO2_via_chiarini`        |  `6` | `xgboost`     | `with_target_autoregressive`                             | `0.236` |  `6.141` |  `8.229` |  `39.96` |
| `NO2_via_chiarini`        | `12` | `xgboost`     | `with_target_without_service_type`                       | `0.237` |  `6.151` |  `8.229` |  `40.46` |
| `NO2_via_chiarini`        | `24` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.210` |  `6.266` |  `8.395` |  `40.61` |
| `O3_giardini_margherita`  |  `1` | `xgboost`     | `with_target_autoregressive`                             | `0.915` |  `4.407` |  `6.179` |  `26.44` |
| `O3_giardini_margherita`  |  `3` | `xgboost`     | `with_target_without_service_type`                       | `0.684` |  `9.004` | `11.807` |  `59.25` |
| `O3_giardini_margherita`  |  `6` | `xgboost`     | `with_target_without_service_type`                       | `0.465` | `11.991` | `15.339` |  `83.74` |
| `O3_giardini_margherita`  | `12` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.329` | `13.682` | `17.339` | `102.28` |
| `O3_giardini_margherita`  | `24` | `extra_trees` | `with_target_without_service_type_or_station_wind_bools` | `0.220` | `15.059` | `18.585` | `129.57` |
| `O3_via_chiarini`         |  `1` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.912` |  `5.085` |  `7.069` |  `27.58` |
| `O3_via_chiarini`         |  `3` | `xgboost`     | `with_target_without_service_type`                       | `0.668` | `10.385` | `13.633` |  `61.36` |
| `O3_via_chiarini`         |  `6` | `xgboost`     | `with_target_without_service_type_or_station_wind_bools` | `0.469` | `13.239` | `17.153` |  `86.63` |
| `O3_via_chiarini`         | `12` | `xgboost`     | `with_target_without_station_wind_bools`                 | `0.436` | `13.762` | `17.593` | `107.09` |
| `O3_via_chiarini`         | `24` | `extra_trees` | `with_target_without_service_type_or_station_wind_bools` | `0.352` | `15.186` | `19.033` | `121.93` |

Main interpretation:

- the largest autoregressive gains are observed for `NO2` and `CO`;
- `C6H6` improves, but much less than `NO2`;
- `O3` is already strong without target history and remains strong with target
  history;
- at long horizons, `ExtraTrees` is competitive for some targets,
  especially `O3_giardini_margherita`, `O3_via_chiarini`, and
  `NO2_giardini_margherita`.

## 8. Model Comparison

The correct answer is no longer simply "`XGBoost` always wins", but a more
precise formulation:

- `XGBoost` remains the dominant model overall;
- its dominance is clearer at short horizons;
- at `24h`, `ExtraTrees` appears more frequently, and `AdaBoost` appears in one
  case.

Concrete exceptions:

- `NO2_giardini_margherita` at `24h`: best model `ExtraTrees`, `R2 = 0.004`;
- `O3_giardini_margherita` at `24h`: best model `ExtraTrees`, `R2 = 0.220`;
- `O3_via_chiarini` at `24h`: best model `ExtraTrees`, `R2 = 0.352`;
- `NO2_porta_san_felice` at `12h` without autoregression: best model `AdaBoost`,
  `R2 = -0.062`, still within a difficult region for all models.

Interpretation:

- the problem is clearly nonlinear;
- tree ensembles are the appropriate model family;
- `XGBoost` remains the main reference model;
- `ExtraTrees` is the most credible challenger for more regular or noisier
  long-horizon targets.

## 9. Single-Target Versus Multioutput

The multioutput setup does not replace the single-target setup as the best
general baseline, but it produces some measurable improvements.

Cases in which multioutput outperforms the best single-target result:

| target                      |      h | multioutput feature set                                               | multioutput R2 | best single-target R2 |      delta |
| --------------------------- | -----: | --------------------------------------------------------------------- | -------------: | --------------------: | ---------: |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_service_type_or_station_wind_bools`   |      `0.070` |             `0.004` | `+0.066` |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_station_wind_bools`                   |      `0.051` |             `0.004` | `+0.047` |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_service_type`                         |      `0.048` |             `0.004` | `+0.044` |
| `C6H6_porta_san_felice`   | `24` | `with_pollutant_context_without_station_wind_bools`                 |      `0.250` |             `0.221` | `+0.029` |
| `NO2_giardini_margherita` | `12` | `with_pollutant_context`                                            |      `0.118` |             `0.089` | `+0.028` |
| `O3_via_chiarini`         | `24` | `with_pollutant_context_without_service_type_or_station_wind_bools` |      `0.371` |             `0.352` | `+0.019` |
| `C6H6_porta_san_felice`   |  `3` | `with_pollutant_context`                                            |      `0.465` |             `0.448` | `+0.017` |
| `NO2_via_chiarini`        |  `6` | `with_pollutant_context_without_service_type`                       |      `0.247` |             `0.236` | `+0.011` |

Conclusion:

- the multi-station and multi-pollutant context contains substantive predictive
  information;
- the advantage is not uniform;
- the clearest gains are observed for `NO2_giardini_margherita`,
  `C6H6_porta_san_felice`, and `O3_via_chiarini`.

## 10. Ablations: Dominant Information Blocks

### 10.1. Average Group Hierarchy

Extended single-target `xgboost` ablation, averaged over targets and horizons:

| removed group                            | mean delta R2 | interpretation                            |
| ---------------------------------------- | ------------: | ----------------------------------------- |
| `meteo`                                |    `+0.087` | strongest average contribution            |
| `other_pollutants`                     |    `+0.026` | second most important average block       |
| `other_pollutants_porta_san_felice`    |    `+0.021` | strong local context at PSF               |
| `rolling_features`                     |    `+0.017` | highly useful recent aggregated memory    |
| `other_pollutants_giardini_margherita` |    `+0.006` | positive but secondary local contribution |
| `urban_traffic`                        |    `+0.004` | positive but limited average contribution |
| `diff_features`                        |    `+0.003` | small but real contribution               |
| `lag_features`                         |    `+0.002` | weaker than rolling features              |
| `wind_transport`                       |    `+0.002` | small average contribution                |
| `airport`                              |    `+0.001` | positive but weak on average              |
| `station_wind_bools`                   |    `-0.001` | nearly null on average                    |
| `airport_service_type`                 |    `-0.002` | selective, not uniform                    |
| `airport_wind_interaction`             |    `-0.003` | selective, not uniform                    |

### 10.2. Where the Airport Block Provides Predictive Information

The aggregate `airport` block shows the clearest gains mainly for:

- `O3_giardini_margherita` at `24h`: `delta R2 = +0.074`
- `C6H6_porta_san_felice` at `24h`: `+0.047`
- `O3_giardini_margherita` at `12h`: `+0.039`
- `NO2_porta_san_felice` at `24h`: `+0.027`

The `airport_service_type` decomposition emerges mainly for:

- `C6H6_porta_san_felice` at `24h`: `+0.043`
- `NO2_porta_san_felice` at `12h`: `+0.037`
- `C6H6_porta_san_felice` at `12h`: `+0.026`
- `NO2_porta_san_felice` at `1h`: `+0.023`
- `NO2_porta_san_felice` at `6h`: `+0.021`
- `CO_porta_san_felice` at `3h`: `+0.016`

The `station_wind_bools` block shows the clearest cases for:

- `CO_porta_san_felice` at `24h`: `+0.081`
- `NO2_giardini_margherita` at `24h`: `+0.053`
- `CO_porta_san_felice` at `12h`: `+0.030`

The joint removal of `service_type + station_wind_bools` concentrates the most
readable losses on:

- `CO_porta_san_felice` at `24h`: `+0.093`
- `NO2_giardini_margherita` at `24h`: `+0.052 / +0.053`
- `NO2_porta_san_felice` at `12h`: `+0.031`
- `NO2_porta_san_felice` at `1h-3h` without autoregression: `+0.028`, `+0.028`

### 10.3. Correct Interpretation of the Ablation

Three points follow:

1. the refined airport-related blocks are not purely redundant;
2. their contribution is selective, not structurally dominant on average;
3. the robust structure of the problem remains governed by:
   - `meteo`
   - `other_pollutants`
   - `rolling_features`

## 11. SHAP and Feature Importance

### 11.1. Group-Level Interpretation

Group-level SHAP confirms the structure already observed in the ablations.

| target                      | group 1                      | group 2                           | group 3                                      | interpretation                             |
| --------------------------- | ---------------------------- | --------------------------------- | -------------------------------------------- | ------------------------------------------ |
| `NO2_porta_san_felice`    | `rolling_features (12.62)` | `target_autoregressive (7.70)`  | `meteo (7.41)`                             | recent memory dominates                    |
| `CO_porta_san_felice`     | `rolling_features (0.191)` | `target_autoregressive (0.154)` | `meteo (0.101)`                            | persistence plus meteorology               |
| `C6H6_porta_san_felice`   | `rolling_features (0.448)` | `meteo (0.281)`                 | `lag_features / other_pollutants (~0.223)` | target more readable from external factors |
| `NO2_giardini_margherita` | `rolling_features (7.65)`  | `meteo (6.88)`                  | `target_autoregressive (3.23)`             | meteorology and memory jointly matter      |
| `NO2_via_chiarini`        | `rolling_features (10.39)` | `target_autoregressive (6.57)`  | `meteo (5.94)`                             | recent dynamics are very strong            |
| `O3_giardini_margherita`  | `rolling_features (31.92)` | `meteo (29.55)`                 | `target_autoregressive (23.27)`            | highly regular structure                   |
| `O3_via_chiarini`         | `rolling_features (33.66)` | `meteo (33.01)`                 | `target_autoregressive (22.65)`            | meteorology and rolling features dominate  |

### 11.2. What `service_type` Adds

The `blq_service_*` features appear in the importance outputs:

- for `NO2_porta_san_felice`, cargo, mail, and combined services emerge most
  clearly;
- for `NO2_via_chiarini`, cargo, combined, and charter services emerge;
- for `NO2_giardini_margherita`, cargo, charter, scheduled, and mail services
  appear;
- for `O3`, charter and cargo often enter;
- for `CO` and `C6H6`, the effect exists but remains smaller.

### 11.3. Final Interpretation of the Model-Interpretation Component

- SHAP confirms that the new airport-related blocks are not artificial;
- ablation clarifies that their impact is selective;
- read correctly, the repository does not support a simple narrative in which
  "BLQ dominates the system".

## 12. `upwind/downwind` Results

### 12.1. Descriptive `downwind - upwind` Contrast

| target                      | unit      | downwind mean | upwind mean | downwind - upwind |
| --------------------------- | --------- | ------------: | ----------: | ----------------: |
| `NO2_porta_san_felice`    | `ug/m3` |     `28.75` |   `29.65` |         `-0.90` |
| `CO_porta_san_felice`     | `mg/m3` |     `0.554` |   `0.468` |        `+0.086` |
| `C6H6_porta_san_felice`   | `ug/m3` |     `1.135` |   `0.973` |        `+0.163` |
| `NO2_giardini_margherita` | `ug/m3` |     `17.37` |   `14.27` |         `+3.10` |
| `NO2_via_chiarini`        | `ug/m3` |     `17.26` |   `16.91` |         `+0.35` |
| `O3_giardini_margherita`  | `ug/m3` |     `38.22` |   `55.82` |        `-17.61` |
| `O3_via_chiarini`         | `ug/m3` |     `34.11` |   `55.57` |        `-21.46` |

First message:

- `CO` and `C6H6` at PSF are higher under downwind conditions;
- `NO2_porta_san_felice` is not;
- `NO2_giardini_margherita` is higher under downwind conditions;
- `O3` at the external stations moves in the opposite direction.

### 12.2. `downwind/upwind` Matching

| target                      | mean diff downwind - upwind |    p-value | interpretation                           |
| --------------------------- | --------------------------: | ---------: | ---------------------------------------- |
| `NO2_porta_san_felice`    |                   `-1.33` | `0.0030` | lower under downwind even after matching |
| `CO_porta_san_felice`     |                 `+0.0186` | `0.0122` | small but robust positive signal         |
| `C6H6_porta_san_felice`   |                 `+0.0152` | `0.3870` | non-robust difference                    |
| `NO2_giardini_margherita` |                  `+0.843` | `0.0024` | robust positive signal                   |
| `NO2_via_chiarini`        |                  `-0.625` | `0.0555` | ambiguous                                |
| `O3_giardini_margherita`  |                   `-5.66` | `<0.001` | much lower under downwind                |
| `O3_via_chiarini`         |                   `-8.98` | `<0.001` | much lower under downwind                |

### 12.3. Block Bootstrap

| target                      | mean effect downwind - upwind | CI95              | interpretation                |
| --------------------------- | ----------------------------: | ----------------- | ----------------------------- |
| `NO2_porta_san_felice`    |                     `-1.03` | crosses `0`     | negative sign but not robust  |
| `CO_porta_san_felice`     |                    `+0.087` | entirely positive | stable positive signal        |
| `C6H6_porta_san_felice`   |                    `+0.164` | entirely positive | fairly stable positive signal |
| `NO2_giardini_margherita` |                     `+3.08` | entirely positive | clear positive signal         |
| `NO2_via_chiarini`        |                     `+0.36` | crosses `0`     | weak/uncertain effect         |
| `O3_giardini_margherita`  |                    `-17.72` | entirely negative | very strong negative signal   |
| `O3_via_chiarini`         |                    `-21.52` | entirely negative | very strong negative signal   |

### 12.4. `high_downwind - low_downwind`

| target                      | effect high_downwind - low_downwind | interpretation                                         |
| --------------------------- | ----------------------------------: | ------------------------------------------------------ |
| `NO2_porta_san_felice`    |                           `+2.72` | increases during downwind hours with high BLQ activity |
| `CO_porta_san_felice`     |                          `-0.136` | opposite sign to a simple monotonic relationship       |
| `C6H6_porta_san_felice`   |                          `-0.174` | opposite sign to a simple monotonic relationship       |
| `NO2_giardini_margherita` |                           `-3.17` | opposite sign                                          |
| `NO2_via_chiarini`        |                           `-6.22` | opposite sign                                          |
| `O3_giardini_margherita`  |                          `+34.15` | very large increase                                    |
| `O3_via_chiarini`         |                          `+41.15` | very large increase                                    |

This table is important because it shows that BLQ activity, local chemistry,
atmospheric mixing, and traffic do not reduce to a simple monotonic relationship.

### 12.5. Threshold Sensitivity

| target                      | diff @0.30 | diff @0.50 | diff @0.70 | diff @0.85 | interpretation  |
| --------------------------- | ---------: | ---------: | ---------: | ---------: | --------------- |
| `NO2_porta_san_felice`    |  `-1.16` |  `-0.90` |  `-0.93` |  `-0.73` | always negative |
| `CO_porta_san_felice`     | `+0.076` | `+0.086` | `+0.083` | `+0.044` | always positive |
| `C6H6_porta_san_felice`   | `+0.138` | `+0.163` | `+0.153` | `+0.096` | always positive |
| `NO2_giardini_margherita` |  `+2.89` |  `+3.10` |  `+3.43` |  `+2.59` | always positive |
| `NO2_via_chiarini`        |  `+0.50` |  `+0.35` |  `-0.54` |  `-3.22` | sign changes    |
| `O3_giardini_margherita`  | `-18.44` | `-17.61` | `-15.23` |  `-6.60` | always negative |
| `O3_via_chiarini`         | `-22.94` | `-21.46` | `-17.30` |  `-5.48` | always negative |

### 12.6. `BLQ x downwind` Regressions

Key message:

- the `blq_x_downwind` term does not produce a simple and coherent signature
  across all targets;
- `NO2_porta_san_felice` does not support the simplest monotonic narrative;
- `O3` at the external stations shows robust but opposite-signed patterns.

### 12.7. Spatial Gradients

The DID gradients show that:

- spatial differences related to wind regime do exist;
- they do not all align with a monotonic narrative such as "closer to the
  airport trajectory implies higher concentrations";
- the real system depends on station, pollutant, and forecast horizon.

### 12.8. Overall Interpretation of the Physical Analysis

- `NO2_porta_san_felice`: does not show the expected simple airport pattern;
- `CO_porta_san_felice`: small but stable positive downwind signal;
- `C6H6_porta_san_felice`: positive descriptive signal, less convincing after
  matching;
- `NO2_giardini_margherita`: one of the clearest cases in favour of a positive
  downwind effect;
- `NO2_via_chiarini`: ambiguous;
- `O3`: robustly negative downwind pattern at the external stations.

Prudent physical conclusion:

- no unique and generalized airport signature emerges;
- some signals are compatible with a role of BLQ activity and wind;
- the effect varies substantially across pollutants and stations;
- a monocausal explanation is not supported.

## 13. Explicit Cross-Pollutant Comparison

The new `cross_pollutant` analysis organizes into a single synthesis what was
previously distributed across forecasting, ablations, and wind contrasts.

### 13.1. Target-Level Synthesis

| target                      | mean autoregressive gain R2 | top group 1          |     delta | top group 2          |     delta | downwind-upwind std units | matched std units | bootstrap std units | stable threshold |
| --------------------------- | --------------------------: | -------------------- | --------: | -------------------- | --------: | ------------------------: | ----------------: | ------------------: | ---------------: |
| `C6H6_porta_san_felice`   |                   `0.046` | `other_pollutants` | `0.096` | `meteo`            | `0.054` |                 `0.247` |         `0.023` |           `0.248` |            `1` |
| `CO_porta_san_felice`     |                   `0.240` | `meteo`            | `0.261` | `other_pollutants` | `0.100` |                 `0.296` |         `0.064` |           `0.301` |            `1` |
| `NO2_giardini_margherita` |                   `0.106` | `meteo`            | `0.192` | `urban_traffic`    | `0.010` |                 `0.323` |         `0.088` |           `0.321` |            `1` |
| `NO2_porta_san_felice`    |                   `0.424` | `other_pollutants` | `0.095` | `rolling_features` | `0.056` |                `-0.066` |        `-0.098` |          `-0.076` |            `1` |
| `NO2_via_chiarini`        |                   `0.251` | `meteo`            | `0.131` | `other_pollutants` | `0.081` |                 `0.034` |        `-0.060` |           `0.035` |            `0` |
| `O3_giardini_margherita`  |                   `0.062` | `meteo`            | `0.272` | `rolling_features` | `0.070` |                `-0.471` |        `-0.151` |          `-0.474` |            `1` |
| `O3_via_chiarini`         |                   `0.049` | `meteo`            | `0.090` | `other_pollutants` | `0.052` |                `-0.548` |        `-0.229` |          `-0.549` |            `1` |

### 13.2. Chemical-Family Synthesis

| pollutant | best no-auto R2 1h | best no-auto R2 24h | best with-auto R2 1h | best with-auto R2 24h | mean autoregressive gain | raw downwind-upwind std units | matched std units | bootstrap std units | high_downwind-low_downwind std units | stable threshold | top groups                                            |
| --------- | -----------------: | ------------------: | -------------------: | --------------------: | -----------------------: | ----------------------------: | ----------------: | ------------------: | -----------------------------------: | ---------------: | ----------------------------------------------------- |
| `C6H6`  |          `0.578` |           `0.179` |            `0.677` |             `0.221` |                `0.046` |                     `0.247` |         `0.023` |           `0.248` |                           `-0.264` |         `1.00` | `other_pollutants`, `meteo`, `rolling_features` |
| `CO`    |          `0.417` |           `0.087` |            `0.717` |             `0.320` |                `0.240` |                     `0.296` |         `0.064` |           `0.301` |                           `-0.466` |         `1.00` | `meteo`, `other_pollutants`, `rolling_features` |
| `NO2`   |          `0.342` |          `-0.002` |            `0.772` |             `0.194` |                `0.261` |                     `0.097` |        `-0.023` |           `0.093` |                           `-0.242` |         `0.67` | `meteo`, `other_pollutants`, `rolling_features` |
| `O3`    |          `0.790` |           `0.258` |            `0.914` |             `0.286` |                `0.056` |                    `-0.510` |        `-0.190` |          `-0.512` |                            `0.982` |         `1.00` | `meteo`, `other_pollutants`, `other_pollutants` |

### 13.3. Final Interpretation of the Comparison

- `NO2` is the most autoregressive family;
- `CO` is intermediate but remains strongly dependent on target history;
- `C6H6` is the target most readable from external factors at Porta San Felice;
- `O3` is the most meteorological and regular target, and the most robustly
  distinguishable in standardized wind-regime contrasts.

## 14. What Can Be Supported by the Results

### 14.1. Supported Statements

- external features contain substantive predictive signal;
- the value of that signal varies substantially across targets;
- `NO2` is strongly driven by persistence;
- `CO` has an intermediate behaviour;
- `C6H6` is more readable from external factors;
- `O3` is highly regular and strongly meteorological;
- the multi-station context contains substantive predictive information;
- `service_type` adds predictive signal in selective cases;
- wind and airport-to-station geometry explain part of the picture, but not
  uniformly.

### 14.2. Unsupported Statements

- strong causality;
- clean attribution of source dominance;
- a simple narrative such as "BLQ always increases the target under downwind
  conditions";
- a unique airport signature valid for all pollutants.

## 15. Interpretative Limits

The main limitations are:

- predictive performance does not imply causality;
- autoregression and interpretation are in tension;
- traffic, meteorology, calendar variables, and other pollutants are correlated;
- multioutput forecasting demonstrates informational dependence, not causality
  among stations;
- `airport_response` remains explanatory, not inferential;
- `upwind/downwind` also remains observational, not experimental.

## 16. Current Conclusion

The repository now supports a more coherent and precise narrative than the one
that was sustainable in the initial versions.

1. The system contains substantive predictive signal from traffic, meteorology, airport
   activity, and multi-station context.
2. The relative weight of these blocks varies substantially across targets.
3. The average structure of the problem is governed primarily by:
   - `meteo`
   - `other_pollutants`
   - `rolling_features`
4. Refined airport-related blocks add predictive information, but mainly in selective
   cases.
5. `NO2` is the most autoregressive family.
6. `CO` is intermediate.
7. `C6H6` is the target most readable from external variables at Porta San
   Felice.
8. `O3` is the most meteorological and regular target, and the most robustly
   distinguishable in wind-regime contrasts.

The most defensible conclusion is therefore neither that BLQ dominates the
system nor that BLQ is irrelevant, but a more cautious formulation:

- the airport is an informative component of the system;
- its effect is selective, not uniform;
- the average dynamics remain governed primarily by meteorology, dependencies
  among pollutants, and recent aggregated memory;
- no single simple airport signature emerges across all targets.

## 17. Location of the Complete Results

This file is a structured and complete synthesis of the most important results.
The full matrices, including all rows and all raw metrics, remain in the CSV
files.

To inspect **all results** without loss of detail:

- complete single-target metrics:
  - `Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv`
- out-of-sample predictions:
  - `Analysis/slurm_full_explain/advanced_temporal_cv_predictions.csv`
- complete multioutput metrics:
  - `Analysis/slurm_full_explain/advanced_multioutput_xgboost_summary.csv`
- targeted ablations:
  - `Analysis/slurm_full_explain/advanced_ablation_summary.csv`
- extended single-target ablations:
  - `Analysis/slurm_full_explain/advanced_extended_ablation_delta_summary.csv`
- extended multioutput ablations:
  - `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_delta_summary.csv`
- SHAP and importance:
  - `Analysis/slurm_full_explain/advanced_group_shap.csv`
  - `Analysis/slurm_full_explain/advanced_multioutput_group_shap.csv`
  - `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances_summary.csv`
  - `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances_summary.csv`
- `upwind/downwind` contrasts and regressions:
  - `Analysis/slurm_full_upwind/upwind_downwind_summary.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_blq_effects.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_matched_summary.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_bootstrap_effects.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_threshold_sensitivity.csv`
  - `Analysis/slurm_full_upwind/multistation_did_summary.csv`
- comparative syntheses:
  - `Analysis/cross_pollutant/cross_pollutant_overview.csv`
  - `Analysis/cross_pollutant/cross_pollutant_family_overview.csv`

This file and the CSV outputs, read together, cover both the descriptive
interpretation and the complete quantitative results.
