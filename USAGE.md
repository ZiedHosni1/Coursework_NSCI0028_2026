# Usage Guide

## 1) Run interactively

```powershell
py main.py
```

The CLI will prompt for:
- dataset
- target (if not fixed by profile)
- task type
- run profile (`fast` or `complete`)
- optional model keys

Dataset-specific preprocessing/featurization is applied automatically from `matpub/dataset_profiles.py` once the dataset name is selected.

## 2) Run non-interactively

```powershell
py main.py --non-interactive --dataset citrine_thermal_conductivity --target k_expt --run-mode complete
```

## 3) Choose approach (speed vs depth)

- Fast run:
```powershell
py main.py --non-interactive --dataset citrine_thermal_conductivity --target k_expt --run-mode fast
```

- Complete run:
```powershell
py main.py --non-interactive --dataset citrine_thermal_conductivity --target k_expt --run-mode complete
```

- Restrict to selected models:
```powershell
py main.py --non-interactive --dataset citrine_thermal_conductivity --target k_expt --run-mode complete --models "ridge,random_forest,svm,xgboost,catboost"
```

## 4) Useful switches

- List datasets:
```powershell
py main.py --list-datasets
```

- Override profile column handling:
```powershell
py main.py --dataset citrine_thermal_conductivity --target k_expt --include-columns "formula" --drop-columns "material_id"
```

- Disable strict integrity assertions (not recommended for publication runs):
```powershell
py main.py --dataset citrine_thermal_conductivity --target k_expt --disable-strict-checks
```

## 5) Output artifacts

Each run writes to a timestamped folder under `outputs/`, including:
- data integrity CSVs
- dataset EDA figures (pairplot, PCA, t-SNE, correlation, missingness)
- per-model diagnostics (residuals, calibration, uncertainty, DoA, SHAP, CV score distribution)
- global comparison figures (performance, runtime, CV with error bars, metric-vs-runtime)
- `report.html`
