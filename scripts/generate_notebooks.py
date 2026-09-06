"""
Generator for production-quality Jupyter Notebooks:
- 01_EDA.ipynb
- 02_Feature_Engineering.ipynb
- 03_Model_Benchmarking.ipynb
- 04_SHAP_Explainability.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def build_eda_notebook():
    cells = [
        md_cell("""# 01. Exploratory Data Analysis (EDA) — Turbofan Degradation Telemetry
### Industrial Predictive Maintenance Portfolio Project | NASA C-MAPSS Dataset

---

## 1. Executive Summary & Purpose
This notebook conducts a rigorous, publication-grade Exploratory Data Analysis (EDA) on the NASA **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** turbofan engine degradation dataset.

### Key Objectives:
- Ingest raw space-delimited telemetry files using the modular `src.data_loader` package.
- Characterize run-to-failure lifespans across engine units.
- Identify and isolate near-zero variance sensors to prevent overfitting.
- Analyze sensor cross-correlation to detect collinear physical phenomena.
- Characterize operational setting regimes across complex multi-condition subsets (FD002/FD004).

### Theoretical Rationale & Literature Reference:
As established by **Saxena et al. (2008)** in *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"* (PHM'08), simulated turbofan engines degrade from an initial healthy state to an unserviceable condition under single or multiple fault modes (High-Pressure Compressor degradation, Fan degradation). Detecting degradation requires isolating informative sensor responses from static atmospheric baselines.
"""),
        code_cell("""import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset, load_rul, COLUMNS, SENSOR_COLUMNS
from src.preprocessor import add_rul_labels, FD001_LOW_VARIANCE_SENSORS

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline"""),
        md_cell("""## 2. Telemetry Ingestion & Structural Inspection
We load the FD001 subset (Sea Level operating condition, High-Pressure Compressor failure mode)."""),
        code_cell("""df_fd001 = load_dataset(subset='FD001', split='train')
print(f"FD001 Training Matrix: {df_fd001.shape[0]} cycles across {df_fd001['engine_id'].nunique()} unique engines.")
df_fd001.head()"""),
        md_cell("""## 3. Sensor Telemetry Variance Analysis
Sensors operating as constant system references or with zero variance convey no degradation signal and must be pruned prior to modeling. In FD001, sensors `s1, s5, s6, s10, s16, s18, s19` exhibit near-zero variance across all 20,631 cycles."""),
        code_cell("""variances = df_fd001[SENSOR_COLUMNS].var().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#e74c3c' if s in FD001_LOW_VARIANCE_SENSORS else '#2980b9' for s in variances.index]
ax.bar(variances.index, variances.values, color=colors, edgecolor='black', linewidth=0.6)
ax.set_yscale('log')
ax.set_xlabel('Sensor Channel')
ax.set_ylabel('Variance (Log Scale)')
ax.set_title('Sensor Telemetry Variance Distribution (Red = Dropped Constant Channels)')
plt.show()"""),
        md_cell("""## 4. Engine Lifecycle Distribution
Run-to-failure engines exhibit variable operational longevity before failure, spanning from ~128 cycles to ~362 cycles with a mean near ~206 cycles."""),
        code_cell("""lifespans = df_fd001.groupby('engine_id')['cycle'].max()

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(lifespans, bins=25, kde=True, color='#16a085', ax=ax)
ax.axvline(lifespans.mean(), color='#c0392b', linestyle='--', label=f'Mean: {lifespans.mean():.1f}')
ax.axvline(lifespans.median(), color='#f39c12', linestyle=':', label=f'Median: {lifespans.median():.1f}')
ax.set_xlabel('Total Operating Cycles to Failure')
ax.set_ylabel('Engine Count')
ax.set_title('Distribution of Turbofan Lifecycles (FD001)')
ax.legend()
plt.show()"""),
        md_cell("""## 5. Multi-Sensor Degradation Trajectories
Visualizing key thermodynamic and aerodynamic sensors (HPC outlet temperature $s3$, LPT outlet temperature $s4$, HPC outlet static pressure $s11$, fuel flow ratio $s12$) reveals distinct non-linear shifts during the final 50–80 cycles of operation."""),
        code_cell("""df_labeled = add_rul_labels(df_fd001, clip_rul=False)
sample_engines = [1, 2, 3]
key_sensors = ['s2', 's3', 's4', 's7', 's11', 's12']

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()

for idx, sensor in enumerate(key_sensors):
    ax = axes[idx]
    for eid in sample_engines:
        subset_df = df_labeled[df_labeled['engine_id'] == eid]
        ax.plot(subset_df['RUL'], subset_df[sensor], label=f'Engine {eid}', alpha=0.8)
    ax.set_title(f'Sensor {sensor} Degradation Profile', fontweight='bold')
    ax.set_ylabel('Sensor Unit')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend()

axes[3].set_xlabel('Remaining Useful Life (RUL -> 0 Failure)')
axes[4].set_xlabel('Remaining Useful Life (RUL -> 0 Failure)')
axes[5].set_xlabel('Remaining Useful Life (RUL -> 0 Failure)')
plt.tight_layout()
plt.show()"""),
        md_cell("""## 6. Sensor Cross-Correlation Matrix
Evaluating Pearson correlation among retained sensors highlights strong collinearity between LPC/HPC temperatures ($s2, s3, s4$) and core pressure channels ($s8, s13$). Tree-based gradient boosters are ideally suited to exploit these non-linear interdependencies."""),
        code_cell("""active_sensors = [s for s in SENSOR_COLUMNS if s not in FD001_LOW_VARIANCE_SENSORS]
corr = df_fd001[active_sensors].corr()

plt.figure(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', square=True, linewidths=0.5)
plt.title('Inter-Sensor Telemetry Correlation Matrix (FD001 Active Sensors)')
plt.show()"""),
        md_cell("""## 7. Conclusions & Modeling Guidance
1. **Low-Variance Drop**: 7 sensors (`s1, s5, s6, s10, s16, s18, s19`) must be removed to avoid uninformative noise.
2. **Degradation Horizon**: Healthy engines exhibit stable baseline telemetry until ~100–125 cycles before failure. This motivates **Piecewise Linear RUL Clipping at 125 cycles**.
3. **Temporal Features**: Moving window averages and cumulative wear indicators are needed to filter high-frequency sensor jitter.""")
    ]
    return create_notebook(cells)


def build_feature_engineering_notebook():
    cells = [
        md_cell("""# 02. Temporal Feature Engineering Pipeline
### Industrial Predictive Maintenance Portfolio Project | NASA C-MAPSS Dataset

---

## 1. Motivation & Technical Objectives
Raw snapshot sensor telemetry suffers from high-frequency measurement jitter and does not inherently reflect the temporal history of degradation. To provide regressors with degradation context, we engineer three classes of features:

1. **Engine-Isolated Rolling Window Statistics** ($\mu, \sigma, \min, \max$ over window $w=10$): Captures trend and operational volatility without crossing engine boundaries.
2. **Cumulative Absolute Change (Wear Proxy)**: Monotonically proxies accumulated thermal and mechanical wear.
3. **Normalized Cycle Position**: Represents relative engine aging within observed telemetry.
4. **Piecewise Linear RUL Clipping ($RUL_{clip} = 125$)**: Capping the target RUL focuses model capacity exclusively on the active degradation zone.
"""),
        code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_dataset, SENSOR_COLUMNS
from src.preprocessor import add_rul_labels, FD001_LOW_VARIANCE_SENSORS, RUL_CLIP
from src.feature_engineer import build_feature_pipeline, add_rolling_features, add_cumulative_wear_features"""),
        md_cell("""## 2. Piecewise Linear RUL Target Formulation
In industrial turbofans, an engine at cycle 5 with 300 cycles to failure exhibits identical telemetry to cycle 50 with 255 cycles to failure. Early in lifecycle, no wear is detectable. We cap $RUL \le 125$."""),
        code_cell("""df_raw = load_dataset('FD001', 'train')
df_unclipped = add_rul_labels(df_raw, clip_rul=False)
df_clipped = add_rul_labels(df_raw, clip_rul=True, rul_clip_limit=RUL_CLIP)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_unclipped[df_unclipped['engine_id'] == 1]['cycle'], df_unclipped[df_unclipped['engine_id'] == 1]['RUL'], label='Linear RUL (Unclipped)', color='#e74c3c', linestyle='--')
ax.plot(df_clipped[df_clipped['engine_id'] == 1]['cycle'], df_clipped[df_clipped['engine_id'] == 1]['RUL'], label=f'Piecewise Linear RUL (Clipped at {RUL_CLIP})', color='#2ecc71', linewidth=2)
ax.set_xlabel('Operational Flight Cycle')
ax.set_ylabel('Target RUL Label')
ax.set_title('Piecewise Linear RUL Labeling Strategy (Engine 1)')
ax.legend()
plt.show()"""),
        md_cell("""## 3. Rolling Window Feature Extraction
Calculated strictly per engine using `groupby('engine_id').transform(...)` with `min_periods=1`."""),
        code_cell("""retained_sensors = [s for s in SENSOR_COLUMNS if s not in FD001_LOW_VARIANCE_SENSORS]
df_feat, _ = build_feature_pipeline(df_clipped, sensor_cols=retained_sensors, window=10)
print(f"Constructed feature matrix with {df_feat.shape[1]} columns across {df_feat.shape[0]} rows.")
df_feat[['engine_id', 'cycle', 's2', 's2_mean_10', 's2_std_10', 's2_cum_change', 'cycle_norm', 'RUL']].head(12)"""),
        md_cell("""## 4. Verification: No Data Leakage Across Engine Boundaries
Validating that rolling and cumulative features reset to 0 at cycle 1 of every engine."""),
        code_cell("""e1_end = df_feat[df_feat['engine_id'] == 1].iloc[-1]
e2_start = df_feat[df_feat['engine_id'] == 2].iloc[0]

print("Engine 1 Final Cycle Cumulative Change (s2):", e1_end['s2_cum_change'])
print("Engine 2 Cycle 1 Cumulative Change (s2)   :", e2_start['s2_cum_change'])
assert e2_start['s2_cum_change'] == 0.0, "Leakage detected across engine boundary!"
print("Assertion Passed: Engine-level isolation confirmed.")""")
    ]
    return create_notebook(cells)


def build_model_benchmarking_notebook():
    cells = [
        md_cell("""# 03. Machine Learning Model Benchmarking & GroupKFold CV
### Industrial Predictive Maintenance Portfolio Project | NASA C-MAPSS Dataset

---

## 1. Objectives & Methodological Rigor
This notebook trains, evaluates, and rigorously compares 4 regression architectures:
1. **Ridge Regression**: Linear regularization baseline.
2. **Random Forest**: Non-linear bagging ensemble.
3. **XGBoost**: Extreme Gradient Boosting with shrinkage and depth penalization.
4. **LightGBM**: Fast histogram-based gradient boosting, optimized via **Optuna**.

### Cross-Validation Strategy:
We strictly enforce **GroupKFold (5 splits)** grouped on `engine_id`. Cycles from the same engine are never distributed between training and validation folds, eliminating temporal data leakage.

### Evaluation Metrics:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **NASA Asymmetric Scoring Function** ($S = \sum (\exp(-d/13)-1)$ for early, $\exp(d/10)-1$ for late predictions).
"""),
        code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.evaluator import nasa_score, evaluate_predictions"""),
        md_cell("""## 2. Benchmark Results Review
Loading cross-validation benchmark results generated by `scripts/train_and_benchmark.py`."""),
        code_cell("""benchmark_df = pd.read_csv(PROJECT_ROOT / "reports" / "model_comparison.csv")
benchmark_df"""),
        md_cell("""## 3. Visual Performance Comparison
Comparing RMSE, MAE, and the NASA Asymmetric Penalty across all models."""),
        code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models = benchmark_df['Model']
palette = ['#2ecc71' if 'Tuned' in m else '#3498db' for m in models]

axes[0].barh(models, benchmark_df['CV_RMSE'], color=palette, edgecolor='black')
axes[0].set_title('Cross-Validation RMSE (Cycles)', fontweight='bold')
axes[0].set_xlabel('RMSE (Lower is Better)')

axes[1].barh(models, benchmark_df['CV_MAE'], color=palette, edgecolor='black')
axes[1].set_title('Cross-Validation MAE (Cycles)', fontweight='bold')
axes[1].set_xlabel('MAE (Lower is Better)')

axes[2].barh(models, benchmark_df['CV_NASA_Score'], color=palette, edgecolor='black')
axes[2].set_title('NASA Asymmetric Penalty', fontweight='bold')
axes[2].set_xlabel('Score (Lower is Better)')

plt.suptitle('NASA C-MAPSS FD001 Model Comparison (5-Fold GroupKFold CV)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()"""),
        md_cell("""## 4. Test Set Evaluation (100 Unseen Turbofan Engines)
The champion model was evaluated against the unseen `test_FD001.txt` dataset and ground-truth `RUL_FD001.txt` at the last observed cycle."""),
        code_cell("""# Display saved test evaluation curve
img_path = PROJECT_ROOT / "reports" / "figures" / "test_prediction_rul_curve.png"
if img_path.exists():
    import matplotlib.image as mpimg
    img = mpimg.imread(str(img_path))
    plt.figure(figsize=(14, 7))
    plt.imshow(img)
    plt.axis('off')
    plt.title('True vs Predicted Remaining Useful Life (100 Test Engines)', fontsize=14)
    plt.show()""")
    ]
    return create_notebook(cells)


def build_shap_notebook():
    cells = [
        md_cell("""# 04. Model Explainability with SHAP (SHapley Additive exPlanations)
### Industrial Predictive Maintenance Portfolio Project | NASA C-MAPSS Dataset

---

## 1. Motivation: Explainability in High-Stakes Industrial Prognostics
In commercial aviation and power generation, predictive maintenance models cannot operate as opaque black boxes. Reliability engineers require diagnostic clarity:
- **Global Interpretability**: Which physical sensor channels drive RUL predictions most decisively across the fleet?
- **Local Interpretability**: Why is an individual engine assigned an impending failure alert (e.g. RUL = 18 cycles)? Which specific components (HPC, HPT, fan) are accelerating failure?

Using game-theoretic **SHAP (TreeExplainer)**, we decompose predictions into additive feature attributions."""),
        code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explainer import create_tree_explainer, get_top_shap_contributors"""),
        md_cell("""## 2. Global Feature Importance: Beeswarm Summary Plot
Displays feature impact on model output. Blue dots represent low feature values; red dots represent high feature values."""),
        code_cell("""img_beeswarm = PROJECT_ROOT / "reports" / "figures" / "shap_beeswarm.png"
if img_beeswarm.exists():
    import matplotlib.image as mpimg
    img = mpimg.imread(str(img_beeswarm))
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()"""),
        md_cell("""## 3. Local Waterfall Explanation (Single Engine Near Critical Failure)
Waterfall plot showing the exact additive breakdown from base value to final predicted RUL."""),
        code_cell("""img_waterfall = PROJECT_ROOT / "reports" / "figures" / "shap_waterfall.png"
if img_waterfall.exists():
    import matplotlib.image as mpimg
    img = mpimg.imread(str(img_waterfall))
    plt.figure(figsize=(11, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()"""),
        md_cell("""## 4. API Integration Test: Live Local Explanation Extraction
Testing the SHAP attribution extraction utility used by the REST API endpoint `POST /predict`."""),
        code_cell("""from api.predict import inference_engine

mock_payload = {
    "engine_id": 42,
    "cycles": [
        {"cycle": 1, "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0, "s2": 641.82, "s3": 1589.7, "s4": 1400.6, "s7": 554.36, "s8": 2388.06, "s9": 9046.19, "s11": 47.47, "s12": 521.66, "s13": 2388.02, "s14": 8138.62, "s15": 8.4195, "s17": 392, "s20": 39.06, "s21": 23.419}
    ]
}

if inference_engine.is_ready:
    res = inference_engine.predict(mock_payload)
    print("REST API Prediction Response:")
    import pprint
    pprint.pprint(res)
else:
    print("Inference engine artifacts are being finalized...")""")
    ]
    return create_notebook(cells)


def main():
    notebooks = {
        "01_EDA.ipynb": build_eda_notebook(),
        "02_Feature_Engineering.ipynb": build_feature_engineering_notebook(),
        "03_Model_Benchmarking.ipynb": build_model_benchmarking_notebook(),
        "04_SHAP_Explainability.ipynb": build_shap_notebook(),
    }

    for name, content in notebooks.items():
        filepath = NOTEBOOKS_DIR / name
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        print(f"Generated {filepath}")


if __name__ == "__main__":
    main()
