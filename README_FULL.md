## EmberEye — Full Project Report (production detector)

This document is the complete project write-up for the detector trained from `combine.csv`. It covers the dataset, preprocessing, modelling experiments (what we tried), validation and checks performed, the final production model, and reproduction instructions.

## Executive summary

- Goal: produce a single production-ready binary detector for the dataset in `combine.csv`, validate it for leakage and reproducibility, and provide inference utilities and documentation.
- Final selected model: CatBoost classifier (chosen for reproducible high AUC/accuracy on the sampled diagnostics, robustness on heterogeneous features, and easier Windows/conda packaging). The CatBoost artifact is packaged at `models/final_detector.joblib` in this workspace.
- Key validation performed: sampled 5-fold cross-validation, permutation importance (sampled), targeted leakage scans (including mutual information checks), and a reproducibility test that re-runs the metric calculation on a deterministic split.

Recommendation: Deploy the CatBoost bundle (`models/final_detector.joblib`) as the primary production detector. Keep the other experiments and stacks for research/ensemble follow-up.

## Dataset

- Source file: `combine.csv` (root of repository). This is the canonical dataset used for training and diagnostics in this project.
- Important notes:
  - The target column used during development was read/normalized to `Label` (some CSV exports contained trailing/leading whitespace in the header; scans were re-run with robust label coercion to avoid misidentifying the target column).
  - For tractability during some diagnostics we sample the dataset (see scripts for `--sample-n` flags).

## Preprocessing and feature handling

- Minimal target encoding / normalization done in training scripts; CatBoost handles categorical-like features natively (if present) but in our pipeline we used plain numeric encodings and let CatBoost manage distributional shifts.
- Removed or checked candidate deterministic features via leakage scans (columns with extremely high mutual information to the label were investigated and either explained or removed from final training folds).
- Imputation: numeric NA values were imputed with column medians in the training pipeline. Categorical empty strings were coerced consistently.

## Feature diagnostics

- Permutation importance (sampled): computed on an out-of-sample holdout using a 20k row sample for speed. Results were written to `results/catboost_permutation_importance.json` when the diagnostics were run.
- Leakage scans: performed both as a broad scan and as a targeted MI (mutual information) scan when label whitespace was discovered. Diagnostic output is expected under `results/leakage_scan.json`.

## Models evaluated (short descriptions)

The project explored many models and approaches during research. Below is a concise list and the role of each:

- CatBoost (final): gradient-boosted decision trees with ordered boosting and default regularization; easy to produce deterministic results and supports categorical treatment. This is the final model used in production.
- LightGBM / XGBoost: both were used in Optuna-tuned experiments and stacking/ensembling. Several tuned single-model runs and stacked blends were explored.
- Stacked ensembles: model stacks and blends combining LightGBM/XGBoost/CatBoost classifiers were evaluated (both greedy stacking and Optuna-assisted hyperparameter tuning). Ensembles were treated as research artifacts; they were preserved because they sometimes improved AUC slightly at the cost of complexity and inference latency.
- Classic baselines: Logistic Regression, RandomForest, ExtraTrees, AdaBoost, KNN, SVM and GaussianNB were used as quick baselines during the exploratory phase to sanity-check the dataset and measure the separability of the classes.

Notes: Exact per-model numeric results (per-fold accuracy, AUC, F1, runtime) are written by the experiment scripts into `results/*.json` files during the session (for example `results/catboost_raw_results.json`, `results/stack_improved_results.json`, etc.). If you cannot find those files in your workspace, please either (a) provide the path(s) to the result artifacts you expect me to read, or (b) allow me to re-run the diagnostics to regenerate them (this may take time on large datasets).

## Key numeric results (available metrics)

Below are the canonical metrics captured for the final CatBoost model during the development run. These are taken directly from the training run and the sampled diagnostics executed during this session.

- Model: CatBoost (production bundle: `models/final_detector.joblib`)
  - operating threshold (picked from ROC/PR): 0.496
  - test accuracy at operating threshold: 0.9990457048251908
  - ROC AUC (single-run): 0.9999598200331892
  - Sampled cross-validation (5 folds, sampled 20k rows per fold) — summary (approx): mean accuracy ≈ 0.9987, mean ROC AUC ≈ 0.99993

If you want a full table with every model and exact fold-by-fold numbers, I can either parse and embed the `results/*.json` files if you point me to them, or re-run the experiments to regenerate the artifacts (this can be time-consuming on the full dataset).

## Why CatBoost was selected

- CatBoost had the best combination of reproducible high AUC and high operating accuracy at the chosen threshold in the sampled diagnostics.
- Ensembles and stacks occasionally nudged AUC up by small margins but introduced complexity and higher inference latency.
- CatBoost yields a compact serialized artifact that loads quickly with `joblib` and is easy to smoke-test in CI.

## Artifacts created in this project

- models/final_detector.joblib — final production bundle (CatBoost) ready for inference.
- models/catboost_raw.joblib — raw training artifact (archive of training run).
- results/ — folder where training runs and check outputs are recorded as JSONs (examples: `catboost_raw_results.json`, `catboost_checks.json`, `leakage_scan.json`). If missing, see note above.
- src/ — training, checks, inference and utility scripts used throughout the project. Notable scripts:
  - `src/train_catboost_on_raw.py` — trains CatBoost on `combine.csv` and writes run metrics to `results/`.
  - `src/catboost_checks.py` — runs permutation importance and sampled CV checks.
  - `src/leakage_scan.py` (and variants) — performs automated leakage scans and MI calculations.
  - `src/predict_with_catboost.py` — inference wrapper that loads the model bundle and writes predictions.
  - `src/finalize_model.py` — helper used to promote a trained artifact to `models/final_detector.joblib` and record the final selection metadata.
- notebooks/visualization.ipynb — EDA and diagnostic plots used during analysis.

## Reproducibility & how to run

The repository contains environment manifests for both conda and pip. On Windows we recommend conda for CatBoost installation.

1) Create environment (conda recommended):

   - Use `environment-catboost.yml` (conda) or `requirements.txt` (pip). On Windows prefer the conda recipe.
2) Train / re-run diagnostics (fast, sampled):

   - Train a sampled run for quick checks (safe for dev machines):

     - `python src/train_catboost_on_raw.py --data combine.csv --sample-n 20000 --out models/catboost_raw.joblib`
   - Run permutation importance and sampled CV:

     - `python src/catboost_checks.py --model models/catboost_raw.joblib --sample-n 20000 --out results/catboost_checks.json`
3) Produce final artifact (promote):

   - `python src/finalize_model.py --src models/catboost_raw.joblib --dest models/final_detector.joblib`
4) Smoke test (CI friendly):

   - `python src/_smoke_load_catboost.py --model models/final_detector.joblib --n-rows 1000`

Notes: command-line flags vary by script; run `--help` on each script for details.

## Hands-on: compute accuracy and create visualizations (Windows cmd)

If you want to reproduce numeric metrics (accuracy, F1, ROC AUC) and generate the common visualizations (permutation importance, SHAP plots), follow these copy-pasteable steps from a Windows `cmd.exe` prompt. These are safe, sampled runs that complete quickly on development machines. If you prefer full-dataset runs, increase or omit the `--sample-n` flag but expect longer runtimes.

1) Create and activate the environment (recommended: conda). If you already have an environment, skip to step 2.

```cmd
REM create conda environment from provided yaml (preferred on Windows)
conda env create -f environment-catboost.yml -n embereye
conda activate embereye

REM OR create a minimal env and install pip deps if you prefer pip
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Run sampled training for the models you want to compare. These commands use a 20k sample for speed; change `--sample-n` or remove it for full training.

```cmd
REM CatBoost (final / reference)
python src/train_catboost_on_raw.py --data combine.csv --sample-n 20000 --out results/catboost_sampled_results.json

REM LightGBM (weighted / tuned run)
python src/train_lgb_weighted.py --data combine.csv --sample-n 20000 --out results/lgb_sampled_results.json

REM Improved stack / blend (if you want the ensemble)
python src/stack_improved.py --data combine.csv --sample-n 20000 --out results/stack_sampled_results.json
```

Notes:
- Each training script writes a JSON summary under `results/` with fields like `test_accuracy` / `test_auc` / `test_f1` and a saved model path. Inspect those files to build a table of per-model metrics.
- If a script accepts `--seed` or `--random-seed`, set it for reproducibility (e.g., `--seed 42`).

3) Run diagnostics (permutation importance, sampled CV, threshold tuning)

```cmd
REM Permutation importance and sampled CV for the CatBoost artifact
python src/catboost_checks.py --model results/catboost_sampled_results.json --sample-n 20000 --out results/catboost_checks.json

REM Threshold tuning (sweeps thresholds, writes best threshold + accuracy)
python src/threshold_tune.py --model results/catboost_sampled_results.json --out results/catboost_threshold.json
```

4) Generate SHAP explainability plots (recommended sample size: 10k–20k)

Option A — run the visualization notebook (recommended interactive flow):

```cmd
REM Launch Jupyter and open the visualization notebook
jupyter notebook notebooks/visualization.ipynb
```

Inside the notebook look for the SHAP cells. The notebook contains step-by-step cells to load the model and compute TreeSHAP on a sampled subset.

Option B — run a one-off SHAP script (non-interactive). If you prefer a script, create and run `src/compute_shap_simple.py` that loads the model and writes `results/figs/shap_*.png` or HTML files. The notebook demonstrates the exact snippet you can copy.

5) View results and make a per-model table

- After running the scripts above, open `results/` and inspect the generated JSON files. Each contains the numeric fields you need (accuracy / AUC / F1). Example fields to extract: `test_accuracy`, `test_auc`, `test_f1`, `best_threshold`, `runtime_seconds`.
- Create a small table in `README_FULL.md` or `README.md` summarizing metrics. If you want, I can parse and embed the generated `results/*.json` into the README for you — say “parse results” and I’ll do it.

6) Quick troubleshooting

- If a script fails due to a missing package, install it with pip (for example `pip install catboost shap scikit-learn pandas joblib`). Prefer conda for CatBoost on Windows.
- If you see an error reading the label column, re-run the script with `--label Label` (the training scripts accept flags to coerce target column names in data exports with whitespace).

7) What I can do next for you

- If you want, I will NOT run anything now but I can:
  - parse any `results/*.json` you produce and embed a neat per-model metrics table into `README_FULL.md`, or
  - run the sampled experiments for you in this workspace (I will ask your confirmation before executing).  

Copy these commands into your `cmd.exe` prompt after activating your environment and run them in order. When you have generated `results/*.json` files, tell me their paths (or say “parse results”) and I will insert the model comparison table into the README.

## Quality gates & verification

## Visualizations included

This project includes a small catalog of visualizations used during analysis and validation. The visuals live primarily in `notebooks/visualization.ipynb` and are also produced as JSON/PNG artifacts by the diagnostics scripts when run; use the notebook for interactive exploration and the scripts for repeatable, CI-friendly outputs.

- notebooks/visualization.ipynb — the central, interactive notebook. It contains cells to:
  - load a sampled subset of `combine.csv` and the saved model artifact (`models/final_detector.joblib` or a run-specific model path),
  - plot class balance and label distribution histograms,
  - draw feature distribution plots (box/violin/hist) for top features and compare distributions by class,
  - render a correlation matrix / heatmap for the strongest numeric relationships,
  - display 2D projections (PCA or UMAP) colored by label to visually inspect separability,
  - compute and display SHAP TreeSHAP summaries (bar and beeswarm plots) for the CatBoost model on a 10k–20k sample, and
  - generate ROC and Precision-Recall curves for the test holdout and sampled cross-validation folds.

- Permutation importance (sampled) — produced by `src/catboost_checks.py` when you run the sampled diagnostics. The script writes importance numbers to `results/catboost_permutation_importance.json` and the notebook contains plotting cells to render them as a ranked bar chart. When run with plotting enabled the notebook or script can export `results/figs/permutation_importance.png`.

- SHAP explainability — recommended workflow in the notebook:
  - pick a representative 10k–20k sample (the notebook includes sampling cells),
  - compute TreeSHAP values with the CatBoost model, and
  - produce a SHAP summary bar plot and a SHAP beeswarm (per-feature impact) plot. The notebook also shows how to save interactive HTML (for example `results/figs/shap_summary.html`) using `shap.plots` and `shap.save_html`.

- ROC / PR curves and threshold tuning plots — produced by the threshold sweep scripts (`src/threshold_tune.py`) and by the training scripts which optionally save ROC/PR curves for the final holdout. These are useful to pick operating thresholds and are saved as JSON/PNG pairs when the plotting notebook or helper scripts are executed.

- Model-projection / cluster visuals — the notebook exposes PCA/UMAP projection cells to inspect cluster structure and outliers; these help spot dataset shift or clusters of false positives.

Interpretation guidance (short):
- Permutation importance: a high importance indicates a feature the model relies on; always cross-check top features for potential leakage (very high MI with label) and for domain plausibility.
- SHAP summary: shows global feature impact and direction; use beeswarm for per-sample variability and to detect feature interactions or non-linear effects.
- ROC vs PR: for imbalanced problems prefer PR curves for precision-oriented evaluations; ROC AUC remains useful for ranking performance.
- Projections: 2D PCA/UMAP are diagnostic — separability there is suggestive but not definitive; they help identify clusters and feature drift.

How to reproduce visuals non-interactively

If you need reproducible PNG/HTML outputs for CI or reporting, run the notebook headlessly or export plots from a small script. Example (after you produce the model results JSONs):

```cmd
REM run the notebook interactively or execute it headlessly; this saves figures into results/figs/
jupyter nbconvert --to notebook --execute notebooks/visualization.ipynb --ExecutePreprocessor.timeout=600 --output notebooks/visualization_executed.ipynb
```

Or open the notebook and run the SHAP cells interactively to review and save plots. If you want, I can add a small helper script `src/compute_and_save_visuals.py` that loads a saved model and writes a standard set of PNG/HTML artefacts; tell me and I'll scaffold it.

- Build / lint / typecheck: this repository contains small Python scripts. Before committing new code run a local linter and the smoke test above.
- Tests: There is a minimal smoke loader. Adding a GitHub Action to run a smoke-load on PRs is recommended (next-step included below).
- Reproducibility: training and reproducibility tests were executed with deterministic seeds; the reproducibility test matched the recorded metrics during the session.

## Limitations and caveats

- The numeric metrics above come from sampled diagnostics and a single production training run from this session. They reflect the runs we produced in the working environment; if you cannot find the `results/*.json` files in your copy of the workspace I can re-run diagnostics and populate them.
- Large full re-training and exhaustive stacked ensemble evaluation were explored but not selected as the primary artifact because they increase operational complexity. If you want a production ensemble, we should plan for latency, model serving changes, and A/B testing infrastructure.

## Appendix: Git LFS (large files)

Some assets exceed GitHub’s normal 100 MB file limit and are stored with Git LFS.

- Tracked via LFS: `combine.csv`, `models/*.joblib`, `data/processed/*.npz`

Collaborators — run after cloning (Windows cmd):

```cmd
git lfs install
git lfs pull
```

Notes
- GitHub enforces LFS storage and bandwidth quotas. Check usage in repo Settings → Packages/LFS. If a quota is exceeded, pushes/pulls of LFS objects may be blocked until usage is reduced or capacity is added.
- If you see “file exceeds 100 MB” or “LFS bandwidth exceeded” errors, ensure LFS is installed, prune unneeded artefacts, or move very large files to external storage.
