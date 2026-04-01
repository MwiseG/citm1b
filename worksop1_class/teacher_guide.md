# Teacher Guide - Workshop 1

## Workshop goal

Students should complete the two main notebooks (`01_exploration.ipynb` and `02_preprocessing_and_baseline.ipynb`).  
Each group should present:
- one plot from exploration
- one validation RMSE value from their baseline model

## Time-boxed flow (slow tempo, interactive)

### 0-10 min: Introduction

- **Teacher says/does:** Explain the mini end-to-end ML flow for today: explore data -> preprocess -> train baseline -> reflect.
- **Students do:** Form small groups and open the workshop folder.
- **Why:** Sets realistic expectations and avoids "just fit a model quickly" behavior.

### 10-30 min: Setup and environment check

- **Teacher says/does:** Ask students to use Colab first, then run `00_check_environment.ipynb`.
- **Students do:** Verify imports and load the first rows of data.
- **Why:** Removes setup friction and keeps workshop time focused on learning, not installation.

### 30-75 min: Exploration notebook

- **Teacher says/does:** Guide groups through data overview and visualization; pause to ask what patterns they observe.
- **Students do:** Complete TODOs in `01_exploration.ipynb`, discuss features, and create at least one plot.
- **Why:** Reduces the pitfall of training models before understanding data quality and feature meaning.

### 75-140 min: Preprocessing and baseline

- **Teacher says/does:** Walk through stratified split, pipeline fitting on train data only, and baseline evaluation.
- **Students do:** Complete TODOs in `02_preprocessing_and_baseline.ipynb` and compute train/validation RMSE.
- **Why:** Highlights data leakage and sampling bias risks, and shows structured evaluation.

### 140-170 min: Group mini-presentations

- **Teacher says/does:** Invite short group reports and connect findings to CRISP-DM stages.
- **Students do:** Share one plot, one validation RMSE, and one pitfall/lesson learned.
- **Why:** Reinforces understanding, reflection, and communication of ML decisions.

## Common ML pitfalls to highlight

- Using the test set too early, leading to optimistic performance estimates.
- Data leakage from fitting preprocessing on validation/test data.
- Sampling bias caused by poor train/validation/test split strategy.
- Overfitting small datasets with unnecessarily complex models.
- Blindly trusting LLM-generated code without understanding it.
