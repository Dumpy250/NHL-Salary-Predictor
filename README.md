# 🏒 NHL Salary Predictor

An exploratory Python machine-learning and Tkinter application that estimates NHL player salary from five draft and performance fields using linear regression. It demonstrates a simple train/evaluate/save workflow and an interactive prediction form—not a production compensation model.

## 📌 Project Overview

The model uses draft year, draft round, overall draft position, games played and goals scored from `train.csv`, with `Salary` as the target.

**Starting `view.py` trains a new model and overwrites `salary_model.joblib` before opening the GUI.** Importing `view` does the same because the training and GUI startup are at module level. The tracked model file does not provide a prediction-only startup mode.

## 🧠 How It Works

### 1. Data Preparation
- Reads `train.csv` from the current working directory using `encoding="latin-1"`.
- Selects the five feature columns and `Salary` directly. There is no scaling, imputation, categorical encoding or explicit data-cleaning pipeline.
- Uses `train_test_split(test_size=0.25, random_state=42)`: a shuffled 75%/25% split of the bundled training CSV.

### 2. Model Training
- Fits scikit-learn's `LinearRegression` on the training portion only.
- Saves that fitted estimator, then prints training and test **Mean Absolute Error (MAE)** to the console.
- Does not use cross-validation, a time-based holdout, a baseline comparison or the upstream dataset's separate test files.

### 3. Model Persistence
- `salary_model.joblib` is **already tracked** and is overwritten on each successful startup training pass.
- Each prediction reloads that file with `joblib.load`; prediction clicks do not retrain, but restarting the application does.
- [Loading joblib/pickle-based models can execute arbitrary code](https://scikit-learn.org/stable/model_persistence.html). Only use artifacts you trust. The checked-in file's training environment and compatibility are not established by its presence; cross-version scikit-learn loading is not supported as a general portability guarantee.

### 4. User Interface
- Tkinter numeric spinboxes validate individual input ranges.
- The Predict button is disabled until all fields are nonempty and valid; Enter also requests prediction after validation.
- The output panel shows entered values and the estimate formatted with `$` and two decimal places. Formatting is not a precision or currency-validation claim.

## 📊 Features Used

| Column | Meaning | GUI range |
|---|---|---|
| `DftYr` | Draft year | 1960–2035 |
| `DftRd` | Draft round | 1–15 |
| `Ovrl` | Overall draft position | 1–500 |
| `GP` | Games played | 0–200 |
| `G` | Goals scored | 0–200 |
| `Salary` | Numeric regression target | Not entered in the GUI |

Input validation does not establish that a combination is realistic or within the training distribution, and predictions are not constrained to be nonnegative.

## 🖥️ Application UI

Enter the five draft/performance fields, then select **Predict** to load the model and display an estimate. The UI is intentionally minimal. Training must finish before the window opens, and model/file errors are not handled through a dedicated recovery workflow.

## 📈 Model Performance

The implementation calculates training MAE and held-out MAE for its single split. MAE is an absolute error in the target's numeric units, not a percentage accuracy score. Training MAE measures fit to the training portion; the held-out MAE concerns only the selected portion of this historical dataset.

No reproducible numerical benchmark is established by the supplied source and artifacts. A saved model file alone does not establish measured accuracy, current-season performance, causal relationships or generalization to new populations.

## Dataset and provenance

The existing project attributes its data to Cam Nugent's [Predict NHL Player Salaries dataset](https://www.kaggle.com/datasets/camnugent/predict-nhl-player-salaries). That data card describes 2016/2017 salaries and an upstream train/test split. This application makes another split within its supplied `train.csv`; it does not read the upstream `test.csv` or `test_salaries.csv`, neither of which is tracked here.

`train.csv` is included, so setup does not require a fresh Kaggle download. Exact upstream-file equivalence and redistribution/licensing rights have not been independently established here. Consult the source dataset's terms before redistributing or using it beyond the project; no license is inferred from public availability.

## Setup and running

These are **source-checked, unexecuted instructions**. Use a graphical desktop with Tcl/Tk support. The supplied `environment.yml` names `salary-predictor` and specifies Python 3.11 plus pandas, scikit-learn, joblib and pip. Library versions, channels and a lockfile are not supplied, so it is not an exact record of the model's training environment. Conda resolution and platform compatibility remain unverified.

From a local checkout, use the repository root containing `view.py`, `train.csv` and `environment.yml`:

```sh
conda env create --file environment.yml
conda activate salary-predictor
```

Environment creation installs dependencies; it is not a read-only check. Use a distinct environment if that name already exists rather than overwriting unrelated work. Installing the Python packages alone does not guarantee a working Tk graphical session.

**The next command trains, evaluates, writes the tracked model file and opens the GUI.** Do not run it merely to inspect the project or expecting to preserve the existing model unchanged. Use a disposable working copy when experimenting, and do not import `view.py` for read-only inspection.

```sh
python view.py
```

Keep the working directory at the repository root: both the CSV and model paths are relative to it. There is no separate train/load switch or prediction-only command in this revision.

## 📁 Project Structure

```text
NHL-Salary-Predictor/
├── view.py                # Training, model loading/prediction and Tkinter UI
├── train.csv              # Included historical training data
├── salary_model.joblib    # Tracked serialized estimator; overwritten by training
├── environment.yml        # Conda specification, not a resolved lockfile
├── README.md
└── .gitattributes
```

## ⚠️ Limitations

- Basic linear regression using only five fields; it omits player position, assists, points, ice time, contract structure, inflation and salary-cap-era effects.
- No explicit missing-value/type cleanup, train/test leakage audit, uncertainty intervals or robust out-of-distribution handling.
- No automated test suite is included. Model output requires independent validation before use.
- Estimates are educational model outputs, not current salary valuations or compensation advice.
