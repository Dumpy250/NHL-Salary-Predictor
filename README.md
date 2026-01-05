# 🏒 NHL Salary Predictor

A Python-based machine learning application that predicts an NHL player’s estimated salary using historical draft and performance data . The project combines **data preprocessing**, **linear regression modeling**, and a lightweight **Tkinter desktop interface** for interactive predictions.

This project was built as an exploratory ML + Python UI application and later cleaned up to follow better software engineering practices.

---

## 📌 Project Overview

The NHL Salary Predictor estimates player salary based on five core features:

- Draft Year  
- Draft Round  
- Overall Draft Position  
- Games Played  
- Goals Scored  

A **Linear Regression model** is trained on historical NHL data https://www.kaggle.com/datasets/camnugent/predict-nhl-player-salaries, evaluated using **Mean Absolute Error (MAE)**, and then saved locally for reuse. Users can interact with the model through a simple GUI that validates inputs and displays predicted salary in real time.

---

## 🧠 How It Works

### 1. Data Preparation
- Training data is loaded from `train.csv`
- Features and target variables are explicitly defined
- Data is split into training and testing sets (75% / 25%)

### 2. Model Training
- A `LinearRegression` model from **scikit-learn** is trained
- Model performance is evaluated using **Mean Absolute Error (MAE)**
- Results are printed to the console

### 3. Model Persistence
- The trained model is serialized using **joblib**
- Saved as a binary file (`salary_model.joblib`)
- Reloaded at prediction time to avoid retraining

### 4. User Interface
- Built using **Tkinter**
- Numeric spinboxes enforce valid input ranges
- Predict button is disabled until inputs are valid
- Estimated salary is displayed in a read-only panel

---

## 📊 Features Used

| Feature  | Description            |
|--------|------------------------|
| `DftYr` | Draft year             |
| `DftRd` | Draft round            |
| `Ovrl`  | Overall draft position |
| `GP`    | Games played           |
| `G`     | Goals scored           |
| `Salary` | Target variable (USD) |

---

## 🖥️ Application UI

The desktop UI allows users to:

- Enter player draft and performance data
- Receive instant salary predictions
- View clearly formatted output
- Avoid invalid inputs through real-time validation

The UI is intentionally minimal to keep focus on **model behavior rather than styling**.

---

## 📈 Model Performance

Example output after training:

- **Training MAE:** ~1.29M USD  
- **Test MAE:** ~1.31M USD  

These results reflect the complexity and variability of NHL salaries and demonstrate the limitations of using a simple linear model for real-world compensation prediction.

---

## 📁 Project Structure

```text
NHL-Salary-Predictor/
│
├── view.py                # Model training, prediction logic, and UI
├── train.csv              # Training dataset
├── salary_model.joblib    # Serialized ML model (generated at runtime)
├── environment.yml        # Conda environment definition
├── README.md              # Project documentation
└── .gitattributes
```
⚠️ Limitations

 Uses a basic linear regression model

 Does not account for:

- Player position

- Assists, points, or ice time

- Contract structure or inflation

- Salary cap era differences

- Predictions should be treated as educational estimates, not real salary evaluations
