# view.py
import tkinter as tk
from tkinter import Spinbox, Text, Label, Button, E
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# --- Config -------------------------------------------------------------------
FEATURE_COLS = ["DftYr", "DftRd", "Ovrl", "GP", "G"]
TARGET_COL = "Salary"
TRAIN_CSV = "train.csv"
MODEL_PATH = "salary_model.joblib"   # keeping your original filename for compatibility

# --- Model --------------------------------------------------------------------
class Dataset:
    def train(self):
        df = pd.read_csv(TRAIN_CSV, encoding="latin-1")
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = LinearRegression().fit(X_train, y_train)

        # Persist model
        joblib.dump(model, MODEL_PATH)

        # Report MAE
        print("Model training results:")
        print(f" - Training MAE: {mean_absolute_error(y_train, model.predict(X_train)):.4f}")
        print(f" - Test MAE:     {mean_absolute_error(y_test, model.predict(X_test)):.4f}")

    def predict_from_inputs(self, values_dict):
        """values_dict: {col: int} matching FEATURE_COLS order."""
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([[values_dict[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
        pred = float(model.predict(X)[0])
        return pred

User1 = Dataset()
User1.train()

# --- UI -----------------------------------------------------------------------
root = tk.Tk()
root.title("Salary Predictor")

T = Text(root, state="disabled", height=10, width=30)

label_3 = Label(root, text="Draft Year")
label_4 = Label(root, text="Draft Rounds")
label_5 = Label(root, text="Overall Draft")
label_6 = Label(root, text="Games Played")
label_7 = Label(root, text="Number of Goals")

def _in_range_int(s: str, lo: int, hi: int) -> bool:
    if s == "":  # allow empty while typing
        return True
    if not s.isdigit():
        return False
    v = int(s)
    return lo <= v <= hi

# validation commands (Tk expects strings; use %P for “proposed value”)
vcmd_year  = (root.register(lambda P: _in_range_int(P, 1960, 2035)), "%P")
vcmd_round = (root.register(lambda P: _in_range_int(P, 1, 15)), "%P")
vcmd_ovr   = (root.register(lambda P: _in_range_int(P, 1, 500)), "%P")
vcmd_games = (root.register(lambda P: _in_range_int(P, 0, 200)), "%P")
vcmd_goals = (root.register(lambda P: _in_range_int(P, 0, 200)), "%P")

draft_year    = Spinbox(root, from_=1960, to=2035, width=8, validate="key", vcmd=vcmd_year)
draft_rounds  = Spinbox(root, from_=1,    to=15,   width=8, validate="key", vcmd=vcmd_round)
overall_draft = Spinbox(root, from_=1,    to=500,  width=8, validate="key", vcmd=vcmd_ovr)
games_played  = Spinbox(root, from_=0,    to=200,  width=8, validate="key", vcmd=vcmd_games)
num_goals     = Spinbox(root, from_=0,    to=200,  width=8, validate="key", vcmd=vcmd_goals)

def _all_valid():
    defs = [
        (draft_year,    1960, 2035),
        (draft_rounds,  1,    15),
        (overall_draft, 1,    500),
        (games_played,  0,    200),
        (num_goals,     0,    200),
    ]
    ok = True
    for w, lo, hi in defs:
        s = w.get().strip()
        good = s != "" and _in_range_int(s, lo, hi)
        w.configure(bg=("white" if good else "#ffecec"))  # live feedback
        ok &= good
    predict_btn.config(state=("normal" if ok else "disabled"))
    return ok

def on_predict():
    if not _all_valid():
        return
    vals = {
        "DftYr": int(draft_year.get()),
        "DftRd": int(draft_rounds.get()),
        "Ovrl":  int(overall_draft.get()),
        "GP":    int(games_played.get()),
        "G":     int(num_goals.get()),
    }
    est = User1.predict_from_inputs(vals)

    T.configure(state="normal")
    T.delete("1.0", tk.END)
    T.insert("end", "Player Details\n")
    for k in FEATURE_COLS:
        T.insert("end", f"- {vals[k]} {k}\n")
    T.insert("end", f"Estimated Salary: ${est:,.2f}\n")
    T.configure(state="disabled")

# Bind live validation
for w in (draft_year, draft_rounds, overall_draft, games_played, num_goals):
    w.bind("<KeyRelease>", lambda e: _all_valid())
    w.config(command=_all_valid)  # when arrows are clicked

# Layout
label_3.grid(row=2, sticky=E); draft_year.grid(row=2, column=1)
label_4.grid(row=3, sticky=E); draft_rounds.grid(row=3, column=1)
label_5.grid(row=4, sticky=E); overall_draft.grid(row=4, column=1)
label_6.grid(row=5, sticky=E); games_played.grid(row=5, column=1)
label_7.grid(row=6, sticky=E); num_goals.grid(row=6, column=1)

predict_btn = Button(root, text="Predict", command=on_predict, state="disabled")
predict_btn.grid(columnspan=1)
T.grid(row=7, column=1)

# Enter key triggers prediction if valid
root.bind("<Return>", lambda e: on_predict())

# initialize UI state
_all_valid()

root.mainloop()