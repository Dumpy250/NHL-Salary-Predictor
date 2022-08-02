from tkinter import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib


class dataset:
    # Load the data set

    def train(self):
        df = pd.read_csv("train.csv", encoding="latin-1")
        X = df[["DftYr", "DftRd", "Ovrl", "GP", "G"]]


        #X = df[["Lastname", "Firstname"]]
        # Create x and y Arrays
        #X = df[["DftYr", "DftRd", "Ovrl", "GP", "G", "A1", "A2", "PTS", "+/-", "E+/-", "PIM", "TOI", "TOI/GP", "TOI%",
        #        "IPP%", "SH%", "SV%", "PDO", "F/60", "A/60", "Pct%", "Diff", "Diff/60", "iCF", "iFF", "iSF", "ixG", "iSCF",
        #        "iRB", "iRS", "iDS", "sDist", "Pass", "iHF", "iHA", "iHDf", "iMiss", "iGVA", "iGVA", "iTKA",
        #        "iBLK", "BLK%", "iFOW", "iFOL", "FO%", "%FOT", "dzFOW", "dzFOL", "nzFOW", "nzFOL", "ozFOW", "ozFOL",
        #        "FOW.Up", "FOW.Down", "FOL.Down", "FOW.Close", "OTG", "1G", "GWG", "ENG", "PSG", "PSA", "G.Bkhd",
        #        "G.Dflct", "G.Slap", "G.S0p", "G.Tip", "G.Wrap", "G.Wrst", "Post", "Over", "Wide", "S.Bkhd",
        #        "S.Dflct", "S.Slap", "S.S0p", "S.Tip", "S.Wrap", "S.Wrst", "iPenT", "iPenD", "iPENT", "iPEND", "iPenDf",
        #        "NPD", "Min", "Maj", "Match", "Misc", "Game", "CF", "CA", "FF", "FA", "SF", "SA", "xGF", "xGA", "SCF",
        #        "SCA", "GF", "GA", "RBF", "RBA", "RSF", "RSA", "DSF", "DSA", "FOW", "FOL", "HF", "HA", "GVA", "TKA",
        #        "PENT", "PEND", "OPS", "DPS", "PS", "OTOI", "Grit", "DAP", "Pace", "GS", "GS/G"]]


        y = df["Salary"]

        # Data needs to be scaled to 0 to 1 for neural network to train correctly
        X_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both the training inputs and outputs
        #X[X.columns] = X_scaler.fit_transform(X[X.columns])
        #y[y.columns] = y_scaler.fit_transform(y[y.columns])


        # Split the data set in a training set (75%) and a test set (25%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Create the Linear Regression model
        model = LinearRegression()

        # train the model
        model.fit(X_train, y_train)

        # Save the trained model to a file so we can use it to make predictions later
        joblib.dump(model, 'Salary.csv')

        # Report how well the model is performing
        print("Model training results:")

        # Report an error rate on the training set
        mse_train = mean_absolute_error(y_train, model.predict(X_train))
        print(f" - Training Set Error: {mse_train}")

        # Report an error rate on the test set
        mse_test = mean_absolute_error(y_test, model.predict(X_test))
        print(f" - Test Set Error: {mse_test}")

    def use(self):
        # Load our trained model
        model = joblib.load('Salary.csv')
        # Define the player that we want to value (with the values in the same order as in the training data)
        #print(draft_year.get(), " ", draft_rounds.get(), " ", overall_draft.get(), " ", games_played.get(), " ", num_goals.get())
        player_1 = [
            int(draft_year.get()),  # DftYr Draft Year
            int(draft_rounds.get()),  # DftRd Draft Round
            int(overall_draft.get()),  # Ovrl Overall drafted
            int(games_played.get()),  # GP Games played
            int(num_goals.get()), # G Goals

            #5, # A1 First assists, primary assists
            #6, #A2 Second Assists, secondary assists
            #10, #PTS Points, Goals plus all assets
            #0, #+/- Plus/minus
            #0, #E+/- A players expected +/-, based on his team and minutes played
            #20,  # PIM Penalties in minutes
            #70241,  # TOI
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #50,  # Ovrl Overall drafted
            #66,  # GP Games played
            #55,  # G Goals
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #2,  # DftRd Draft Round
            #5,  # A1
            #6,  # A2
            #7,  # PTS
            #8,  # =/-
            #9,  # E+/-
            #1990,  # DftYr Draft Year
            #2,  # DftRd Draft Round
            #2,  # DftRd Draft Round
            #1,
            #2,


        ]
        # scikit-learn assumes you want to predict the values for multiple players at once, so it expects an array.
        # We only want to estimate the value of a single player, so there will only be one item in our array.
        players = [
            player_1

        ]
        # Make a prediction for each player in the players array (we only have one)
        predicted_salary = model.predict(players)
        # Since we are only predicting the price of one player, grab the first prediction returned
        predicted_value = predicted_salary[0]
        # Print the results
        print("Player details:")
        print(f"- {player_1[0]} Draft Year")
        print(f"- {player_1[1]} Draft Rounds")
        print(f"- {player_1[2]} Overall Draft")
        print(f"- {player_1[3]} Games Played")
        print(f"Estimated Salary: ${predicted_value:,.2f}")
        # Print the results on the GUI
        T.configure(state='normal')
        #T.insert('end', f"- {player_1[0]} Draft Year", f"- {player_1[1]} Draft Rounds", f"- {player_1[2]} Overall Draft"
                 #, f"- {player_1[3]} Games Played", f"Estimated Salary: ${predicted_value:,.2f}")
        T.insert('end', "Player Details" + '\n')
        T.insert('end', f"- {player_1[0]} Draft Year" + '\n')
        T.insert('end', f"- {player_1[1]} Draft Rounds" + '\n')
        T.insert('end', f"- {player_1[2]} Overall Draft" + '\n')
        T.insert('end', f"- {player_1[3]} Games Played" + '\n')
        T.insert('end', f"Estimated Salary: ${predicted_value:,.2f}" + '\n')
        T.configure(state='disabled')


User1 = dataset()
User1.train()

root = Tk()
T = Text(root, state='disabled', height=10, width=30)
label_1 = Label(root, text="Player Last Name")
label_2 = Label(root, text="Player First Name")
label_3 = Label(root, text="Draft Year")
label_4 = Label(root, text="Draft Rounds")
label_5 = Label(root, text="Overall Draft")
label_6 = Label(root, text="Games Played")
label_7 = Label(root, text="Number of Goals")

last_name = Entry(root)
first_name = Entry(root)
draft_year = Entry(root)
draft_rounds = Entry(root)
overall_draft = Entry(root)
games_played = Entry(root)
num_goals = Entry(root)


#label_1.grid(row=0, sticky=E)
#label_2.grid(row=1, sticky=E)
label_3.grid(row=2, sticky=E)
label_4.grid(row=3, sticky=E)
label_5.grid(row=4, sticky=E)
label_6.grid(row=5, sticky=E)
label_7.grid(row=6, sticky=E)


#last_name.grid(row=0, column=1)
#first_name.grid(row=1, column=1)
draft_year.grid(row=2, column=1)
draft_rounds.grid(row=3, column=1)
overall_draft.grid(row=4, column=1)
games_played.grid(row=5, column=1)
num_goals.grid(row=6, column=1)
c = Button(root, text="Predict", command=User1.use)
c.grid(columnspan=1)
T.grid(row=7, column=1)

root.mainloop()
