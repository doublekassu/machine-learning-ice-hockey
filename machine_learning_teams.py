import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def machine_learning(training_season_start, training_season_end, test_season_start, test_season_end):
    
    matches = pd.read_csv("all_teams.csv", index_col=0)

    #Mark home and away teams to integers, 0 for away and 1 for home
    matches["home_or_away_code"] = matches["home_or_away"].astype("category").cat.codes
    #Mark home team and opposingteam with a unique ID
    matches["opp_code"] = matches["opposingTeam"].astype("category").cat.codes
    matches["hometeam_code"] = matches["playerTeam"].astype("category").cat.codes


    #Drop rows which situation column isn't "all" to make analyzing easier
    matches = matches[matches["situation"] == "all"]
    
    #Set a winner column. When the team has scored more than the opponent, set it to 1
    matches["winner"] = (matches["goalsFor"] > matches["goalsAgainst"]).astype(int)

    #Use RandomForestClassifier as the algorithm to recognize non-linear patterns in the data
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    def set_season_years(season_start_year, season_end_year, data):
        start_season_date = "0901"
        end_season_date = "0901"

        season_start_date_str = str(season_start_year) + start_season_date
        season_start_date_int = int(season_start_date_str)

        season_end_date_str = str(season_end_year) + end_season_date
        season_end_date_int = int(season_end_date_str)

        dataset = data[(data["gameDate"] >= season_start_date_int) & (data["gameDate"] < season_end_date_int)]
        return dataset

    #Train set is seasons 2008-2023
    train = set_season_years(training_season_start, training_season_end, matches)

    #Test set is 2023-2024
    test = set_season_years(test_season_start, test_season_end, matches)

    #Make predictions based on if the game is a playoff game, if it's a home or away game and the teams playing

    predictors = ["playoffGame", "home_or_away_code", "opp_code", "hometeam_code"]

    #Train the algorithm to predict the outcome of the match
    rf.fit(train[predictors], train["winner"])

    predictions = rf.predict(test[predictors])
    from sklearn.metrics import accuracy_score
    #The prediction algorithm was 53.54% accurate predicting the outcome of the match. 51.13% accurate predicting the winner

    default_attributes_acc = accuracy_score(test["winner"], predictions)
    print("Accuracy of predicting the outcome with " , predictors, " as the predictors: ", round(default_attributes_acc, 4))

    print("\nAccuracy of predicting the winner ", round(precision_score(test["winner"], predictions), 4))

    #Group all teams. Start making predictions also depending on each team's recent results

    #Sort by game date, take the average of the last 3 rows and assign it to the current row
    #Drop NaN values (can't have an average of last 3 games if there have been no games before)
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("gameDate")
        rolling_stats = group[cols].rolling(3, closed="left").mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset = new_cols)
        return group

    #The avg of these attributes in the last 3 games are calculated and used in next predictions
    cols = ["goalsFor", "goalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst", "penalityMinutesFor", "penalityMinutesAgainst", "shotAttemptsFor", "shotAttemptsAgainst"]
    new_cols = [f"{c}_rolling" for c in cols]

    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel(0)

    #Set inedxes to each game from both team's perspective
    matches_rolling["index"] = range(matches_rolling.shape[0])
    matches_rolling = matches_rolling.set_index("index")

    #Function to make testing, training and analyzing data easier
    def make_predictions(data, predictors):
        train = set_season_years(training_season_start, training_season_end, data)
        test = set_season_years(test_season_start, test_season_end, data)
        rf.fit(train[predictors], train["winner"])
        predictions = rf.predict(test[predictors])
        combined = pd.DataFrame(dict(actual=test["winner"], prediction=predictions), index = test.index)
        precision = precision_score(test["winner"], predictions)
        return combined, precision

    combined, precision = make_predictions(matches_rolling, predictors + new_cols)
    print("\nAccuracy of predicting the winner with the average attributes of the last 3 games", round(precision, 4))

    #Accuracy of when the model predicted the other team to win and the other to lose (model being the most sure of the outcome)
    combined = combined.merge(matches_rolling[["gameDate", "playerTeam", "opposingTeam", "winner"]], left_index=True, right_index=True)
    merged = combined.merge(combined, left_on=["gameDate", "playerTeam"], right_on=["gameDate", "opposingTeam"])

    counts = merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)]["actual_x"].value_counts()

    extra_attributes_acc = counts.get(1, 0) / (counts.get(1,0) + counts.get(0, 0))
    print("\nAccuracy of predicting the winner and loser on a game ", round(extra_attributes_acc, 4))

    return round(default_attributes_acc*100, 2), round(extra_attributes_acc*100, 2)