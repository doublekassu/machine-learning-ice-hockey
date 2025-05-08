import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from pathlib import Path
import os

#Training data
def load_data():
    #base_path = r"C:\Users\Samuli\NHL_ML\machine-learning-ice-hockey\Skaters"
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / "Skaters"
    
    df_2019 = pd.read_csv(base_path / "skaters_2019.csv")
    df_2020 = pd.read_csv(base_path / "skaters_2020.csv")
    df_2021 = pd.read_csv(base_path / "skaters_2021.csv")
    df_2022 = pd.read_csv(base_path / "skaters_2022.csv")

    df_train = pd.concat([df_2019, df_2020, df_2021, df_2022,]) #Combine data
    df_train= df_train[df_train["situation"] == "all"] #Select all types

    #Testing data
    df_2023 = pd.read_csv(base_path / "skaters_2023.csv")
    df_2023= df_2023[df_2023["situation"] == "all"]
    
    print("Sarakkeet:", df_train.columns.tolist())
    
    return df_train, df_2023




def train_model(df_train):
    # Define features and target
    features = [
        "I_F_xGoals", "I_F_shotsOnGoal", "I_F_rebounds",
        "I_F_highDangerxGoals", "I_F_mediumDangerxGoals",
        "I_F_lowDangerxGoals", "I_F_shotAttempts"
    ]
    target = "I_F_goals"
    
    # Training data
    X_train = df_train[features]
    y_train = df_train[target]
    
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features, target


def find_player_by_id(df, pelaaja_id):
    player_row = df[df["playerId"] == pelaaja_id]
    return player_row

def evaluate_model(model, features, df_test):
    """
    Evaluate the model on test data and return accuracy metrics
    """
    X_test = df_test[features]
    y_test = df_test["I_F_goals"]
    
    y_pred = model.predict(X_test)
    r2 = 100*r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return r2, rmse, y_test.mean(), y_pred.mean()

def predict_player_goals(model, features, df_train, df_2023, player_id):
    """
    Predict goals for a specific player and compare with actual results
    """
    player_data_from_train = df_train[df_train["playerId"] == player_id]
    if player_data_from_train.empty:

        return None, None, None, None, "Player did not play in the training seasons."
    
    # Find player in 2023 data
    player_row_2023 = df_2023[df_2023["playerId"] == player_id]
    
    if player_row_2023.empty:
        return None, None, None, None, "Player not found in 2023 data"
    
    player_name = player_row_2023["name"].values[0]
    actual_goals = player_row_2023["I_F_goals"].values[0]
    
    # Get features for prediction
    player_features = player_row_2023[features]
    
    # Make prediction
    predicted_goals = model.predict(player_features)[0]
    
    
    
    # Calculate accuracy
    if actual_goals > 0:
        accuracy = 100 - abs(predicted_goals - actual_goals) / actual_goals * 100
    else:
        accuracy = 0
    
    return player_name, predicted_goals, actual_goals, accuracy, None