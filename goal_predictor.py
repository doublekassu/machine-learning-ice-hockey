import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

#Training data
def load_data():
    base_path = r"C:\Users\Samuli\NHL_ML\machine-learning-ice-hockey\Skaters"
    
    df_2019 = pd.read_csv(os.path.join(base_path, "skaters_2019.csv"))
    df_2020 = pd.read_csv(os.path.join(base_path, "skaters_2020.csv"))
    df_2021 = pd.read_csv(os.path.join(base_path, "skaters_2021.csv"))
    df_2022 = pd.read_csv(os.path.join(base_path, "skaters_2022.csv"))

    df_train = pd.concat([df_2019, df_2020, df_2021, df_2022,]) #Combine data
    df_train= df_train[df_train["situation"] == "all"] #Select all types

    #Testing data
    df_2023 = pd.read_csv(os.path.join(base_path, "skaters_2023.csv"))
    df_2023= df_2023[df_2023["situation"] == "all"]
    
    print("Sarakkeet:", df_train.columns.tolist())
    
    return df_train, df_2023




def train_model(df_train,df_2023):
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

#def find_player_by_id(df_2023, pelaaja_id):
 #   player_row = df_2023[df_2023["playerId"] == pelaaja_id]
  #  return player_row

def find_player_by_id(df_train, pelaaja_id):
    player_row = df_train[df_train["playerId"] == pelaaja_id]
    return player_row

