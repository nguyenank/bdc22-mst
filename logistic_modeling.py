from copy import copy
from turtle import pos
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import copy
import pandas as pd
import os
import csv
import sys
from hockey_rink import BDCRink
import random

conf_path = os.getcwd()

sys.path.append(conf_path)

from hockey_mst import (
    mst_properties,
    ind_var_calculation
)

#display stuff
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)

game_df = pd.read_csv("all_powerplays_4-13-22_danger_situations_ozone.csv")

def prepare_data(game_df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """taking a raw game of data and turning it into what we want for the prep"""
    game_df[["O Players","D Players","All MST","All_Avg_Edge","All_Total_Edge","All_Avg_Edges per Player","O MST", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player","D MST", "D_Avg_Edge","D_Total_Edge","D_Avg_Edges per Player", "OD_MST_Ratio", "All_OCR"]] = None

    x_cols = ['away_x_1', 'away_x_2','away_x_3', 'away_x_4','away_x_5', 'away_x_6','away_x_7','home_x_1', 'home_x_2','home_x_3', 'home_x_4','home_x_5', 'home_x_6','home_x_7']
    y_cols = ['away_y_1', 'away_y_2','away_y_3', 'away_y_4','away_y_5', 'away_y_6','away_y_7','home_y_1', 'home_y_2','home_y_3', 'home_y_4','home_y_5', 'home_y_6','home_y_7']
    positions = ['away_position_1', 'away_position_2','away_position_3', 'away_position_4','away_position_5', 'away_position_6','away_position_7','home_position_1', 'home_position_2','home_position_3', 'home_position_4','home_position_5', 'home_position_6','home_position_7']
    vars = ["high_danger_within_four",
            "distance_to_attacking_net", 
            "All_Avg_Edge", 
            "All_Total_Edge",
            "O_Avg_Edge",
            "O_Total_Edge",
            "O_Avg_Edges_per_Player", 
            "D_Avg_Edge",
            "D_Total_Edge",
            "OD_MST_Ratio", 
            "All_OCR"]

    df = copy.deepcopy(game_df)

    # print(copy_df[['x_coord','away_x_1', 'away_x_2','away_x_3', 'away_x_4','away_x_5', 'away_x_6','away_x_7','home_x_1', 'home_x_2','home_x_3', 'home_x_4','home_x_5', 'home_x_6','home_x_7']].iloc[358])
    df.loc[(df['period'] == 2), 'y_coord'] = (-(game_df.loc[(df['period'] == 2), 'y_coord'] - (85 / 2)) + (85 / 2))
    df.loc[(df['period'] == 2), y_cols] = (-(game_df.loc[(df['period'] == 2), y_cols]-(85 / 2)) + (85/2))
    df.loc[(df['period'] == 2), 'x_coord'] = (-(game_df.loc[(df['period'] == 2), 'x_coord']- (200/2)) + (200/2))
    df.loc[(df['period'] == 2), x_cols] = (-(game_df.loc[(df['period'] == 2), x_cols]- (200/2)) + (200/2))

    df_w_ind_vars = ind_var_calculation(df, x_cols=x_cols, y_cols=y_cols, positions=positions)

    ind_vars = copy.deepcopy(vars)
    ind_vars.remove('high_danger_within_four')

    df_non_na = df_w_ind_vars.fillna(value = np.nan).dropna(subset = vars)
    df_no_na = df_non_na[vars]
    # X = df_no_na[ind_vars].reset_index().drop(columns = 'index')
    # y = df_no_na['high_danger_within_four']

    return df_no_na

# re-sampling option: https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html
# another good option: https://towardsdatascience.com/bootstrap-resampling-2b453bb036ec

game_df = prepare_data(game_df=game_df)

def data_partition(game_df, prop = 0.4):

    vars = ["high_danger_within_four",
        "distance_to_attacking_net", 
        "All_Avg_Edge", 
        "All_Total_Edge",
        "O_Avg_Edge",
        "O_Total_Edge",
        "O_Avg_Edges_per_Player", 
        "D_Avg_Edge",
        "D_Total_Edge",
        "OD_MST_Ratio", 
        "All_OCR"]

    no = len(game_df[game_df.high_danger_within_four == 0])
    yes = len(game_df[game_df.high_danger_within_four == 1])

    samp_from = game_df[game_df.high_danger_within_four == 1]
    other = game_df[game_df.high_danger_within_four == 0]

    goal = round((prop * no) / (1 - prop))

    new_samples = pd.DataFrame(columns=vars)

    random.seed(1423)

    for i in range(1, (goal + 1) - yes):

        s = samp_from.sample()
        new_samples = new_samples.append(s, ignore_index=True)

        # print(i)

    # plt.hist(x = new_samples.O_Total_Edge)
    # plt.show()

    # plt.hist(x = samp_from.O_Total_Edge)
    # plt.show()
    data = samp_from.append(new_samples, ignore_index=True)
    data = data.append(other, ignore_index=True).sample(frac = 1).sample(frac = 1).reset_index(drop = True)

    X = data.drop(columns = ['high_danger_within_four'])
    y = data[['high_danger_within_four']]

    return X, y

X, y = data_partition(game_df=game_df, prop=0.425)

train_x, train_y, test_x, test_y = train_test_split(X, y, test_size=0.2, random_state=366)

#applying logistic model to training data
model1_log = LogisticRegression(solver='liblinear', max_iter=10000, random_state=43)
model1_log.fit(train_x, test_x)
