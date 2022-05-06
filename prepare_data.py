"""
function to prepare data for modeling etc"""

from copy import copy
import os
import sys
import pandas as pd
import numpy as np
import copy

conf_path = os.getcwd()

sys.path.append(conf_path)

from hockey_mst import (
    mst_properties,
    ind_var_calculation
)

def prepare_data(game_df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """taking a raw game of data and turning it into what we want for the prep
    takes a game dataframe as the input and returns the edited dataframe"""
    game_df[["O Players","D Players","All MST","All_Avg_Edge","All_Total_Edge","All_Avg_Edges per Player","O MST", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player","D MST", "D_Avg_Edge","D_Total_Edge","D_Avg_Edges per Player", "OD_MST_Ratio", "All_OCR"]] = None
    # gonna want to comment this out lmao
    # game_df['angle'] = 1
    game_df['angle_to_attacking_net'] = game_df['angle_to_attacking_net'] + 180

    if 'assumed_danger_states' in game_df.columns:
        game_df['high_danger_within_four'] = game_df['assumed_danger_states']
        # print("Big Ballin'")

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
            "All_OCR",
            'angle_to_attacking_net']

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

    df_no_na.columns = df_no_na.columns.str.strip().str.lower()
    # X = df_no_na[ind_vars].reset_index().drop(columns = 'index')
    # y = df_no_na['high_danger_within_four']

    return df_no_na