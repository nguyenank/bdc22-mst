import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('all_powerplays_4-23-22_cleaned_trimmed.csv') #24 of 35 powerplays included, replace with 4-21 data asap

# data.game_date + data.team_name + data.opp

# frame_diff = np.diff(data['frame_id'])
def calculate_velocities(data: pd.DataFrame) -> pd.DataFrame:
    """
    Is this code bad? Yes.
    Does it calculate X/Y player velos? Mostly?
    This is my understanding of what we're looking for from the velocities on our projects data
    """
    data['clock_diff'] = np.insert(np.diff(data['clock_seconds']), 0, 0, axis=0)
    # assuming that if it's greater than 30 seconds between events, it's two separate games
    # not the best strategy but hey, it's late 
    data.clock_diff = np.where(data.clock_diff.abs() > 30, np.nan, data.clock_diff)

    data['home_x_1_diff'] = np.insert(np.diff(data.home_x_1), 0, np.nan, axis=0)
    data['home_x_2_diff'] = np.insert(np.diff(data.home_x_2), 0, np.nan, axis=0)
    data['home_x_3_diff'] = np.insert(np.diff(data.home_x_3), 0, np.nan, axis=0)
    data['home_x_4_diff'] = np.insert(np.diff(data.home_x_4), 0, np.nan, axis=0)
    data['home_x_5_diff'] = np.insert(np.diff(data.home_x_5), 0, np.nan, axis=0)
    data['home_x_6_diff'] = np.insert(np.diff(data.home_x_6), 0, np.nan, axis=0)

    data['home_y_1_diff'] = np.insert(np.diff(data.home_y_1), 0, np.nan, axis=0)
    data['home_y_2_diff'] = np.insert(np.diff(data.home_y_2), 0, np.nan, axis=0)
    data['home_y_3_diff'] = np.insert(np.diff(data.home_y_3), 0, np.nan, axis=0)
    data['home_y_4_diff'] = np.insert(np.diff(data.home_y_4), 0, np.nan, axis=0)
    data['home_y_5_diff'] = np.insert(np.diff(data.home_y_5), 0, np.nan, axis=0)
    data['home_y_6_diff'] = np.insert(np.diff(data.home_y_6), 0, np.nan, axis=0)

    data['away_x_1_diff'] = np.insert(np.diff(data.away_x_1), 0, np.nan, axis=0)
    data['away_x_2_diff'] = np.insert(np.diff(data.away_x_2), 0, np.nan, axis=0)
    data['away_x_3_diff'] = np.insert(np.diff(data.away_x_3), 0, np.nan, axis=0)
    data['away_x_4_diff'] = np.insert(np.diff(data.away_x_4), 0, np.nan, axis=0)
    data['away_x_5_diff'] = np.insert(np.diff(data.away_x_5), 0, np.nan, axis=0)
    data['away_x_6_diff'] = np.insert(np.diff(data.away_x_6), 0, np.nan, axis=0)

    data['away_y_1_diff'] = np.insert(np.diff(data.away_y_1), 0, np.nan, axis=0)
    data['away_y_2_diff'] = np.insert(np.diff(data.away_y_2), 0, np.nan, axis=0)
    data['away_y_3_diff'] = np.insert(np.diff(data.away_y_3), 0, np.nan, axis=0)
    data['away_y_4_diff'] = np.insert(np.diff(data.away_y_4), 0, np.nan, axis=0)
    data['away_y_5_diff'] = np.insert(np.diff(data.away_y_5), 0, np.nan, axis=0)
    data['away_y_6_diff'] = np.insert(np.diff(data.away_y_6), 0, np.nan, axis=0)

    data['home_x_1_velo'] = data.home_x_1_diff / data.clock_diff
    data['home_x_2_velo'] = data.home_x_2_diff / data.clock_diff
    data['home_x_3_velo'] = data.home_x_3_diff / data.clock_diff
    data['home_x_4_velo'] = data.home_x_4_diff / data.clock_diff
    data['home_x_5_velo'] = data.home_x_5_diff / data.clock_diff
    data['home_x_6_velo'] = data.home_x_6_diff / data.clock_diff

    data['home_y_1_velo'] = data.home_y_1_diff / data.clock_diff
    data['home_y_2_velo'] = data.home_y_2_diff / data.clock_diff
    data['home_y_3_velo'] = data.home_y_3_diff / data.clock_diff
    data['home_y_4_velo'] = data.home_y_4_diff / data.clock_diff
    data['home_y_5_velo'] = data.home_y_5_diff / data.clock_diff
    data['home_y_6_velo'] = data.home_y_6_diff / data.clock_diff

    data['away_x_1_velo'] = data.away_x_1_diff / data.clock_diff
    data['away_x_2_velo'] = data.away_x_2_diff / data.clock_diff
    data['away_x_3_velo'] = data.away_x_3_diff / data.clock_diff
    data['away_x_4_velo'] = data.away_x_4_diff / data.clock_diff
    data['away_x_5_velo'] = data.away_x_5_diff / data.clock_diff
    data['away_x_6_velo'] = data.away_x_6_diff / data.clock_diff

    data['away_y_1_velo'] = data.away_y_1_diff / data.clock_diff
    data['away_y_2_velo'] = data.away_y_2_diff / data.clock_diff
    data['away_y_3_velo'] = data.away_y_3_diff / data.clock_diff
    data['away_y_4_velo'] = data.away_y_4_diff / data.clock_diff
    data['away_y_5_velo'] = data.away_y_5_diff / data.clock_diff
    data['away_y_6_velo'] = data.away_y_6_diff / data.clock_diff

    data.loc[:, data.columns.str.contains("_velo")]

    return data