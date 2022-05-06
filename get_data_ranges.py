import pandas as pd
from prepare_data import prepare_data

game_df = prepare_data(game_df=pd.read_csv("all_powerplays_4-23-22_cleaned_final.csv"))

maxes = pd.DataFrame(game_df.drop(columns = ['high_danger_within_four']).max(axis = 0)).T
mins = pd.DataFrame(game_df.drop(columns = ['high_danger_within_four']).min(axis = 0)).T

output = []

maxes['type'] = 'maximum'
mins['type'] = 'minimum'

output.append(mins)
output.append(maxes)

output = pd.concat(output)
output.to_csv(path_or_buf='variable_ranges/variable_ranges.csv', index=False)