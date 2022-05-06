import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper_functions.prepare_data import prepare_data

data = prepare_data(game_df=pd.read_csv("data/all_powerplays_4-23-22_cleaned_final.csv"))

sns.histplot(data=data, x = 'distance_to_attacking_net').set(title = "Distance to Attacking Net Distribution")
plt.show()

sns.histplot(data=data, x = 'all_avg_edge').set(title = "Avg Edges Distribution")
plt.show()