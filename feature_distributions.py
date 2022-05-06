import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper_functions.prepare_data import prepare_data

data = prepare_data(game_df=pd.read_csv("data/all_powerplays_4-23-22_cleaned_final.csv"))

sns.histplot(data=data, x = 'distance_to_attacking_net').set(title = "Distance to Attacking Net Distribution")
plt.savefig('distribution_plots/distance_net.png')
plt.clf()

sns.histplot(data=data, x = 'all_avg_edge').set(title = "Avg Edges Distribution")
plt.savefig('distribution_plots/all_avg_edge.png')
plt.clf()
# some outliers at 25+?

sns.histplot(data=data, x = 'all_total_edge').set(title = "Total Edges Distribution")
plt.savefig('distribution_plots/all_total_edge.png')
plt.clf()

sns.histplot(data=data, x = 'o_avg_edge').set(title = "Avg Edges on Offense Distribution")
plt.savefig('distribution_plots/offense_avg_edge.png')
plt.clf()
# maybe a few outliers around 5ish?

sns.histplot(data=data, x = 'o_total_edge').set(title = "Offense Total Edges Distribution")
plt.savefig('distribution_plots/offense_total_edge.png')
plt.clf()

sns.histplot(data=data, x = 'o_avg_edges_per_player').set(title = "Avg Offense Edges per Player Distribution")
plt.savefig('distribution_plots/avg_offense_edge_player.png')
plt.clf()
# meh

sns.histplot(data=data, x = 'd_avg_edge').set(title = "Avg Defense Edges Distribution")
plt.savefig('distribution_plots/defense_avg_edge.png')
plt.clf()

sns.histplot(data=data, x = 'd_total_edge').set(title = "Defense Total Edges Distribution")
plt.savefig('distribution_plots/defense_total_edge.png')
plt.clf()

sns.histplot(data=data, x = 'od_mst_ratio').set(title = "Offense/Defense MST Ratio Distribution")
plt.savefig('distribution_plots/off_def_mst_ratio.png')
plt.clf()

sns.histplot(data=data, x = 'all_ocr').set(title = "Opponent Connection Distribution")
plt.savefig('distribution_plots/avg_offense_edge_player.png')
plt.clf()

sns.histplot(data=data, x = 'angle_to_attacking_net').set(title = "Angle to Net Distribution")
plt.savefig('distribution_plots/angle_net.png')
plt.clf()