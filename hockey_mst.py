from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sb
import copy
from tqdm import tqdm

#defunct grid binning
#player_control_area = 5 #feet radius
#bins = np.mgrid[0:201:5,0:86:5].T.flatten()
#grid = bins.reshape(int(len(bins)/2),2)
#for bin in range(len(grid)-1):
#    print(grid[bin][0],grid[bin][1])
#    print(bin)
#    df.loc[(((df['x_coord'] < grid[bin][0]+player_control_area) & (df['x_coord'] >= grid[bin][0])) & ((df['y_coord'] < grid[bin][1]+player_control_area) & (df['y_coord'] >= grid[bin][1]))),"Grid Position"] = bin

#calculating most MST properties
def mst_properties(player_positions, player_teams=None):
    no_players = len(player_positions) #number of players

    #MST calculation
    p2p_distances = np.empty(shape=(no_players, no_players))
    coordinates = player_positions.T
    for i in range(no_players):
        p2p_distances[i] = np.sqrt((coordinates[0]-player_positions[i][0])**2 + (coordinates[1]-player_positions[i][1])**2)
    tree = minimum_spanning_tree(csr_matrix(p2p_distances)).toarray()

    #avg edge length and total edge length
    avg_edge_length = np.mean(tree[~(tree == 0)])
    tot_edge_length = np.sum(tree[~(tree == 0)])
    plotted = np.argwhere(tree > 0)

    #counting the number of edges each player has and taking the average of that array ot get avg_edges_per_player
    edge_player_connections = []
    for i in plotted:
        j,k = i
        edge_player_connections.append(j)
        edge_player_connections.append(k)
    unique_vals, edge_player_connections = np.unique(np.array(edge_player_connections),return_counts=True)
    avg_edges_per_player = np.mean(edge_player_connections)

    #opponent connection ratio calculation: assign each MST edge a 0 (teamate to teamate connection) or a 1 (opponent to opponent connection) and take the mean of the assigned MST values
    #closer to 1 means more people are paired up with opponents and closer to 0 means paired up with teamates
    if player_teams is not None:
        pairing_list = []
        for i in plotted:
            j,k = i
            if player_teams[j] == player_teams[k]:
                pairing_list.append(0)
            else:
                pairing_list.append(1)
        opponent_connection_ratio = np.mean(np.array(pairing_list))

        return tree, avg_edge_length, tot_edge_length, avg_edges_per_player, opponent_connection_ratio

    else:
        return tree, avg_edge_length, tot_edge_length, avg_edges_per_player

def ind_var_calculation(df, x_cols, y_cols, positions):
    #MST variable calculations
    for i in range(len(df)):
        x_coords = df[x_cols].iloc[i]
        y_coords = df[y_cols].iloc[i]

        #for 5 on 4 events  where we have tracking data proceed with MST calculations
        event_team_strength = re.search('(\w) on (\w)',df['situation_type'].iloc[i]).group(1)
        opp_team_strength = re.search('(\w) on (\w)',df['situation_type'].iloc[i]).group(2)
        if ~((x_coords[:7].isnull().all()) | (x_coords[7:].isnull().all())) & (x_coords[:7].count() > 1) & (x_coords[7:].count() > 1):
            # print(str(i)+"/"+str(len(df)))

            #getting coordinates, positions, and respective teams (last one if for OCR calculation) for event
            raw_coord_pairs = np.array([x_coords,y_coords]).T
            player_raw_teams = np.empty(len(raw_coord_pairs))
            player_raw_teams[:7].fill(0) #one is away
            player_raw_teams[7:].fill(1) #two is home
            player_role = df[positions].iloc[i]

            #MST calculation
            #getting rid of goalies for calculation of all player MST
            raw_all_coord_pairs = raw_coord_pairs[~(player_role == "Goalie")]
            player_teams = player_raw_teams[~(player_role == "Goalie")]
            all_coord_pairs = raw_all_coord_pairs[~np.isnan(raw_all_coord_pairs)]
            player_teams = player_teams[~(np.isnan(raw_all_coord_pairs)[:,0])]
            all_coord_pairs = all_coord_pairs.reshape(int(len(all_coord_pairs)/2),2)

            #variable calculations for MST properties with all players
            df['All MST'].iloc[i],df['All_Avg_Edge'].iloc[i], df['All_Total_Edge'].iloc[i], df["All_Avg_Edges per Player"].iloc[i], df["All_OCR"].iloc[i] = mst_properties(all_coord_pairs, player_teams)


            #variable calculations for 2 MSTs: one with with offensive players and one with defensive players
            if df['venue'].iloc[i] == 'home': #home team is the offensive team
                #excluding goalie and empty coordinate spots
                raw_home_coord_pairs = raw_coord_pairs[7:][~(player_role[7:] == "Goalie")]
                home_coord_pairs = raw_home_coord_pairs[~np.isnan(raw_home_coord_pairs)]
                home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
                df['O Players'].iloc[i] = len(home_coord_pairs)
                df['O MST'].iloc[i],df['O_Avg_Edge'].iloc[i], df['O_Total_Edge'].iloc[i], df["O_Avg_Edges_per_Player"].iloc[i] = mst_properties(home_coord_pairs)

                #leaving goalie in for defensive team because it matters to the model
                raw_away_coord_pairs = raw_coord_pairs[:7]#[~(player_role[7:] == "Goalie")]
                away_coord_pairs = raw_away_coord_pairs[~np.isnan(raw_away_coord_pairs)]
                away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
                df['D Players'].iloc[i] = len(away_coord_pairs)
                df['D MST'].iloc[i],df['D_Avg_Edge'].iloc[i], df['D_Total_Edge'].iloc[i], df["D_Avg_Edges per Player"].iloc[i] = mst_properties(away_coord_pairs)
            elif df['venue'].iloc[i] == 'away': #away is offensive team
                #leaving goalie in for defensive team because it matters to the model
                raw_home_coord_pairs = raw_coord_pairs[7:]#[~(player_role[7:] == "Goalie")]
                home_coord_pairs = raw_home_coord_pairs[~np.isnan(raw_home_coord_pairs)]
                home_coord_pairs = home_coord_pairs.reshape(int(len(home_coord_pairs)/2),2)
                df['D Players'].iloc[i] = len(home_coord_pairs)
                df['D MST'].iloc[i],df['D_Avg_Edge'].iloc[i], df['D_Total_Edge'].iloc[i], df["D_Avg_Edges per Player"].iloc[i] = mst_properties(home_coord_pairs)

                #excluding goalie and empty coordinate spots
                raw_away_coord_pairs = raw_coord_pairs[:7][~(player_role[7:] == "Goalie")]
                away_coord_pairs = raw_away_coord_pairs[~np.isnan(raw_away_coord_pairs)]
                away_coord_pairs = away_coord_pairs.reshape(int(len(away_coord_pairs)/2),2)
                df['O Players'].iloc[i] = len(away_coord_pairs)
                df['O MST'].iloc[i],df['O_Avg_Edge'].iloc[i], df['O_Total_Edge'].iloc[i], df["O_Avg_Edges_per_Player"].iloc[i] = mst_properties(away_coord_pairs)
            #calculating MST ratio betweeen offensive and defense average edge length
            df["OD_MST_Ratio"].iloc[i] = df['O_Avg_Edge'].iloc[i]/df['D_Avg_Edge'].iloc[i]

    #deep copy of dataframe to return correctly
    df_copy = copy.deepcopy(df)
    return df_copy
