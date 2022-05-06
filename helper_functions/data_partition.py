import copy
import pandas as pd
import random

def data_partition(game_df, type = "over", prop = 0.4):

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

    ind_vars = copy.deepcopy(vars) #["distance_to_attacking_net","All_Avg_Edge", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge", "D_Total_Edge", "OD_MST_Ratio", "All_OCR"]
    ind_vars.remove("high_danger_within_four")

    no = len(game_df[game_df.high_danger_within_four == 0])
    yes = len(game_df[game_df.high_danger_within_four == 1])

    new_samples = pd.DataFrame(columns=vars)

    if type == "over":
        samp_from = game_df[game_df.high_danger_within_four == 1]
        other = game_df[game_df.high_danger_within_four == 0]
        
        goal = round((prop * no) / (1 - prop))
        random.seed(1423)

        for i in range(1, (goal + 1) - yes):

            s = samp_from.sample()
            new_samples = new_samples.append(s, ignore_index=True)

        # print(i)
    elif type == "under":
        samp_from = game_df[game_df.high_danger_within_four == 0]
        other = game_df[game_df.high_danger_within_four == 1]

        goal = round((yes / prop) - yes)
        random.seed(1234)

        for i in range(1, (goal + 1)):
            s = samp_from.sample()
            new_samples = new_samples.append(s, ignore_index=True)

    # plt.hist(x = new_samples.O_Total_Edge)
    # plt.show()

    # plt.hist(x = samp_from.O_Total_Edge)
    # plt.show()
    data = samp_from.append(new_samples, ignore_index=True)
    data = data.append(other, ignore_index=True).sample(frac = 1).sample(frac = 1).reset_index(drop = True)

    # X = data[ind_vars].reset_index().drop(columns = ['index'])
    # y = data['high_danger_within_four'].astype(int)

    return data

def resample_data(X, Y, prop = 0.45):
    data = pd.DataFrame(Y).reset_index(drop=True).join(pd.DataFrame(X))

    no = len(data[data.high_danger_within_four == 0])
    yes = len(data[data.high_danger_within_four == 1])

    samp_from = data[data.high_danger_within_four == 1]
    other = data[data.high_danger_within_four == 0]

    goal = round((prop * no) / (1 - prop))

    new_samples = pd.DataFrame()
    random.seed(1324)

    for i in range(1, (goal + 1) - yes):
        s = samp_from.sample()
        new_samples = new_samples.append(s)

    data = samp_from.append(new_samples, ignore_index=True)
    data = data.append(other, ignore_index=True).sample(frac = 1).sample(frac = 1).reset_index(drop = True)

    X = data.drop(columns=['high_danger_within_four'])
    Y = data[['high_danger_within_four']]

    return X, Y
