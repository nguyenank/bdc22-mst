"""
function to split dataset"""

from copy import copy
import copy

def split_data(game_df):
    """
    splits data into X, y sets"""
    
    vars = ['high_danger_within_four', 
            'distance_to_attacking_net', 
            'all_avg_edge',
            'all_total_edge', 
            'o_avg_edge', 
            'o_total_edge',
            'o_avg_edges_per_player', 
            'd_avg_edge', 
            'd_total_edge', 
            'od_mst_ratio',
            'all_ocr', 
            'angle_to_attacking_net']

    ind_vars = copy.deepcopy(vars) #["distance_to_attacking_net","All_Avg_Edge", "O_Avg_Edge","O_Total_Edge","O_Avg_Edges_per_Player", "D_Avg_Edge", "D_Total_Edge", "OD_MST_Ratio", "All_OCR"]
    ind_vars.remove("high_danger_within_four")

    X = game_df[ind_vars].reset_index().drop(columns = ['index'])
    y = game_df['high_danger_within_four'].astype(int)

    return X, y