import pandas as pd

game_df=pd.read_csv("all_powerplays_4-23-22_cleaned_final.csv")

vars = ["high_danger_within_four",
        "distance_to_net", 
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

game_df[vars]