"""Importing libraries and functions used for modeling purposes"""
import json
import copy

# Data Manipulation Packages
import pandas as pd 
import numpy as np

# Plotting Packages
import matplotlib.pyplot as plt
import seaborn as sn
from hockey_rink import BDCRink

# Modeling Packages/Functions
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest,
    chi2
)
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc,
    PrecisionRecallDisplay
)

# Custom-built functions
import hockey_mst
from modelling_and_plotting import X_w_inter # applies the MST calculations to a given dataframe
from prepare_data import prepare_data # function that preps the dataframe for the hockey_mst function
from split_data import split_data # splits data into variables + target
from data_partition import resample_data # re-sample to bring 0/1 classes to roughly even (or preferred proportion)

# editing display options
plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)

game_df = prepare_data(game_df=pd.read_csv("all_powerplays_4-23-22_cleaned_updated.csv"))

x, y = split_data(game_df=game_df)

print(len(y[y == 1]), " successes", sep="")
print(len(y[y == 0]), " not successes", sep="")

# def get_interactions(x): 
interactions = PolynomialFeatures(interaction_only=True, include_bias=True)
x_w_inter = interactions.fit_transform(X=x)
inter_vars_raw = interactions.get_feature_names_out()

new_names = []

for i in inter_vars_raw:
    n = i.replace(' ', '_')
    new_names.append(n)

    # return x_w_inter, new_names

# x_w_inter, new_names = get_interactions(x = x)
# Carlie, run the code above and this line below and it shows that there are negative values
pd.set_option('display.max_rows', 75)
(x_w_inter < 0).any()
x_df = pd.DataFrame(x_w_inter, columns=new_names)
(x_df < 0).any(0)
# (x_w_inter < 0).any(0)

def variable_selection(x_w_inter, y, new_names, k = 40): 
    selection = SelectKBest(chi2, k = k)
    trans_x = selection.fit_transform(x_w_inter, y)