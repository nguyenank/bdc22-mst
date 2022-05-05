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
import hockey_mst # applies the MST calculations to a given dataframe
from prepare_data import prepare_data # function that preps the dataframe for the hockey_mst function
from split_data import split_data # splits data into variables + target
from data_partition import resample_data

plt.rcParams["font.family"] = "Consolas"
pd.set_option('precision', 5)